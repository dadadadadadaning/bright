"""Restore the bound Q numerical profile without importing or initializing CUDA.

This module uses only the standard library. Environment restoration precedes
Torch/NumPy imports; the original fresh_source loader retains its full gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
OVERLAP = {"torch_version": "torch_version", "cuda_version": "cuda_version",
           "deterministic_algorithms": "deterministic_algorithms",
           "cudnn_allow_tf32": "cudnn_allow_tf32", "matmul_allow_tf32": "cuda_matmul_allow_tf32",
           "float32_matmul_precision": "float32_matmul_precision",
           "cublas_workspace_config": "cublas_workspace_config"}
BOOL_FIELDS = ("deterministic_algorithms", "deterministic_warn_only", "cudnn_deterministic",
               "cudnn_benchmark", "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_enabled",
               "fp16_reduced_precision_reduction", "bf16_reduced_precision_reduction")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "Expected JSON object: " + str(path))
    return value


def differences(expected, actual, prefix=""):
    """Field-wise evidence, never a replacement for original exact equality."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        rows = []
        for key in sorted(set(expected) | set(actual)):
            field = prefix + ("." if prefix else "") + key
            if key not in expected or key not in actual:
                rows.append(dict(field=field, expected=expected.get(key), actual=actual.get(key),
                                 missing_from="expected" if key not in expected else "actual"))
            else:
                rows.extend(differences(expected[key], actual[key], field))
        return rows
    return [] if expected == actual else [dict(field=prefix, expected=expected, actual=actual)]


def read_bound_profile(root, step):
    require(type(step) is int and 1 <= step <= 5000, "Checkpoint step must be 1..5000")
    root = Path(root).expanduser().resolve(strict=True)
    cp = root / "train/checkpoints" / ("checkpoint_last.pt" if step == 5000 else "checkpoint_%06d.pt" % step)
    side_path = cp.with_suffix(".pt.json")
    manifest_path = root / "train/manifest.json"
    side, manifest = read(side_path), read(manifest_path)
    require(side.get("file") == cp.name and side.get("step") == step
            and side.get("smoke") is False and side.get("mode") == "train"
            and side.get("sha256") == digest(cp), "Numeric bootstrap checkpoint identity/hash differs")
    info = side.get("protocol_info")
    require(isinstance(info, dict) and info == manifest.get("protocol_info"),
            "Numeric bootstrap checkpoint/manifest protocol differs")
    source_path = Path(info["source_protocol"]).expanduser().resolve(strict=True)
    require(source_path == Path(manifest["arguments"]["protocol"]).expanduser().resolve(strict=True)
            and digest(source_path) == info["source_protocol_sha256"], "Numeric bootstrap source hash differs")
    source = read(source_path)
    require(source.get("format") == "SOFT_MEMORY_Q_SOURCE_V1", "Wrong Q source protocol")
    e0_path = Path(source["e0_metadata_protocol"]).expanduser().resolve(strict=True)
    require(digest(e0_path) == source["e0_metadata_protocol_sha256"], "Numeric bootstrap E0 metadata hash differs")
    e0 = read(e0_path)
    full, compact = e0.get("runtime_settings"), info.get("execution_settings")
    require(isinstance(full, dict) and bool(full) and isinstance(compact, dict), "Missing saved numerical profile")
    for short, long in OVERLAP.items():
        require(short in compact and long in full and compact[short] == full[long],
                "Recorded Q/Memory numerical profiles disagree: " + short)
    for key in BOOL_FIELDS:
        require(type(full.get(key)) is bool, "Missing/invalid recorded numerical flag: " + key)
    for key in ("num_threads", "num_interop_threads"):
        require(type(full.get(key)) is int and full[key] > 0, "Invalid recorded thread count: " + key)
    require(full.get("default_dtype") in ("torch.float32", "torch.float64", "torch.float16", "torch.bfloat16"),
            "Unsupported saved default dtype")
    require(full.get("float32_matmul_precision") in ("highest", "high", "medium"), "Invalid saved matmul precision")
    env = full.get("environment")
    require(isinstance(env, dict) and set(env) == set(ENV_KEYS), "Unexpected recorded environment schema")
    env = dict(env, CUBLAS_WORKSPACE_CONFIG=full["cublas_workspace_config"],
               CUDA_LAUNCH_BLOCKING=compact["cuda_launch_blocking"])
    require(all(v is None or isinstance(v, str) for v in env.values()), "Invalid recorded environment value")
    return dict(full=full, compact=compact, environment=env,
                checkpoint=str(cp), checkpoint_sha256=side["sha256"],
                bindings={str(p): digest(p) for p in (side_path, manifest_path, source_path, e0_path)})


def restore_environment(profile):
    require("torch" not in sys.modules and "numpy" not in sys.modules,
            "Numeric environment must be restored before importing Torch or NumPy; use a fresh process")
    env = profile["environment"]
    require(set(env) == set(ENV_KEYS) | {"CUBLAS_WORKSPACE_CONFIG", "CUDA_LAUNCH_BLOCKING"},
            "Environment restoration accepts only recorded numerical keys")
    before = {key: os.environ.get(key) for key in env}
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return before


def restore_torch(profile, torch):
    """Host setters only: never call full runtime_settings/current_device here."""
    require(not torch.cuda.is_initialized(), "Saved numeric settings require a fresh CUDA process")
    full = profile["full"]
    require(str(torch.__version__) == full["torch_version"] and torch.version.cuda == full["cuda_version"],
            "PyTorch/CUDA versions differ from recorded Q runtime")
    torch.set_num_threads(full["num_threads"])
    if torch.get_num_interop_threads() != full["num_interop_threads"]:
        torch.set_num_interop_threads(full["num_interop_threads"])
    torch.set_default_dtype(getattr(torch, full["default_dtype"].split(".")[1]))
    torch.use_deterministic_algorithms(full["deterministic_algorithms"], warn_only=full["deterministic_warn_only"])
    torch.backends.cudnn.deterministic = full["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = full["cudnn_benchmark"]
    torch.backends.cudnn.enabled = full["cudnn_enabled"]
    torch.set_float32_matmul_precision(full["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = full["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = full["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = full["fp16_reduced_precision_reduction"]
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = full["bf16_reduced_precision_reduction"]
    require(not torch.cuda.is_initialized(), "Numeric host restoration unexpectedly initialized CUDA")


def check_binding(profile, formal):
    require(profile["checkpoint"] == str(formal.checkpoint)
            and profile["checkpoint_sha256"] == formal.checkpoint_manifest["sha256"]
            and profile["compact"] == formal.saved["protocol_info"]["execution_settings"],
            "Restored runtime is not bound to this serialized checkpoint")
    for path, sha in profile["bindings"].items():
        require(digest(path) == sha, "Numerical profile source changed after bootstrap: " + path)


def bootstrap_cli(argv, default_root):
    """Help/report do not inspect any checkpoint or change numeric settings."""
    if not argv or argv[0] not in ("plan", "run") or "--help" in argv or "-h" in argv:
        return None
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command")
    parser.add_argument("--train-root", default=default_root)
    parser.add_argument("--step", type=int, default=3000)
    args, _ = parser.parse_known_args(argv)
    profile = read_bound_profile(args.train_root, args.step)
    before = restore_environment(profile)
    print("NUMERIC_PROFILE_RESTORE " + json.dumps(dict(
        num_threads=profile["full"]["num_threads"], num_interop_threads=profile["full"]["num_interop_threads"],
        previous_environment=before, restored_environment=profile["environment"],
        original_full_gate_preserved=True, cuda_not_initialized=True)), flush=True)
    return profile
