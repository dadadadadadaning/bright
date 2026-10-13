"""Run one saved pool input/backend in its own process. Never train or resume.

Use native and contiguous as separate subprocesses: a CUDA assert poisons its
process context. An exception handler writes CPU records only and never probes
CUDA or retries after a device failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import traceback

import torch

from pool_ops import (ATOL, RTOL, compare_results, contiguous_index_algorithm,
                      cpu_reference, limited_cpu_threads, native_original_algorithm,
                      restore_layout, tensor_layout, validate_cpu_snapshot, validate_layout)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True, help="New nonexistent result directory")
    p.add_argument("--backend", required=True, choices=("cpu", "native", "contiguous"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--runtime-json", help="Required for CUDA: original source runtime settings")
    return p


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_runtime(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("runtime-json must contain an object")
    for key in ("source_runtime_settings", "runtime_settings"):
        if key in data:
            data = data[key]
            break
    if not isinstance(data, dict):
        raise ValueError("Runtime settings must be an object")
    return data


def current_runtime(cpu_only=False):
    if cpu_only:
        return {"device": "cpu", "cuda_not_probed": True,
                "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda}
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "python_version": platform.python_version(), "platform": platform.platform(),
            "cuda_available": True,
            "device": {"index": index, "name": properties.name,
                       "capability": list(torch.cuda.get_device_capability(index)),
                       "total_memory": properties.total_memory},
            "num_threads": torch.get_num_threads(), "num_interop_threads": torch.get_num_interop_threads(),
            "default_dtype": str(torch.get_default_dtype()), "cudnn_enabled": torch.backends.cudnn.enabled,
            "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "environment": {key: os.environ.get(key) for key in (
                "CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING")}


def apply_runtime(requested, device=None):
    required = ("cuda_matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark",
                "cudnn_deterministic", "deterministic_algorithms", "deterministic_warn_only",
                "float32_matmul_precision", "cublas_workspace_config", "torch_version", "cuda_version")
    missing = [key for key in required if key not in requested]
    if missing:
        raise ValueError("Missing source runtime settings: " + ", ".join(missing))
    for key in required[:6]:
        if type(requested[key]) is not bool:
            raise TypeError("Expected boolean runtime setting: " + key)
    if requested["torch_version"] != str(torch.__version__) or requested["cuda_version"] != torch.version.cuda:
        raise ValueError("Source/current torch or CUDA version mismatch; use the original server environment")
    if requested["cublas_workspace_config"] != os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be set by the parent before worker startup")
    # Apply the exact source settings, not an implicit deterministic workaround.
    torch.set_float32_matmul_precision(requested["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = requested["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = requested["cudnn_allow_tf32"]
    torch.backends.cudnn.benchmark = requested["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = requested["cudnn_deterministic"]
    torch.use_deterministic_algorithms(requested["deterministic_algorithms"],
                                       warn_only=requested["deterministic_warn_only"])
    for key, owner, attr in (
            ("cudnn_enabled", torch.backends.cudnn, "enabled"),
            ("fp16_reduced_precision_reduction", torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"),
            ("bf16_reduced_precision_reduction", torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction")):
        if key in requested:
            if type(requested[key]) is not bool:
                raise TypeError("Expected boolean source setting: " + key)
            setattr(owner, attr, requested[key])
    if device is not None:
        torch.cuda.set_device(torch.device(device))
    actual = current_runtime()
    changed = {key: {"requested": requested[key], "actual": actual[key]}
               for key in required if actual[key] != requested[key]}
    if requested.get("cudnn_version") is not None and requested["cudnn_version"] != actual["cudnn_version"]:
        changed["cudnn_version"] = {"requested": requested["cudnn_version"], "actual": actual["cudnn_version"]}
    for key in ("python_version", "platform", "cuda_available", "device", "num_threads",
                "num_interop_threads", "default_dtype", "cudnn_enabled",
                "fp16_reduced_precision_reduction", "bf16_reduced_precision_reduction", "environment"):
        if key in requested and requested[key] != actual[key]:
            changed[key] = {"requested": requested[key], "actual": actual[key]}
    if changed:
        raise ValueError("Runtime does not match source: " + json.dumps(changed))
    return actual


@torch.no_grad()
def run(args):
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    source, output = Path(args.input), Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=False)
    summary = {"format": "JOINT_POOL_LAYOUT_REPLAY_V1", "complete": False,
               "execution_complete": False, "gates_passed": False, "backend": args.backend,
               "training": False, "test_accessed": False, "full_validation_evidence": False,
               "input": str(source.resolve()), "input_sha256": sha256(source),
               "tolerance": {"atol": ATOL, "rtol": RTOL}, "repeats": [], "failure": None}
    phase = "load_saved_cpu_input"
    save(output / "summary.json", summary)
    try:
        snapshot = torch.load(source, map_location="cpu", weights_only=True)
        check = validate_cpu_snapshot(snapshot)
        layout = snapshot.get("layout")
        recorded_layout = isinstance(layout, dict) and all(key in layout for key in ("content", "weights", "ids"))
        if not recorded_layout:
            layout = {key: tensor_layout(snapshot[key]) for key in ("content", "weights", "ids")}
        storage_sizes = {key: validate_layout(snapshot[key], layout[key]) for key in ("content", "weights", "ids")}
        manifest = {"input": str(source.resolve()), "input_sha256": summary["input_sha256"],
                    "image": snapshot.get("image"), "call": snapshot.get("call"),
                    "caller": snapshot.get("caller"), "backend": args.backend,
                    "device": "cpu" if args.backend == "cpu" else args.device,
                    "repeats": args.repeats, "source_layout_recorded": recorded_layout,
                    "source_layout": layout, "restored_storage_elements": storage_sizes,
                    "input_check": check, "runtime_requested": None, "runtime_actual": None,
                    "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                    "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
                    "training": False, "test_accessed": False, "full_validation_evidence": False,
                    "candidate_intervention": "clone IDs and materialize expanded integer indices" if args.backend == "contiguous" else None}
        summary.update(image=snapshot.get("image"), call=snapshot.get("call"), input_check=check,
                       source_layout_recorded=recorded_layout)
        save(output / "manifest.json", manifest)
        phase = "cpu_reference"
        reference = cpu_reference(snapshot)
        summary["cpu_reference_check"] = compare_results(reference, reference)
        save(output / "summary.json", summary)
        if not summary["cpu_reference_check"]["passed"]:
            raise ValueError("CPU reference produced nonfinite results")
        phase = "runtime_configuration"
        runtime_path = getattr(args, "runtime_json", None)
        if runtime_path:
            manifest["runtime_requested"] = read_runtime(runtime_path)
            manifest["runtime_json_sha256"] = sha256(runtime_path)
        if args.backend != "cpu":
            if not runtime_path:
                raise ValueError("CUDA replay requires --runtime-json from this failed run")
            if not recorded_layout:
                raise ValueError("Captured layout metadata is missing; cannot claim original CUDA-layout replay")
            save(output / "manifest.json", manifest)
            manifest["runtime_actual"] = apply_runtime(manifest["runtime_requested"], device=args.device)
        else:
            manifest["runtime_actual"] = current_runtime(cpu_only=True)
        save(output / "manifest.json", manifest)
        device = torch.device("cpu" if args.backend == "cpu" else args.device)
        if args.backend != "cpu" and device.type != "cuda":
            raise ValueError("native/contiguous worker requires a CUDA device; use cpu for CPU replay")
        phase = "restore_captured_layout"
        tensors = [restore_layout(snapshot[key], layout[key], device) for key in ("content", "weights", "ids")]
        manifest["restored_layout"] = {key: tensor_layout(value) for key, value in zip(("content", "weights", "ids"), tensors)}
        save(output / "manifest.json", manifest)
        algorithm = contiguous_index_algorithm if args.backend == "contiguous" else native_original_algorithm
        first = None
        with limited_cpu_threads():
            for repeat in range(args.repeats):
                phase = "pool_" + args.backend + "_repeat_" + str(repeat + 1)
                result = algorithm(*tensors, snapshot["token_count"])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                phase = "copy_successful_outputs_to_cpu"
                host = {key: value.detach().cpu().clone() for key, value in result.items()}
                phase = "compare_cpu_outputs"
                if first is None:
                    first = host
                row = {"repeat": repeat + 1, "vs_cpu": compare_results(host, reference),
                       "vs_first": compare_results(host, first)}
                summary["repeats"].append(row)
                save(output / "summary.json", summary)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if not row["vs_cpu"]["passed"] or not row["vs_first"]["passed"]:
                    raise ValueError("Fixed 2e-5 CPU-reference/repeatability tolerance failed; no relaxation")
        summary.update(complete=True, execution_complete=True, gates_passed=True)
        save(output / "summary.json", summary)
        print("SAVED INPUT REPLAY COMPLETE. One input only. No training. No formal BRIGHT validation.", flush=True)
        return summary
    except Exception as error:
        # Do not access GPU tensors, call empty_cache, query memory, or synchronize.
        summary["failure"] = {"phase": phase, "type": type(error).__name__, "message": str(error)}
        summary["error"] = type(error).__name__ + ": " + str(error)
        summary["phase"] = phase
        save(output / "summary.json", summary)
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    run(parser().parse_args())
