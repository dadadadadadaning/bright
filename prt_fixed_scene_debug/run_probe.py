"""Fixed saved P/R scene probes, with one fresh process per backend policy."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

MODES = ("legacy", "no_tf32", "cudnn_deterministic", "no_cudnn")
DEFAULT_CHECK = "/root/shared-nvme/BRIGHT-checkpoints/SRAC_PRT_SEQUENCE_EXPERIMENTS_V1/run_20261012_175929_5f5edbb9"


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read_record(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def write_record(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def load_scene(root):
    import torch
    root = Path(root).resolve()
    check_path = root / "check_02/check.json"
    check = read_record(check_path)
    if (check.get("format") != "PRT_SEQUENCE_CHECK_V1" or check.get("mode") != "capture"
            or check.get("complete") is not True or check.get("scene_captured") is not True
            or check.get("scene_file") != "scene.pt" or check.get("training") is not False
            or check.get("test_accessed") is not False
            or check.get("updates_per_arm") != dict(p=0, r=0, t=0)):
        raise ValueError("Require the saved, untrained P/R failure scene")
    scene_path = root / "check_02/scene.pt"
    if file_sha(scene_path) != check.get("scene_sha256"):
        raise ValueError("Scene SHA256 differs from original capture")
    scene = torch.load(scene_path, map_location="cpu", weights_only=True)
    if (scene.get("format") != "PRT_SEQUENCE_SCENE_V1"
            or scene.get("diagnostic_only") is not True or scene.get("training") is not False
            or scene.get("test_accessed") is not False or scene.get("full_validation_evidence") is not False
            or scene.get("image") != check.get("scene_image") or scene.get("index") != check.get("scene_index")
            or scene.get("provenance", {}).get("protocol_sha256") != check.get("protocol_sha256")
            or scene.get("provenance", {}).get("original_source_sha256") != check.get("original_source_sha256")):
        raise ValueError("Scene identity or scope differs from capture")
    def digest_tensor(value):
        h = hashlib.sha256(str(value.dtype).encode())
        h.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        h.update(value.detach().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    for values, hashes in (
        (scene["tensors"], scene["tensor_sha256"]),
        (scene["parent_classifier_state"], scene["state_sha256"]["parent_classifier_state"]),
        (scene["private_heads_state"], scene["state_sha256"]["private_heads_state"]),
    ):
        if set(values) != set(hashes) or any(digest_tensor(v) != hashes[k] for k, v in values.items()):
            raise ValueError("Internal scene tensor/state checksum failed")
    runtime = scene["provenance"]["runtime_settings"]
    if runtime != check.get("runtime_settings"):
        raise ValueError("Capture and scene runtime differ")
    return scene, check, {str(check_path): file_sha(check_path), str(scene_path): file_sha(scene_path)}


def source_inputs(repo, scene, check):
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    from bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1 import core
    from bda_benchmark.experiments.damageformer_srac_memory_metric_eval_v2.native import protocol_record
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import mean_pool
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.grouped_pool import GroupedRouting
    if core.source_hash() != check["original_source_sha256"]:
        raise ValueError("Installed executable source changed since capture")
    path = Path(scene["provenance"]["source_failed_run"]) / "fresh_e0/protocol.json"
    _path, record = protocol_record(path)
    if file_sha(path) != check["protocol_sha256"]:
        raise ValueError("Original passed E0 changed")
    if record.get("runtime_settings") != scene["provenance"]["runtime_settings"]:
        raise ValueError("Scene runtime differs from the original passed E0")
    if record["selected_images"][scene["index"]] != scene["image"]:
        raise ValueError("Captured scene differs from original Val order")
    return mean_pool, GroupedRouting, path, core.source_hash


def configure_mode(runtime, mode, device):
    import torch
    required = ("cuda_matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark",
                "cudnn_deterministic", "deterministic_algorithms", "deterministic_warn_only",
                "float32_matmul_precision", "cudnn_enabled")
    if any(k not in runtime for k in required):
        raise ValueError("Saved runtime is incomplete")
    if str(torch.__version__) != runtime.get("torch_version") or torch.version.cuda != runtime.get("cuda_version"):
        raise ValueError("Torch/CUDA versions differ from capture")
    if torch.backends.cudnn.version() != runtime.get("cudnn_version"):
        raise ValueError("cuDNN version differs from capture")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != runtime.get("cublas_workspace_config"):
        raise ValueError("cuBLAS workspace setting differs from capture")
    environment = runtime.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("Captured runtime environment record is missing")
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if name not in environment or os.environ.get(name) != environment[name]:
            raise ValueError("Runtime environment differs from capture: " + name)
    torch.set_float32_matmul_precision(runtime["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = runtime["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = runtime["cudnn_allow_tf32"]
    torch.backends.cudnn.benchmark = runtime["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = runtime["cudnn_deterministic"]
    torch.backends.cudnn.enabled = runtime["cudnn_enabled"]
    torch.use_deterministic_algorithms(runtime["deterministic_algorithms"], warn_only=runtime["deterministic_warn_only"])
    if "num_threads" in runtime:
        torch.set_num_threads(runtime["num_threads"])
    if "num_interop_threads" in runtime:
        torch.set_num_interop_threads(runtime["num_interop_threads"])
    if mode == "no_tf32":
        torch.backends.cudnn.allow_tf32 = False
    elif mode == "cudnn_deterministic":
        torch.backends.cudnn.deterministic = True
    elif mode == "no_cudnn":
        torch.backends.cudnn.enabled = False
    elif mode != "legacy":
        raise ValueError("Unknown mode")
    selected = torch.device(device)
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; there is no automatic CPU fallback")
        torch.cuda.set_device(selected)
        properties = torch.cuda.get_device_properties(selected)
        captured = runtime.get("device", {})
        if (properties.name != captured.get("name") or properties.total_memory != captured.get("total_memory")
                or list(torch.cuda.get_device_capability(selected)) != captured.get("capability")
                or (selected.index or 0) != captured.get("index")):
            raise ValueError("Selected GPU differs from capture")
    return dict(mode=mode, cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_enabled=torch.backends.cudnn.enabled,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                environment={name: os.environ.get(name) for name in
                             ("CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                device=str(selected))


def worker(cli):
    out = Path(cli.output).resolve()
    for immutable in (Path(cli.repo).resolve(), Path(cli.check_root).resolve()):
        if out == immutable or immutable in out.parents:
            raise ValueError("Worker output enters original repository/evidence")
    out.mkdir(parents=True, exist_ok=False)
    record = dict(complete=False, diagnostic_only=True, mode=cli.worker,
                  training=False, test_accessed=False, full_validation_evidence=False,
                  historical_failure_resolved=False, downstream_authorized=False)
    started = time.perf_counter()
    write_record(out / "result.json", record)
    try:
        scene, check, ledger = load_scene(cli.check_root)
        mean_pool, routing, protocol, source_hash = source_inputs(cli.repo, scene, check)
        ledger[str(protocol)] = file_sha(protocol)
        own = file_sha(__file__)
        before = source_hash()
        record.update(runtime=configure_mode(scene["provenance"]["runtime_settings"], cli.worker, cli.device),
                      image=scene["image"], scene_sha256=check["scene_sha256"],
                      probe_sha256=own, source_sha256=before, source_ledger=ledger)
        if "run_fixed_probe" not in globals():
            from probe_core import run_fixed_probe as probe
        else:
            probe = globals()["run_fixed_probe"]
        record["probe"] = probe(scene, cli.device, mean_pool, routing, repeats=3)
        if source_hash() != before or file_sha(__file__) != own or any(file_sha(p) != h for p, h in ledger.items()):
            raise RuntimeError("Source/scene/probe changed during observation")
        record.update(complete=True, elapsed_seconds=time.perf_counter()-started)
        write_record(out / "result.json", record)
        print("MODE_COMPLETE=" + cli.worker, flush=True)
        return record
    except BaseException as error:
        # No GPU operations or fingerprints after a possible CUDA failure.
        record.update(error=type(error).__name__ + ": " + str(error), elapsed_seconds=time.perf_counter()-started)
        write_record(out / "result.json", record)
        raise


def overview(record):
    probe = record["probe"]
    pools = [call for rows in probe["pool"].values() for call in rows]
    linear = [call for rows in probe["r"].values() for call in rows]
    return dict(
        fixed_inputs_and_heads_unchanged=probe["fixed_inputs_and_heads_unchanged"],
        p_fixed_repeat_max_abs=max(row["vs_first_parent"]["max_abs"] for row in probe["p"]),
        p_fixed_all_within_tolerance=all(row["vs_first_parent"]["passed"] for row in probe["p"]),
        p_fixed_all_bitwise_equal=all(row["vs_first_parent"]["exact_different_values"] == 0 for row in probe["p"]),
        p_first_at_saved_point=probe["p"][0]["at_saved_worst_point"],
        r_fixed_repeat_max_abs=max(row["vs_first_parent_same_input"]["max_abs"] for row in linear),
        r_fixed_all_within_tolerance=all(row["vs_first_parent_same_input"]["passed"] for row in linear),
        pool_fixed_repeat_max_abs=max(row["mean_vs_first_pool"]["max_abs"] for row in pools),
        pool_fixed_all_within_tolerance=all(row["mean_vs_first_pool"]["passed"] for row in pools),
        pool_fixed_all_bitwise_equal=all(row["mean_vs_first_pool"]["exact_different_values"] == 0 for row in pools),
        pool_first_at_saved_point=pools[0]["at_saved_worst_point"],
        pool_first_vs_captured_original=pools[0]["mean_vs_captured_original"],
        pool_first_vs_captured_recomputed=pools[0]["mean_vs_captured_recomputed"],
    )


def main(cli):
    if cli.worker:
        return worker(cli)
    scene, check, ledger = load_scene(cli.check_root)
    _mean, _routing, protocol, _hash = source_inputs(cli.repo, scene, check)
    ledger[str(protocol)] = file_sha(protocol)
    plan = dict(image=scene["image"], planned_modes=list(MODES), repetitions=3,
                processes="One new process per mode, serial; first calls retained",
                operators="P Conv; R fixed-input Linear; same-plan and rebuilt-plan Q mean",
                no_frontend_forward=True, training=False, test_accessed=False,
                full_validation_evidence=False, historical_failure_resolved=False,
                downstream_authorized=False)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if cli.plan_only:
        return plan
    root = Path(cli.output).resolve() if cli.output else Path(
        "/root/shared-nvme/BRIGHT-checkpoints/SRAC_PRT_FIXED_SCENE_V1"
    ) / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    repo, check_root = Path(cli.repo).resolve(), Path(cli.check_root).resolve()
    if root == repo or repo in root.parents or root == check_root or check_root in root.parents:
        raise ValueError("New output must be outside repository and original failure")
    root.mkdir(parents=True, exist_ok=False)
    print("OUTPUT_ROOT=" + str(root), flush=True)
    state = dict(**plan, diagnostic_complete=False, modes={}, input_ledger=ledger,
                 probe_sha256=file_sha(__file__))
    write_record(root / "summary.json", state)
    try:
        for mode in MODES:
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--repo", cli.repo, "--check-root", cli.check_root,
                       "--device", cli.device, "--worker", mode, "--output", str(root / mode)]
            log = root / (mode + ".log")
            print("Starting " + mode, flush=True)
            with log.open("x") as f:
                process = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT)
                try:
                    rc = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            result_path = root / mode / "result.json"
            result = read_record(result_path) if result_path.is_file() else {}
            state["modes"][mode] = dict(returncode=rc, result_file=str(result_path), record=result)
            write_record(root / "summary.json", state)
            if rc != 0 or result.get("complete") is not True:
                print(log.read_text(errors="replace"), flush=True)
                raise RuntimeError("Mode failed; stopped without retry: " + mode)
        if any(file_sha(p) != h for p, h in ledger.items()) or file_sha(__file__) != state["probe_sha256"]:
            raise RuntimeError("Input/probe files changed")
        state["diagnostic_complete"] = True
        state["overview"] = {mode: overview(row["record"]) for mode, row in state["modes"].items()}
        write_record(root / "summary.json", state)
        print("OVERVIEW=" + json.dumps(state["overview"], ensure_ascii=False), flush=True)
        print("SUMMARY=" + str(root / "summary.json"), flush=True)
        return state
    except BaseException as error:
        state["error"] = type(error).__name__ + ": " + str(error)
        write_record(root / "summary.json", state)
        raise


def parser():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--repo", default="/root/shared-nvme/BRIGHT-master")
    p.add_argument("--check-root", default=DEFAULT_CHECK)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--worker", choices=MODES, help=argparse.SUPPRESS)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
