"""Fixed saved-scene operators only. Caller controls the process backend policy.

No frontend, labels, training, old-bank rewrite, retry or threshold override.
Fresh GPU transfers do not reproduce the original allocator/execution history.
"""
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F


def _fixed_hash(value):
    if value.device.type != "cpu":
        raise ValueError("Hash accepts an existing CPU snapshot only")
    checksum = hashlib.sha256()
    checksum.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
    checksum.update(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return checksum.hexdigest()


def _fixed_coordinate(flat, shape):
    result = []
    for length in reversed(shape):
        result.append(flat % length)
        flat //= length
    return list(reversed(result))


def _fixed_close(actual, reference, *, logits=False, probability=False):
    """CPU FP64 comparison at the unchanged elementwise 2e-5 gate."""
    if actual.device.type != "cpu" or reference.device.type != "cpu":
        raise ValueError("Compare only already-copied CPU outputs")
    if actual.shape != reference.shape:
        raise ValueError("Fixed-scene comparison shape mismatch")
    a, b = actual.double(), reference.double()
    finite = torch.isfinite(a) & torch.isfinite(b)
    if not bool(finite.all()):
        raise FloatingPointError("Nonfinite fixed-scene output/reference")
    delta = (a - b).abs()
    point = _fixed_coordinate(int(delta.reshape(-1).argmax()), a.shape)
    report = dict(passed=bool((delta <= 2e-5 + 2e-5 * b.abs()).all()),
                  atol=2e-5, rtol=2e-5, max_abs=float(delta.max()),
                  mean_abs=float(delta.mean()), compared_values=a.numel(),
                  outside_tolerance=int((delta > 2e-5 + 2e-5 * b.abs()).sum()),
                  exact_different_values=int((actual != reference).sum()),
                  max_index=point, actual_at_max=float(a[tuple(point)]),
                  reference_at_max=float(b[tuple(point)]))
    if logits or probability:
        dim = 1 if actual.ndim == 4 else -1
        key = "logit_argmax_differences" if logits else "full_resolution_prediction_differences"
        report[key] = int((actual.argmax(dim) != reference.argmax(dim)).sum())
    return report


def _fixed_meta(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                stride=list(value.stride()), storage_offset=value.storage_offset(),
                contiguous=value.is_contiguous(), pointer_mod_256=value.data_ptr() % 256,
                version=value._version, requires_grad=value.requires_grad)


def _fixed_copy(value):
    cpu = value.detach().cpu().clone()
    if not bool(torch.isfinite(cpu).all()):
        raise FloatingPointError("Nonfinite fixed-scene operator output")
    return cpu


def _fixed_probability(logits, size):
    probability = logits.float().softmax(1)
    probability = F.interpolate(probability, size=size, mode="bilinear", align_corners=False)
    return probability / probability.sum(1, keepdim=True).clamp_min(1e-8)


def _fixed_p_reference(tensors, state):
    point = _fixed_close(tensors["p_original_logits"], tensors["p_private_logits"], logits=True)["max_index"]
    batch, cls, row, col = point
    values = tensors["feature"][batch, :, row, col].double()
    weight = state["weight"][cls, :, 0, 0].double()
    products = values * weight
    bias = float(state["bias"][cls]) if "bias" in state else 0.
    signed, absolute = float(products.sum()), float(products.abs().sum())
    reference = signed + bias
    condition = absolute / abs(reference) if reference != 0 else None
    old = float(tensors["p_original_logits"][tuple(point)])
    private = float(tensors["p_private_logits"][tuple(point)])
    return dict(index=point, cpu_fp64_reference=reference, signed_dot_sum=signed,
                dot_abs_sum=absolute, bias=bias, condition_ratio=condition,
                zero_reference=reference == 0, captured_original=old, captured_private=private,
                original_abs_error=abs(old-reference), private_abs_error=abs(private-reference),
                diagnostic_only=True)


def _fixed_mean_reference(tensors):
    point = _fixed_close(tensors["original_mean_tokens"], tensors["recomputed_mean_tokens"])["max_index"]
    batch, region, channel = point
    slots, pixels = (tensors["ids"] == region).nonzero(as_tuple=True)
    weight = tensors["q"][batch, slots, pixels].double()
    values = tensors["feature"][batch, channel].reshape(-1)[pixels].double()
    products, mass = weight * values, float(weight.sum())
    reference = float(products.sum()) / mass if mass > 0 else 0.
    signed, absolute = float(products.sum()), float(products.abs().sum())
    old, new = (float(tensors[name][tuple(point)]) for name in
                ("original_mean_tokens", "recomputed_mean_tokens"))
    return dict(index=point, route_occurrences=len(slots), mass_fp64=mass,
                signed_numerator=signed, absolute_numerator_sum=absolute,
                cancellation_ratio=absolute/abs(signed) if signed else None,
                cpu_fp64_reference=reference, captured_original=old, captured_recomputed=new,
                original_abs_error=abs(old-reference), recomputed_abs_error=abs(new-reference),
                diagnostic_only=True, denominator="exact positive mass; zero token for empty mass")


@torch.no_grad()
def run_fixed_probe(scene, device, mean_pool, GroupedRouting, repeats=3):
    """Exactly three repeats per predefined case; no policy switch or frontend.

    Parent/copy Conv calls and every R input case share immutable input values.
    Shared-plan/rebuilt-plan pools call the injected unmodified mean operator.
    All host copies and changed allocation history are explicitly diagnostic.
    """
    if type(repeats) is not int or repeats != 3:
        raise ValueError("Exactly three predetermined repetitions are required")
    tensors, parent_state = scene["tensors"], scene["parent_classifier_state"]
    private = scene["private_heads_state"]
    names = ("feature", "q", "ids", "original_mean_tokens", "recomputed_mean_tokens")
    for name in names:
        value = tensors[name]
        if value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
            raise ValueError("Require finite validated CPU scene inputs: " + name)
    if (tensors["feature"].ndim != 4 or tensors["feature"].dtype != torch.float32
            or not tensors["feature"].is_contiguous()
            or tensors["feature"].storage_offset() != 0):
        raise ValueError("This contrast preserves the captured contiguous offset-zero F layout")
    for name in ("original_mean_tokens", "recomputed_mean_tokens"):
        if not tensors[name].is_contiguous() or tensors[name].storage_offset() != 0:
            raise ValueError("Require the captured contiguous offset-zero mean-token layout")
    for arm in ("p", "r"):
        for name, value in parent_state.items():
            if not torch.equal(value, private[arm + "." + name]):
                raise ValueError("Copied head differs from original: " + arm + "." + name)
    for value in list(parent_state.values()) + list(private.values()):
        if value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
            raise ValueError("Require finite CPU classifier state")
    channels = parent_state["weight"].shape[1]
    if (parent_state["weight"].shape != (4, channels, 1, 1)
            or tensors["feature"].shape[1] != channels):
        raise ValueError("Require matching four-class affine 1x1 state")
    device = torch.device(device)
    fixed = {name: tensors[name].to(device).clone() for name in names}
    heads = {}
    for role in ("parent", "p", "r"):
        head = nn.Conv2d(channels, 4, 1, bias="bias" in parent_state)
        state = parent_state if role == "parent" else {
            key: private[role + "." + key] for key in parent_state}
        head.load_state_dict(state, strict=True)
        # Reproduce the source's parameter policy under no_grad. state_dict()
        # snapshots detach parameters, so their metadata cannot recover these
        # flags; private heads were trainable even during initial evaluation.
        heads[role] = head.to(device).eval().requires_grad_(role != "parent")
    tracked = dict(fixed)
    tracked.update({role + "." + key: value for role, head in heads.items()
                    for key, value in head.state_dict().items()})
    before = {name: _fixed_hash(_fixed_copy(value)) for name, value in tracked.items()}
    report = dict(diagnostic_only=True, full_validation_evidence=False, training=False,
                  test_accessed=False, original_thresholds_unchanged=True,
                  allocation_history_restored=False, scene_image=scene.get("image"),
                  repeat_count=3, classifier_channels=channels,
                  parameter_policy="Original P/R source trainability: parent=False, private p/r=True; detached state_dict metadata cannot recover flags",
                  execution_grad_enabled=torch.is_grad_enabled(),
                  input_layout={name: _fixed_meta(value) for name, value in fixed.items()},
                  head_layout={role: {name: _fixed_meta(value) for name, value in head.named_parameters()}
                               for role, head in heads.items()},
                  initial_parameter_equal=True, before_sha256=before,
                  p_cpu_fp64_reference=_fixed_p_reference(tensors, parent_state),
                  mean_cpu_fp64_reference=_fixed_mean_reference(tensors), p=[], r={}, pool={})
    first_p = first_p_probability = None
    p_point = tuple(report["p_cpu_fp64_reference"]["index"])
    p_reference = report["p_cpu_fp64_reference"]["cpu_fp64_reference"]
    for role in ("parent", "p"):
        for repeat in range(repeats):
            logits = heads[role](fixed["feature"])
            probability = _fixed_copy(_fixed_probability(logits, scene["output_size"]))
            output = _fixed_copy(logits)
            if first_p is None:
                first_p, first_p_probability = output, probability
            report["p"].append(dict(role=role, repeat=repeat+1,
                logits_requires_grad=logits.requires_grad,
                at_saved_worst_point=dict(index=list(p_point), value=float(output[p_point]),
                    cpu_fp64_reference=p_reference,
                    abs_error_vs_fp64=abs(float(output[p_point])-p_reference)),
                vs_first_parent=_fixed_close(output, first_p, logits=True),
                vs_captured_original=_fixed_close(output, tensors["p_original_logits"], logits=True),
                vs_captured_private=_fixed_close(output, tensors["p_private_logits"], logits=True),
                probability_vs_first_parent=_fixed_close(probability, first_p_probability, probability=True),
                probability_vs_captured_original=_fixed_close(probability, tensors["pixel_grid"], probability=True),
                probability_vs_captured_private=_fixed_close(probability, tensors["p"], probability=True)))
    first_r = {}
    for case in ("original_mean_tokens", "recomputed_mean_tokens"):
        calls = []
        for role in ("parent", "r"):
            for repeat in range(repeats):
                logits = F.linear(fixed[case], heads[role].weight[:, :, 0, 0], heads[role].bias)
                output = _fixed_copy(logits)
                if case not in first_r:
                    first_r[case] = output
                calls.append(dict(role=role, repeat=repeat+1,
                    logits_requires_grad=logits.requires_grad,
                    vs_first_parent_same_input=_fixed_close(output, first_r[case], logits=True),
                    vs_captured_original=_fixed_close(output, tensors["r_original_logits"], logits=True),
                    vs_captured_private=_fixed_close(output, tensors["r_private_logits"], logits=True)))
        report["r"][case] = calls
    report["r_input_effect"] = dict(
        mean_difference=_fixed_close(tensors["recomputed_mean_tokens"], tensors["original_mean_tokens"]),
        same_parent_weight_logits_difference=_fixed_close(first_r["recomputed_mean_tokens"],
                                                         first_r["original_mean_tokens"], logits=True),
        interpretation="Input counterfactual, not fixed-input repeatability")
    shared = GroupedRouting(fixed["ids"], scene["count"])
    first_pool = first_mass = None
    mean_point = tuple(report["mean_cpu_fp64_reference"]["index"])
    mean_reference = report["mean_cpu_fp64_reference"]["cpu_fp64_reference"]
    for case in ("shared_plan", "rebuilt_plan"):
        calls = []
        for repeat in range(repeats):
            plan = shared if case == "shared_plan" else GroupedRouting(fixed["ids"], scene["count"])
            mean, mass = mean_pool(fixed["feature"], fixed["q"], fixed["ids"], scene["count"], routing=plan)
            output, output_mass = _fixed_copy(mean), _fixed_copy(mass)
            if first_pool is None:
                first_pool, first_mass = output, output_mass
            calls.append(dict(repeat=repeat+1,
                at_saved_worst_point=dict(index=list(mean_point), value=float(output[mean_point]),
                    cpu_fp64_reference=mean_reference,
                    abs_error_vs_fp64=abs(float(output[mean_point])-mean_reference)),
                mean_vs_first_pool=_fixed_close(output, first_pool),
                mean_vs_captured_original=_fixed_close(output, tensors["original_mean_tokens"]),
                mean_vs_captured_recomputed=_fixed_close(output, tensors["recomputed_mean_tokens"]),
                mass_vs_first_pool=_fixed_close(output_mass, first_mass),
                mass_vs_captured_original=_fixed_close(output_mass, tensors["original_mass"]),
                mass_vs_captured_recomputed=_fixed_close(output_mass, tensors["recomputed_mass"])))
        report["pool"][case] = calls
    # Deliberately no finally: after a CUDA fault propagate without hashing or
    # accessing any device tensor again. Caller preserves the primary exception.
    after = {name: _fixed_hash(_fixed_copy(value)) for name, value in tracked.items()}
    report.update(after_sha256=after, fixed_inputs_and_heads_unchanged=before == after,
                  operator_calls=dict(p_conv=6, r_linear=12, mean_pool=6))
    if before != after:
        raise RuntimeError("Fixed input/classifier state changed during probe")
    return report


"""Fixed saved P/R scene probes, with one fresh process per backend policy."""

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
