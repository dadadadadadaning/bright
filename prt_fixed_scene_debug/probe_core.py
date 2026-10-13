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
