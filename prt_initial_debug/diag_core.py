"""Read-only fixed-input P/R numerical diagnosis; never a formal evaluation.

Backend probes run only the fixed classifier, softmax, unpool and resize. They
must not wrap A1/frontend execution. No thresholds, weights or source files are
changed. CPU float64 dot products are references, not replacement predictions.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

ATOL = RTOL = 2e-5


def tensor_meta(tensor):
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                device=str(tensor.device), stride=list(tensor.stride()),
                storage_offset=tensor.storage_offset(),
                is_contiguous=tensor.is_contiguous(),
                data_ptr_mod_256=tensor.data_ptr() % 256,
                requires_grad=tensor.requires_grad, version=tensor._version,
                sha256=tensor_digest(tensor))


def tensor_digest(tensor):
    """Hash logical values, independently of allocator alignment or strides."""
    data = tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
    checksum = hashlib.sha256()
    checksum.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
    checksum.update(data.numpy().tobytes())
    return checksum.hexdigest()


def _index(flat, shape):
    result = []
    for length in reversed(shape):
        result.append(flat % length)
        flat //= length
    return list(reversed(result))


def close_report(actual, reference, *, atol=ATOL, rtol=RTOL):
    """Elementwise gate, with symmetric metadata and the worst absolute point."""
    if actual.shape != reference.shape:
        return dict(passed=False, reason="shape mismatch",
                    actual_shape=list(actual.shape), reference_shape=list(reference.shape))
    a, b = actual.detach(), reference.detach()
    finite = torch.isfinite(a) & torch.isfinite(b)
    difference = (a.double() - b.double()).abs()
    allowed = atol + rtol * b.double().abs()
    outside = (~finite) | (difference > allowed)
    flat = int(torch.where(finite, difference, torch.full_like(difference, float("inf")))
               .reshape(-1).argmax().item()) if a.numel() else None
    point = _index(flat, a.shape) if flat is not None else None
    def value(tensor):
        val = float(tensor[tuple(point)].item()) if point is not None else None
        return val if val is None or math.isfinite(val) else None
    report = dict(passed=not bool(outside.any()), atol=atol, rtol=rtol,
                  compared_values=a.numel(), nonfinite_values=int((~finite).sum()),
                  outside_tolerance=int(outside.sum()), max_index=point,
                  max_abs=float(difference[finite].max().item()) if bool(finite.any()) else None,
                  actual_at_max=value(a), reference_at_max=value(b))
    if a.ndim in (3, 4):
        dim = 1 if a.ndim == 4 else -1
        report["prediction_differences"] = int((a.argmax(dim) != b.argmax(dim)).sum())
    return report


def backend_flags():
    return dict(cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_enabled=torch.backends.cudnn.enabled,
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)


@contextmanager
def temporary_backend_flags(*, cudnn_allow_tf32=None, cudnn_enabled=None):
    """Restore every touched flag, including when classifier/unpool raises."""
    before = backend_flags()
    try:
        if cudnn_allow_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32
        if cudnn_enabled is not None:
            torch.backends.cudnn.enabled = cudnn_enabled
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = before["cudnn_allow_tf32"]
        torch.backends.cudnn.enabled = before["cudnn_enabled"]


def _weight(head, kind):
    return head.weight if kind == "p" or head.weight.ndim == 2 else head.weight[:, :, 0, 0]


def _validate(kind, tensor, head):
    if kind not in ("p", "r"):
        raise ValueError("kind must be p or r")
    if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
        raise ValueError("Require finite fixed FP32 input")
    if isinstance(head, nn.Conv2d):
        if (head.kernel_size != (1, 1) or head.stride != (1, 1)
                or head.padding != (0, 0) or head.dilation != (1, 1) or head.groups != 1):
            raise ValueError("Require original affine 1x1 convolution")
    elif kind != "r" or not isinstance(head, nn.Linear):
        raise ValueError("P requires Conv2d; R requires affine Conv2d or Linear")
    weight = _weight(head, kind)
    channel = tensor.shape[1] if kind == "p" else tensor.shape[-1]
    if tensor.ndim != (4 if kind == "p" else 3) or channel != weight.shape[1]:
        raise ValueError("Input channels/shape do not match classifier")


def _classify(kind, tensor, head, functional=False):
    if kind == "r":
        return F.linear(tensor, _weight(head, kind), head.bias)
    if functional:
        return F.conv2d(tensor, head.weight, head.bias, stride=head.stride,
                        padding=head.padding, dilation=head.dilation, groups=head.groups)
    return head(tensor)


def _stages(kind, tensor, head, output_size, unpool, functional=False):
    logits = _classify(kind, tensor, head, functional)
    probability = logits.float().softmax(1 if kind == "p" else -1)
    grid = probability if kind == "p" else unpool(probability)
    resized = F.interpolate(grid, output_size, mode="bilinear", align_corners=False)
    resized = resized / resized.sum(1, keepdim=True).clamp_min(1e-8)
    stages = dict(logits=logits, softmax=probability, resize=resized)
    if kind == "r":
        stages["unpool"] = grid
    for name, value in stages.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite fixed-head diagnostic stage: " + name)
    return stages


def _dot_reference(kind, tensor, head, point):
    if point is None:
        return None
    if kind == "p":
        batch, channel, row, col = point
        vector = tensor[batch, :, row, col]
    else:
        batch, token, channel = point
        vector = tensor[batch, token]
    weight = _weight(head, kind)[channel]
    if kind == "p":
        weight = weight[:, 0, 0]
    x, w = vector.detach().cpu().double(), weight.detach().cpu().double()
    bias = float(head.bias[channel].detach().cpu().double()) if head.bias is not None else 0.
    return dict(reference_kind="CPU float64 fixed-input affine dot; diagnostic only",
                index=point, dot_without_bias=float(torch.dot(x, w)),
                bias=bias, logit=float(torch.dot(x, w)) + bias,
                input_l2=float(x.norm()), weight_l2=float(w.norm()))


def _compare(kind, tensor, actual, reference, head, ref_head):
    results = {name: close_report(value, reference[name]) for name, value in actual.items()}
    results["logits"]["cpu_float64_actual_head"] = _dot_reference(
        kind, tensor, head, results["logits"]["max_index"])
    results["logits"]["cpu_float64_reference_head"] = _dot_reference(
        kind, tensor, ref_head, results["logits"]["max_index"])
    return results


@torch.no_grad()
def compare_fixed_heads(kind, input_tensor, original_head, cloned_head, output_size,
                        unpool=None, run_backend_probes=True,
                        captured_stages: Mapping[str, torch.Tensor] | None = None):
    """Return JSON-safe per-stage repeat/copy/backend evidence on one fixed input.

    R uses F.linear even when its weight is stored in the original Conv2d.
    `unpool` accepts token probabilities and returns a BCHW probability grid.
    `captured_stages` may hold logits/softmax/unpool/resize from the real legacy
    call; comparison is to original_1 at the unchanged incoming backend flags.
    """
    for head in (original_head, cloned_head):
        _validate(kind, input_tensor, head)
    if kind == "r" and unpool is None:
        raise ValueError("R requires an injected probability unpool function")
    incoming = backend_flags()
    snapshots = [[parameter.detach().clone() for parameter in head.parameters()]
                 for head in (original_head, cloned_head)]
    state_equal = (original_head.state_dict().keys() == cloned_head.state_dict().keys()
                   and all(torch.equal(value, cloned_head.state_dict()[name])
                           for name, value in original_head.state_dict().items()))
    report = dict(diagnostic_only=True, smoke=True, training=False,
                  full_validation_evidence=False, test_accessed=False,
                  formal_evaluation=False, thresholds_changed=False, kind=kind,
                  same_input_for_all_calls=True, initial_parameter_equal=state_equal,
                  input=tensor_meta(input_tensor), incoming_backend=incoming,
                  heads={label: {name: tensor_meta(parameter)
                                 for name, parameter in head.named_parameters()}
                         for label, head in (("original", original_head), ("clone", cloned_head))},
                  modes={})
    modes = [("legacy", {})]
    if run_backend_probes:
        modes += [("cudnn_tf32_off_fixed_head", dict(cudnn_allow_tf32=False)),
                  ("cudnn_disabled_fixed_head", dict(cudnn_enabled=False))]
    legacy, succeeded = None, False
    try:
        for label, flags in modes:
            with temporary_backend_flags(**flags):
                variants = [("original_1", original_head, False), ("original_2", original_head, False),
                            ("clone_1", cloned_head, False), ("clone_2", cloned_head, False),
                            ("functional_original", original_head, True),
                            ("functional_clone", cloned_head, True)]
                outputs = {name: _stages(kind, input_tensor, head, output_size, unpool, functional)
                           for name, head, functional in variants}
                reference = outputs["original_1"]
                mode = dict(backend=backend_flags(), comparisons={})
                for name, head, _ in variants[1:]:
                    mode["comparisons"][name + "_vs_original_1"] = _compare(
                        kind, input_tensor, outputs[name], reference, head, original_head)
                if legacy is None:
                    legacy = reference
                    report["captured_vs_fixed"] = {
                        stage: close_report(value, reference[stage])
                        for stage, value in (captured_stages or {}).items() if stage in reference}
                else:
                    mode["original_1_vs_legacy_original_1"] = _compare(
                        kind, input_tensor, reference, legacy, original_head, original_head)
                report["modes"][label] = mode
        succeeded = True
    finally:
        report["backend_restored"] = backend_flags() == incoming
        # CUDA device errors poison the process: preserve the primary exception
        # and perform no further tensor operation, copy, hash or alternate probe.
        report["parameters_unchanged"] = report["input_unchanged"] = None
        if succeeded:
            report["parameters_unchanged"] = all(
                torch.equal(parameter, snapshot)
                for head, copies in zip((original_head, cloned_head), snapshots)
                for parameter, snapshot in zip(head.parameters(), copies))
            report["input_unchanged"] = tensor_digest(input_tensor) == report["input"]["sha256"]
    return report
