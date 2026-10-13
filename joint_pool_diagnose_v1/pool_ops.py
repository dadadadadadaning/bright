"""Standalone saved-input pool replay; no imports or changes to BRIGHT code.

The contiguous candidate changes integer index layout only. Floating point
operation order, duplicate routes and the 1e-6 denominator clamp are retained.
"""
from __future__ import annotations

from contextlib import contextmanager
import torch

ATOL = 2e-5
RTOL = 2e-5
OUTPUT_KEYS = ("tokens", "mass", "reconstruction")


def native_original_algorithm(content, probabilities, candidate_ids, token_count):
    """Literal arithmetic of the source soft_pool_unpool implementation."""
    b, c, h, w = content.shape
    n = h * w
    if probabilities.shape != (b, 9, n):
        raise ValueError("Association probability shape does not match content")
    values = content.flatten(2).transpose(1, 2)
    numerator = content.new_zeros((b, token_count, c))
    mass = content.new_zeros((b, token_count, 1))
    for j in range(9):
        idx = candidate_ids[j].view(1, n, 1)
        value_idx = idx.expand(b, n, c)
        mass_idx = idx.expand(b, n, 1)
        weight = probabilities[:, j].unsqueeze(-1)
        numerator.scatter_add_(1, value_idx, values * weight)
        mass.scatter_add_(1, mass_idx, weight)
    tokens = numerator / mass.clamp_min(1.0e-6)
    reconstruction = content.new_zeros((b, n, c))
    for j in range(9):
        idx = candidate_ids[j].view(1, n, 1).expand(b, n, c)
        reconstruction.add_(tokens.gather(1, idx) * probabilities[:, j].unsqueeze(-1))
    return {"tokens": tokens, "mass": mass.squeeze(-1),
            "reconstruction": reconstruction.transpose(1, 2).reshape(b, c, h, w)}


def contiguous_index_algorithm(content, probabilities, candidate_ids, token_count):
    """Same arithmetic, with independent contiguous integer indexing tensors.

    Full B*N*C index tensors are intentional: channel chunking would introduce
    another execution change into this first controlled diagnostic.
    """
    b, c, h, w = content.shape
    n = h * w
    if probabilities.shape != (b, 9, n):
        raise ValueError("Association probability shape does not match content")
    candidate_ids = candidate_ids.clone(memory_format=torch.contiguous_format)
    values = content.flatten(2).transpose(1, 2)
    numerator = content.new_zeros((b, token_count, c))
    mass = content.new_zeros((b, token_count, 1))
    for j in range(9):
        idx = candidate_ids[j].view(1, n, 1)
        value_idx = idx.expand(b, n, c).contiguous()
        mass_idx = idx.expand(b, n, 1).contiguous()
        weight = probabilities[:, j].unsqueeze(-1)
        numerator.scatter_add_(1, value_idx, values * weight)
        mass.scatter_add_(1, mass_idx, weight)
    tokens = numerator / mass.clamp_min(1.0e-6)
    reconstruction = content.new_zeros((b, n, c))
    for j in range(9):
        idx = candidate_ids[j].view(1, n, 1).expand(b, n, c).contiguous()
        reconstruction.add_(tokens.gather(1, idx) * probabilities[:, j].unsqueeze(-1))
    return {"tokens": tokens, "mass": mass.squeeze(-1),
            "reconstruction": reconstruction.transpose(1, 2).reshape(b, c, h, w)}


def tensor_layout(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "storage_offset": tensor.storage_offset(), "dtype": str(tensor.dtype),
            "device": str(tensor.device), "contiguous": tensor.is_contiguous()}


def validate_cpu_snapshot(snapshot):
    if not isinstance(snapshot, dict) or snapshot.get("format") != "native_pool_input_v1":
        raise ValueError("Expected a trusted native_pool_input_v1 snapshot")
    content, weights, ids = (snapshot[k] for k in ("content", "weights", "ids"))
    count = snapshot["token_count"]
    if not all(isinstance(t, torch.Tensor) for t in (content, weights, ids)):
        raise TypeError("Snapshot fields must be tensors")
    if any(t.device.type != "cpu" for t in (content, weights, ids)):
        raise ValueError("Validate saved values on CPU before any CUDA call")
    if content.ndim != 4 or min(content.shape) < 1:
        raise ValueError("Content must be nonempty [B,C,H,W]")
    b, _, h, w = content.shape
    if tuple(weights.shape) != (b, 9, h*w) or tuple(ids.shape) != (9, h*w):
        raise ValueError("Expected weights[B,9,H*W] and IDs[9,H*W]")
    if type(count) is not int or count < 1:
        raise ValueError("token_count must be a positive Python integer")
    if ids.dtype != torch.long:
        raise TypeError("IDs must have dtype torch.long")
    if not content.is_floating_point() or weights.dtype != content.dtype:
        raise TypeError("Content and weights must have the same floating dtype")
    if not bool(torch.isfinite(content).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("Nonfinite input content/weights")
    if bool((weights < 0).any()):
        raise ValueError("Negative input weights")
    if bool(((ids < 0) | (ids >= count)).any()):
        raise ValueError("Out-of-range input ID")
    return {"passed": True, "finite": True, "ids_in_range": True,
            "nonnegative_weights": True, "id_min": int(ids.min()), "id_max": int(ids.max()),
            "weight_min": float(weights.min()), "weight_max": float(weights.max()),
            "weight_sum_min": float(weights.sum(1).min()),
            "weight_sum_max": float(weights.sum(1).max()),
            "weights_normalization_required": False}


def validate_layout(tensor, layout):
    """Accept provably nonoverlapping strides and consistent broadcast axes.

    Positive-stride overlaps which cannot be proved absent by span ordering are
    rejected conservatively. Do not silently turn such inputs contiguous.
    """
    if tensor.device.type != "cpu":
        raise ValueError("Layout validation requires CPU saved values")
    shape, stride = layout.get("shape"), layout.get("stride")
    offset = layout.get("storage_offset")
    if not isinstance(shape, (list, tuple)) or not isinstance(stride, (list, tuple)):
        raise ValueError("Missing captured shape/stride")
    if tuple(shape) != tuple(tensor.shape) or len(shape) != len(stride):
        raise ValueError("Captured shape or stride rank mismatch")
    if any(type(v) is not int or v < 0 for v in stride) or type(offset) is not int or offset < 0:
        raise ValueError("Negative or noninteger captured stride/offset")
    if layout.get("dtype") != str(tensor.dtype):
        raise ValueError("Captured dtype mismatch")
    span = 1
    for step, size in sorted((step, size) for size, step in zip(shape, stride) if size > 1 and step > 0):
        if step < span:
            raise ValueError("Ambiguous or internally overlapping positive-stride layout")
        span += (size - 1) * step
    for axis, (size, step) in enumerate(zip(shape, stride)):
        if size > 1 and step == 0:
            if not torch.equal(tensor, tensor.narrow(axis, 0, 1).expand_as(tensor)):
                raise ValueError("Broadcast layout conflicts with stored input values")
    return offset + 1 + sum((size - 1) * step for size, step in zip(shape, stride))


def restore_layout(tensor, layout, device):
    """Restore captured logical layout and storage offset, preserving holes."""
    storage_elements = validate_layout(tensor, layout)
    # empty_strided allocates the backing 1-D storage, including any offset/holes.
    backing = torch.empty_strided((storage_elements,), (1,), dtype=tensor.dtype, device=device)
    view = backing.as_strided(tuple(layout["shape"]), tuple(layout["stride"]), layout["storage_offset"])
    destination, source = view, tensor.to(device)
    for axis, (size, step) in enumerate(zip(layout["shape"], layout["stride"])):
        if size > 1 and step == 0:
            destination = destination.narrow(axis, 0, 1)
            source = source.narrow(axis, 0, 1)
    destination.copy_(source)
    # This access is before any candidate kernel; no such check follows a fault.
    if not torch.equal(view.detach().cpu(), tensor):
        raise ValueError("Reconstructed captured layout changed logical input values")
    return view


def compare_results(actual, expected):
    """Strict fixed atol+rtol; inputs must already be safe CPU tensors."""
    result = {}
    for key in OUTPUT_KEYS:
        aa, bb = actual[key], expected[key]
        if aa.device.type != "cpu" or bb.device.type != "cpu":
            raise ValueError("Copy outputs to CPU before compare_results")
        if aa.shape != bb.shape or aa.dtype != bb.dtype:
            result[key] = {"passed": False, "reason": "shape_or_dtype_mismatch"}
            continue
        finite = bool(torch.isfinite(aa).all()) and bool(torch.isfinite(bb).all())
        diff = (aa.double() - bb.double()).abs()
        bad = diff > (ATOL + RTOL * bb.double().abs())
        result[key] = {"passed": finite and not bool(bad.any()), "finite": finite,
                       "max_abs": float(diff.max()) if finite else None,
                       "mean_abs": float(diff.mean()) if finite else None,
                       "elements_outside_tolerance": int(bad.sum()), "atol": ATOL, "rtol": RTOL}
    result["passed"] = all(result[k]["passed"] for k in OUTPUT_KEYS)
    return result


@contextmanager
def limited_cpu_threads():
    original = torch.get_num_threads()
    try:
        torch.set_num_threads(min(original, 4))
        yield
    finally:
        torch.set_num_threads(original)


def cpu_reference(snapshot):
    validate_cpu_snapshot(snapshot)
    with torch.no_grad(), limited_cpu_threads():
        return native_original_algorithm(snapshot["content"], snapshot["weights"], snapshot["ids"],
                                         snapshot["token_count"])
