"""Independent numerical diagnostics for saved FP32 pooling inputs.

No training, repair acceptance, runtime flag changes or automatic fallback lives
here. FP64 paths promote the *saved* FP32 values; they do not regenerate raw
features, squared features, Q or attention. Comparisons retain fixed 2e-5
absolute and relative tolerances and report failures rather than masking them.
"""
from __future__ import annotations

import heapq
import math

import torch


ATOL = 2e-5
RTOL = 2e-5
OUTPUT_KEYS = ("tokens", "mass", "reconstruction")


def _metadata(content, probabilities, candidate_ids, token_count):
    if not all(isinstance(value, torch.Tensor)
               for value in (content, probabilities, candidate_ids)):
        raise TypeError("Pool inputs must be tensors")
    if content.ndim != 4 or min(content.shape) < 1:
        raise ValueError("Content must be nonempty [B,C,H,W]")
    batch, _, height, width = content.shape
    if probabilities.shape != (batch, 9, height * width):
        raise ValueError("Expected probabilities[B,9,H*W]")
    if candidate_ids.shape != (9, height * width) or candidate_ids.dtype != torch.long:
        raise ValueError("Expected long candidate IDs[9,H*W]")
    if type(token_count) is not int or token_count < 1:
        raise ValueError("token_count must be a positive Python integer")
    if content.dtype not in (torch.float32, torch.float64) or probabilities.dtype != content.dtype:
        raise TypeError("Content and weights must share FP32 or FP64 dtype")
    if content.device != probabilities.device or content.device != candidate_ids.device:
        raise ValueError("Pool inputs must share a device")


def validate_snapshot(snapshot):
    """Check trusted saved values on CPU, without any CUDA query or operation."""
    if not isinstance(snapshot, dict) or snapshot.get("format") != "native_pool_input_v1":
        raise ValueError("Expected native_pool_input_v1 saved input")
    content, weights, ids = (snapshot[key] for key in ("content", "weights", "ids"))
    count = snapshot["token_count"]
    _metadata(content, weights, ids, count)
    if any(value.device.type != "cpu" for value in (content, weights, ids)):
        raise ValueError("Validate saved input values on CPU before CUDA diagnostics")
    if content.dtype != torch.float32:
        raise TypeError("This diagnostic promotes captured FP32 inputs only")
    if not bool(torch.isfinite(content).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("Nonfinite saved content or weights")
    if bool((weights < 0).any()):
        raise ValueError("Negative saved pooling weight")
    if bool(((ids < 0) | (ids >= count)).any()):
        raise ValueError("Out-of-range saved region ID")
    return {"passed": True, "content_dtype": str(content.dtype),
            "weights_dtype": str(weights.dtype), "ids_in_range": True,
            "finite": True, "nonnegative_weights": True,
            "id_min": int(ids.min()), "id_max": int(ids.max()),
            "weights_normalization_required": False}


@torch.no_grad()
def pool_literal(content, probabilities, candidate_ids, token_count, *,
                 index_layout="native", precision="fp32"):
    """Source arithmetic with an explicit index-layout and precision contrast.

    Native FP32 retains the input tensors and expanded integer index views.
    Contiguous materializes only integer indices. FP64 also changes floating
    arithmetic precision deliberately and must not be called original FP32
    replay. Duplicate routes and the original clamp are preserved throughout.
    """
    _metadata(content, probabilities, candidate_ids, token_count)
    if index_layout not in ("native", "contiguous") or precision not in ("fp32", "fp64"):
        raise ValueError("Use native/contiguous layout and fp32/fp64 precision")
    if content.dtype != torch.float32:
        raise TypeError("pool_literal requires captured FP32 source values")
    dtype = torch.float32 if precision == "fp32" else torch.float64
    content = content.to(dtype=dtype)
    probabilities = probabilities.to(dtype=dtype)
    if index_layout == "contiguous":
        candidate_ids = candidate_ids.clone(memory_format=torch.contiguous_format)
    b, c, h, w = content.shape
    n = h * w
    values = content.flatten(2).transpose(1, 2)
    numerator = content.new_zeros((b, token_count, c))
    mass = content.new_zeros((b, token_count, 1))
    for j in range(9):
        idx = candidate_ids[j].view(1, n, 1)
        value_idx = idx.expand(b, n, c)
        mass_idx = idx.expand(b, n, 1)
        if index_layout == "contiguous":
            value_idx = value_idx.contiguous()
            mass_idx = mass_idx.contiguous()
        weight = probabilities[:, j].unsqueeze(-1)
        numerator.scatter_add_(1, value_idx, values * weight)
        mass.scatter_add_(1, mass_idx, weight)
    tokens = numerator / mass.clamp_min(1.0e-6)
    reconstruction = content.new_zeros((b, n, c))
    for j in range(9):
        idx = candidate_ids[j].view(1, n, 1).expand(b, n, c)
        if index_layout == "contiguous":
            idx = idx.contiguous()
        reconstruction.add_(tokens.gather(1, idx) * probabilities[:, j].unsqueeze(-1))
    return {"tokens": tokens, "mass": mass.squeeze(-1),
            "reconstruction": reconstruction.transpose(1, 2).reshape(b, c, h, w)}


@torch.no_grad()
def cpu_reference64(snapshot, *, channel_chunk=64, max_work_elements=2**20):
    """FP64 grouped gather/reduce reference, independent of scatter arithmetic.

    Sort routes by token ID once, gather each token's original pixel/weight
    contributions, and reduce in double precision. Only temporary contribution
    arrays are bounded; returned reconstruction necessarily has full size.
    This does not call pool_literal, scatter_add, index_add or a CUDA API.
    """
    validate_snapshot(snapshot)
    if type(channel_chunk) is not int or channel_chunk < 1:
        raise ValueError("channel_chunk must be a positive integer")
    if type(max_work_elements) is not int or max_work_elements < 1:
        raise ValueError("max_work_elements must be a positive integer")
    content, weights, ids = (snapshot[key] for key in ("content", "weights", "ids"))
    count = snapshot["token_count"]
    b, c, h, w = content.shape
    if max_work_elements < b:
        raise ValueError("Working budget cannot hold one contribution per batch")
    n = h * w
    channel_chunk = min(channel_chunk, c, max_work_elements // b)
    values = content.flatten(2)
    flat_weights = weights.reshape(b, 9*n)
    flat_ids = ids.reshape(-1)
    order = torch.argsort(flat_ids, stable=True)
    sorted_ids = flat_ids.index_select(0, order)
    region_ids, lengths = torch.unique_consecutive(sorted_ids, return_counts=True)
    numerator = torch.zeros((b, count, c), dtype=torch.float64)
    mass = torch.zeros((b, count), dtype=torch.float64)
    offset = 0
    for region, length in zip(region_ids.tolist(), lengths.tolist()):
        members = order[offset:offset + length]
        offset += length
        weight_chunk = max(1, max_work_elements // b)
        for begin in range(0, length, weight_chunk):
            selected = members[begin:begin + weight_chunk]
            mass[:, region] += flat_weights.index_select(1, selected).double().sum(1)
        for start in range(0, c, channel_chunk):
            end = min(start + channel_chunk, c)
            route_chunk = max(1, max_work_elements // (b * (end-start)))
            block = torch.zeros((b, end-start), dtype=torch.float64)
            for begin in range(0, length, route_chunk):
                selected = members[begin:begin + route_chunk]
                pixels = selected.remainder(n)
                chosen = values[:, start:end].index_select(2, pixels).double()
                ww = flat_weights.index_select(1, selected).double()
                block += (chosen * ww[:, None]).sum(2)
            numerator[:, region, start:end] = block
    tokens = numerator / mass.clamp_min(1.0e-6)[..., None]
    reconstruction = torch.zeros((b, c, n), dtype=torch.float64)
    for start in range(0, c, channel_chunk):
        end = min(start + channel_chunk, c)
        pixel_chunk = max(1, max_work_elements // (b * (end-start)))
        token_block = tokens[:, :, start:end]
        for begin in range(0, n, pixel_chunk):
            finish = min(begin + pixel_chunk, n)
            block = torch.zeros((b, finish-begin, end-start), dtype=torch.float64)
            for slot in range(9):
                chosen = token_block.index_select(1, ids[slot, begin:finish])
                ww = weights[:, slot, begin:finish].double()
                block += chosen * ww[..., None]
            reconstruction[:, start:end, begin:finish] = block.transpose(1, 2)
    return {"tokens": tokens, "mass": mass,
            "reconstruction": reconstruction.reshape(b, c, h, w)}


def _flat_block(value, begin, end):
    if value.is_contiguous():
        return value.view(-1)[begin:end]
    # Advanced indexing gathers only this logical block. reshape(-1) could make
    # an entire extra copy of the large transpose-layout reconstruction.
    remaining = torch.arange(begin, end, dtype=torch.long)
    coordinates = []
    for axis, size in enumerate(value.shape):
        stride = math.prod(value.shape[axis+1:])
        coordinates.append(remaining.div(stride, rounding_mode="floor"))
        remaining = remaining.remainder(stride)
    return value[tuple(coordinates)]


def _coordinates(index, shape):
    result = []
    for axis in range(len(shape)):
        stride = math.prod(shape[axis+1:])
        result.append(index // stride)
        index %= stride
    return result


@torch.no_grad()
def compare(actual, reference, *, max_examples=8, work_elements=2**20):
    """Fixed asymmetric tolerance, bounded temporary memory and worst failures.

    |actual-reference| <= 2e-5 + 2e-5*|reference|. The reference controls the
    relative term, so swapping the arguments can change a borderline result.
    FP32 and FP64 outputs may be compared without downcasting the reference.
    All tensors must already be CPU-side; no CUDA access happens here.
    """
    if type(max_examples) is not int or max_examples < 0:
        raise ValueError("max_examples must be a nonnegative integer")
    if type(work_elements) is not int or work_elements < 1:
        raise ValueError("work_elements must be a positive integer")
    result = {}
    for key in OUTPUT_KEYS:
        aa, bb = actual[key], reference[key]
        if not isinstance(aa, torch.Tensor) or not isinstance(bb, torch.Tensor):
            raise TypeError("Comparison fields must be tensors")
        if aa.device.type != "cpu" or bb.device.type != "cpu":
            raise ValueError("Copy successful outputs to CPU before comparison")
        if aa.shape != bb.shape or not aa.is_floating_point() or not bb.is_floating_point():
            result[key] = {"passed": False, "reason": "shape_or_floating_type_mismatch",
                           "actual_shape": list(aa.shape), "reference_shape": list(bb.shape)}
            continue
        count = aa.numel()
        if count == 0:
            result[key] = {"passed": False, "reason": "empty_output"}
            continue
        max_abs = total_abs = max_ratio = 0.0
        outside = nonfinite = 0
        worst = []
        for begin in range(0, count, work_elements):
            end = min(begin + work_elements, count)
            a = _flat_block(aa, begin, end).double()
            b = _flat_block(bb, begin, end).double()
            finite = torch.isfinite(a) & torch.isfinite(b)
            difference = (a-b).abs()
            tolerance = ATOL + RTOL*b.abs()
            ratio = difference/tolerance
            bad = ~finite | (difference > tolerance)
            outside += int(bad.sum())
            nonfinite += int((~finite).sum())
            safe_diff = torch.where(finite, difference, 0.0)
            safe_ratio = torch.where(finite, ratio, 0.0)
            max_abs = max(max_abs, float(safe_diff.max()))
            total_abs += float(safe_diff.sum())
            max_ratio = max(max_ratio, float(safe_ratio.max()))
            if max_examples and bool(bad.any()):
                ranking = torch.where(~finite, torch.inf, ratio).masked_fill(~bad, -torch.inf)
                picked = torch.topk(ranking, min(max_examples, int(bad.sum())))
                for score, position in zip(picked.values.tolist(), picked.indices.tolist()):
                    index = begin + position
                    entry = {"index": _coordinates(index, aa.shape),
                             "flat_index": index,
                             "actual": float(a[position]) if bool(torch.isfinite(a[position])) else None,
                             "reference": float(b[position]) if bool(torch.isfinite(b[position])) else None,
                             "absolute_error": float(difference[position]) if bool(finite[position]) else None,
                             "allowed_error": float(tolerance[position]) if bool(torch.isfinite(b[position])) else None,
                             "normalized_error": score if math.isfinite(score) else None,
                             "nonfinite": not bool(finite[position])}
                    item = (score, -index, entry)
                    if len(worst) < max_examples:
                        heapq.heappush(worst, item)
                    elif item[:2] > worst[0][:2]:
                        heapq.heapreplace(worst, item)
        result[key] = {"passed": outside == 0, "finite": nonfinite == 0,
                       "actual_dtype": str(aa.dtype), "reference_dtype": str(bb.dtype),
                       "shape": list(aa.shape), "atol": ATOL, "rtol": RTOL,
                       "max_abs": max_abs if not nonfinite else None,
                       "mean_abs": total_abs/count if not nonfinite else None,
                       "elements_outside_tolerance": outside, "outside_tolerance": outside,
                       "total_elements": count, "fraction_outside": outside/count,
                       "nonfinite_elements": nonfinite,
                       "max_normalized_error": max_ratio if not nonfinite else None,
                       "worst_failing_examples": [item[2] for item in
                           sorted(worst, key=lambda item: (-item[0], -item[1]))]}
    result["passed"] = all(result[key]["passed"] for key in OUTPUT_KEYS)
    return result
