"""Self-contained E0 protocol and same-pass A1/Q helpers for metric evaluation.

The native classifier output is captured from the original front forward. An
extra classifier invocation may differ under the legacy CUDA numerical mode,
so it must never be used as the formal identity gate or Q probability input.
"""
from __future__ import annotations

import json
from pathlib import Path


def protocol_record(path):
    path = Path(path)
    path = path / "protocol.json" if path.is_dir() else path
    record = json.loads(path.read_text(encoding="utf-8"))
    required = (
        record.get("format") == "SRAC_HARDQUERY_PROTOCOL_V1",
        record.get("complete") is True,
        record.get("full_validation_evidence") is True,
        record.get("reproduction", {}).get("passed") is True,
        record.get("image_count") == 450,
        record.get("scope") == "all",
        record.get("smoke") is False,
        record.get("profile") == "legacy",
        record.get("pool_mode") == "native",
    )
    if not all(required):
        raise ValueError("Requires passed legacy/native full-Val450 E0 protocol")
    if any(record.get(key) is not True for key in (
            "input_files_unchanged", "code_unchanged",
            "frozen_evidence_unchanged", "numerical_unchanged")):
        raise ValueError("E0 source or runtime immutability gate failed")
    names = record.get("selected_images")
    if not isinstance(names, list) or len(names) != 450 or len(set(names)) != 450:
        raise ValueError("E0 Val list/order is incomplete")
    return path, record


def sum_nested_counts(dest, src):
    for key, value in src.items():
        if isinstance(value, dict):
            sum_nested_counts(dest.setdefault(key, {}), value)
        elif isinstance(value, list):
            old = dest.setdefault(key, [0] * len(value))
            for index, item in enumerate(value):
                old[index] += item
        else:
            dest[key] = dest.get(key, 0) + value


class NativeA1LogitsCapture:
    """Capture the single native A1 classifier output from one front forward."""

    def __init__(self, front):
        self._calls = 0
        self._last = None
        self._handle = front.base.clf_2.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        self._calls += 1
        self._last = output.detach()

    def clear(self):
        self._calls = 0
        self._last = None

    def take(self):
        if self._calls != 1 or self._last is None:
            raise RuntimeError(f"Expected exactly one native A1 classifier call, got {self._calls}")
        logits = self._last
        self.clear()
        return logits

    def close(self):
        self._handle.remove()


def native_grid_probq(result, feature, low_logits, output_size):
    """Pool the native A1 grid probabilities through the frozen Q routes."""
    import torch
    from torch.nn import functional as F
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.feature_core import (
        mean_pool, resize_probability, unpool, validate_routes)
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.grouped_pool import GroupedRouting
    from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.shared import close_report

    required = ("q", "ids", "count", "shape", "old_pixel")
    if any(key not in result for key in required):
        raise ValueError("Incomplete front result for native A1/Q probe")
    shape = tuple(result["shape"])
    if (tuple(feature.shape[-2:]) != shape or low_logits.ndim != 4
            or low_logits.shape[0] != feature.shape[0]
            or low_logits.shape[1] != 4 or tuple(low_logits.shape[-2:]) != shape
            or tuple(result["old_pixel"].shape[-2:]) != tuple(output_size)):
        raise ValueError("Native A1 logits, feature, Q grid or output size disagree")
    if low_logits.device != feature.device or not bool(torch.isfinite(low_logits).all()):
        raise ValueError("Native A1 logits have wrong device or non-finite values")
    q, ids, count = result["q"], result["ids"], result["count"]
    validate_routes(q, ids, count, shape)
    native_probability = F.interpolate(low_logits.float(), size=output_size,
                                       mode="bilinear", align_corners=False).softmax(1)
    old_probability = result["old_pixel"].float().softmax(1)
    replay = close_report(native_probability, old_probability)
    if replay.get("finite") is not True:
        raise ValueError("Native A1 replay contains non-finite probabilities")
    routing = GroupedRouting(ids, count)
    pooled, _ = mean_pool(low_logits.float().softmax(1), q, ids, count,
                          routing=routing)
    grid_probq = resize_probability(unpool(pooled, q, ids, shape), output_size)
    return grid_probq, replay
