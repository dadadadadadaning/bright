"""Observe failed probability contracts without repairing or relaxing them."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch


class DiagnosticProbabilityError(ValueError):
    def __init__(self, image, arm, cause, detail):
        super().__init__(f"{cause}; image={image}; arm={arm}; see probability_failure.json")
        self.detail = detail


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _sum_stats(value, tolerance=2e-5):
    x = value.detach().to(device="cpu", dtype=torch.float64)
    finite = torch.isfinite(x)
    delta = (x - 1).abs()
    return dict(min=_number(x.min()), max=_number(x.max()),
                max_abs_error=_number(delta.max()),
                nonfinite_count=int((~finite).sum()),
                outside_tolerance=int(((delta > tolerance) | ~finite).sum()))


def probability_stats(value):
    """Native-device sum and independent CPU float64 sum are diagnostics only."""
    cpu = value.detach().cpu().clone()
    result = dict(shape=list(value.shape), stride=list(value.stride()),
                  dtype=str(value.dtype), device=str(value.device),
                  finite=bool(torch.isfinite(cpu).all()), negative_count=int((cpu < 0).sum()),
                  min=_number(cpu.min()) if cpu.numel() else None,
                  max=_number(cpu.max()) if cpu.numel() else None)
    if cpu.ndim != 4 or cpu.shape[1] != 4 or not cpu.numel():
        return result
    native_sum = value.detach().sum(1)
    native_bad = (~torch.isfinite(native_sum) | ((native_sum - 1).abs() > 2e-5)).cpu()
    double_sum = cpu.double().sum(1)
    result["native_dtype_sum"] = _sum_stats(native_sum)
    result["cpu_float64_sum"] = _sum_stats(double_sum)
    bad = native_bad | ~torch.isfinite(double_sum) | ((double_sum - 1).abs() > 2e-5)
    result["first_bad_pixels"] = []
    for b, y, x in bad.nonzero()[:8].tolist():
        result["first_bad_pixels"].append(dict(batch=b, y=y, x=x,
            probabilities=[_number(v) for v in cpu[b, :, y, x]],
            cpu_float64_sum=_number(double_sum[b, y, x])))
    return result


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(child) for key, child in value.items()}
    return value


def checked_predictions(probabilities, arms, size, *, validator, image,
                        output=None, context=None):
    """Call the historical validator once per arm in its original order.

    Capture only after a ValueError. No normalization, clamp, retry, changed
    argmax, added successful-path CUDA reductions, or continued evaluation.
    """
    predictions = {}
    for arm in arms:
        try:
            predictions[arm] = validator(probabilities[arm], size)
        except ValueError as error:
            detail = dict(format="SRAC_HEAD_PROBABILITY_FAILURE_V1", complete=False,
                          full_validation_evidence=False, image=image, arm=arm,
                          error=f"{type(error).__name__}: {error}",
                          normalization_atol=2e-5, normalization_rtol=0,
                          historical_metric_tolerance=1e-5,
                          expected_output_size=list(size), arms={},
                          rule="Observation only; original validator failed; no correction or retry")
            try:
                for name in arms:
                    detail["arms"][name] = probability_stats(probabilities[name])
                if context and "q" in context:
                    q = context["q"]
                    detail["q"] = dict(shape=list(q.shape), dtype=str(q.dtype),
                                       finite=bool(torch.isfinite(q).all()),
                                       nonnegative=bool((q >= 0).all()),
                                       sum=_sum_stats(q.sum(1)))
                if output is not None:
                    root = Path(output)
                    root.mkdir(parents=True, exist_ok=True)
                    reference = {"p": "pixel_grid", "r": "region_mean", "t": "direct_region"}.get(arm)
                    kept = {arm: probabilities[arm]}
                    if reference and reference in probabilities:
                        kept[reference] = probabilities[reference]
                    path = root / "probability_failure.pt"
                    # Only frozen forward tensors, never GT labels.
                    payload = _cpu_tree(dict(image=image, arm=arm, probabilities=kept,
                                             context=context or {}))
                    with path.open("xb") as stream:
                        torch.save(payload, stream)
                    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import digest
                    detail["snapshot"] = dict(file=path.name, sha256=digest(path), bytes=path.stat().st_size)
            except Exception as capture_error:
                detail["capture_error"] = f"{type(capture_error).__name__}: {capture_error}"
            if output is not None:
                try:
                    with (Path(output) / "probability_failure.json").open("x", encoding="utf-8") as stream:
                        json.dump(detail, stream, ensure_ascii=False, indent=2, allow_nan=False)
                except Exception as write_error:
                    detail["json_write_error"] = f"{type(write_error).__name__}: {write_error}"
            raise DiagnosticProbabilityError(image, arm, str(error), detail) from error
    return predictions
