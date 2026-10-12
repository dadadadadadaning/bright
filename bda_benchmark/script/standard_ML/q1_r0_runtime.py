"""Parameter-free building-region aggregation primitives for Q1-R0."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable, Mapping

import numpy as np
import torch

try:  # Fast production path; the dependency-free fallback keeps tests portable.
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - exercised only on minimal environments
    cv2 = None

from bda_benchmark.script.standard_ML.q0_readout_runtime import (
    event_name,
    metrics_from_confusion,
)


Q1_R0_METHODS = (
    "pixel_vote", "mean", "max", "top10", "top20", "lse", "area"
)


def connected_components(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Dependency-free connected components, used once per validation image.

    The implementation is intentionally explicit so Q1 does not silently
    depend on a particular OpenCV/SciPy build on the cluster.
    """
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError(f"component mask must be 2-D, got {value.shape}")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    labels = np.zeros(value.shape, dtype=np.int32)
    if cv2 is not None:
        count, labels = cv2.connectedComponents(
            value.astype(np.uint8), connectivity=connectivity
        )
        return labels.astype(np.int32, copy=False), int(count - 1)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    height, width = value.shape
    component = 0
    for row, col in zip(*np.nonzero(value)):
        if labels[row, col] != 0:
            continue
        component += 1
        labels[row, col] = component
        queue = deque([(int(row), int(col))])
        while queue:
            current_row, current_col = queue.popleft()
            for delta_row, delta_col in offsets:
                rr, cc = current_row + delta_row, current_col + delta_col
                if (
                    0 <= rr < height and 0 <= cc < width
                    and value[rr, cc] and labels[rr, cc] == 0
                ):
                    labels[rr, cc] = component
                    queue.append((rr, cc))
    return labels, component


def aggregate_probabilities(
    probabilities: torch.Tensor,
    method: str,
    *,
    lse_beta: float = 10.0,
    area_threshold: float = 0.5,
) -> torch.Tensor:
    """Aggregate an [N,3] region without changing its class definition."""
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("region probabilities must have shape [N,3]")
    if probabilities.shape[0] == 0:
        raise ValueError("cannot aggregate an empty region")
    method = str(method).lower()
    if method == "mean":
        return probabilities.mean(0)
    if method == "max":
        return probabilities.max(0).values
    if method in ("top10", "top20"):
        fraction = 0.10 if method == "top10" else 0.20
        count = max(1, int(np.ceil(fraction * probabilities.shape[0])))
        return probabilities.topk(count, dim=0, largest=True, sorted=False).values.mean(0)
    if method == "lse":
        if lse_beta <= 0:
            raise ValueError("LSE beta must be positive")
        return torch.logsumexp(lse_beta * probabilities, dim=0) / lse_beta - (
            np.log(probabilities.shape[0]) / lse_beta
        )
    if method == "area":
        area = (probabilities >= float(area_threshold)).float().mean(0)
        return area if float(area.sum()) > 0 else probabilities.mean(0)
    raise ValueError(f"unsupported Q1-R0 aggregation: {method!r}")


class RegionProtocolAccumulator:
    def __init__(self, methods: Iterable[str] = Q1_R0_METHODS) -> None:
        self.methods = tuple(methods)
        self.region_confusions = {
            method: torch.zeros(3, 3, dtype=torch.long) for method in self.methods
        }
        self.pixel_confusions = {
            method: torch.zeros(3, 3, dtype=torch.long) for method in self.methods
        }
        self.event_confusions = {
            method: defaultdict(lambda: torch.zeros(3, 3, dtype=torch.long))
            for method in self.methods
        }
        self.regions = 0
        self.mixed_regions = 0
        self.small_regions = 0
        self.covered_building_pixels = 0
        self.total_building_pixels = 0

    @staticmethod
    def _increment(confusion: torch.Tensor, target: int, prediction: int, weight: int = 1):
        confusion[target, prediction] += int(weight)

    def add_image(
        self,
        probabilities: torch.Tensor,
        damage: torch.Tensor,
        support: torch.Tensor,
        sample_name: str,
        *,
        minimum_pixels: int,
        minimum_building_fraction: float,
        minimum_class_purity: float,
        lse_beta: float,
        area_threshold: float,
    ) -> None:
        probabilities = probabilities.detach().float().cpu()
        damage = damage.detach().long().cpu()
        support = support.detach().bool().cpu()
        if probabilities.shape[0] != 3 or probabilities.shape[-2:] != damage.shape:
            raise ValueError("Q1-R0 probability/label shape mismatch")
        building = (damage >= 1) & (damage <= 3)
        self.total_building_pixels += int(building.sum())
        labels, component_count = connected_components(support.numpy(), connectivity=8)
        event = event_name(sample_name)
        pixel_prediction = probabilities.argmax(0)
        for component_index in range(1, component_count + 1):
            region = torch.from_numpy(labels == component_index)
            region_size = int(region.sum())
            if region_size < minimum_pixels:
                self.small_regions += 1
                continue
            region_building = region & building
            building_count = int(region_building.sum())
            if building_count / max(region_size, 1) < minimum_building_fraction:
                self.mixed_regions += 1
                continue
            counts = torch.stack([(damage[region_building] == value).sum() for value in (1, 2, 3)])
            target = int(counts.argmax())
            purity = float(counts[target] / max(building_count, 1))
            if purity < minimum_class_purity:
                self.mixed_regions += 1
                continue
            self.regions += 1
            self.covered_building_pixels += building_count
            region_values = probabilities[:, region].T
            predictions: dict[str, int] = {
                # This is a region-level majority vote over the pixel head, not
                # a raw pixel metric.  Keeping the name explicit avoids
                # accidentally comparing two different evaluation units.  It
                # uses the declared proposal region rather than GT building
                # pixels, so the practical P3 protocol does not leak the label
                # mask into prediction.
                "pixel_vote": int(
                    torch.mode(pixel_prediction[region]).values
                )
            }
            for method in self.methods:
                if method == "pixel_vote":
                    continue
                score = aggregate_probabilities(
                    region_values,
                    method,
                    lse_beta=lse_beta,
                    area_threshold=area_threshold,
                )
                predictions[method] = int(score.argmax())
            for method, prediction in predictions.items():
                self._increment(self.region_confusions[method], target, prediction)
                self._increment(
                    self.pixel_confusions[method], target, prediction, building_count
                )
                self._increment(self.event_confusions[method][event], target, prediction)

    def compute(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for method in self.methods:
            region_metrics = metrics_from_confusion(self.region_confusions[method])
            pixel_metrics = metrics_from_confusion(self.pixel_confusions[method])
            events = {
                name: metrics_from_confusion(confusion)
                for name, confusion in sorted(self.event_confusions[method].items())
            }
            event_macro = float(np.mean([value["macro_f1"] for value in events.values()])) if events else 0.0
            result[method] = {
                "region": region_metrics,
                "pixel_weighted": pixel_metrics,
                "event_macro_f1": event_macro,
                "events": events,
            }
        return {
            "methods": result,
            "eligible_regions": self.regions,
            "mixed_or_background_regions": self.mixed_regions,
            "small_regions": self.small_regions,
            "building_pixel_coverage": self.covered_building_pixels / max(self.total_building_pixels, 1),
            "total_building_pixels": self.total_building_pixels,
        }


def decide_r0(payload: Mapping[str, object]) -> dict[str, object]:
    methods = payload["methods"]
    assert isinstance(methods, Mapping)
    mean = methods["mean"]
    candidates = ("max", "top10", "top20", "lse", "area")
    best = max(
        candidates,
        key=lambda name: (
            methods[name]["region"]["damaged_recall"],
            methods[name]["region"]["macro_f1"],
            methods[name]["event_macro_f1"],
        ),
    )
    best_result = methods[best]
    damaged_gain_pp = 100.0 * (
        best_result["region"]["damaged_recall"] - mean["region"]["damaged_recall"]
    )
    macro_delta_pp = 100.0 * (
        best_result["region"]["macro_f1"] - mean["region"]["macro_f1"]
    )
    event_deltas = {}
    common_events = sorted(set(mean["events"]) & set(best_result["events"]))
    for event in common_events:
        event_deltas[event] = 100.0 * (
            best_result["events"][event]["macro_f1"]
            - mean["events"][event]["macro_f1"]
        )
    nonnegative = sum(value >= 0 for value in event_deltas.values())
    passed = bool(
        damaged_gain_pp >= 2.0
        and macro_delta_pp >= -1.0
        and nonnegative >= min(10, len(common_events))
    )
    return {
        "decision": "PROCEED_TO_Q1_STAGE_READOUTS" if passed else "STOP_OR_REDESIGN_LOCAL_AGGREGATION",
        "passed": passed,
        "best_local_method": best,
        "best_minus_mean_pp": {
            "damaged_recall": damaged_gain_pp,
            "macro_f1": macro_delta_pp,
        },
        "nonnegative_events": nonnegative,
        "comparable_events": len(common_events),
        "event_macro_f1_delta_pp": event_deltas,
        "thresholds": {
            "damaged_recall_gain_pp": 2.0,
            "macro_f1_min_delta_pp": -1.0,
            "minimum_nonnegative_events": min(10, len(common_events)),
        },
    }


__all__ = [
    "Q1_R0_METHODS", "RegionProtocolAccumulator", "aggregate_probabilities",
    "connected_components", "decide_r0",
]
