"""Reusable, evaluation-only paired evidence. Ground truth never makes predictions."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_srac_approx_v1.audit import boundary_mask
from bda_benchmark.experiments.damageformer_srac_cause_audit_v1.diagnostics import paired_summary
from bda_benchmark.experiments.damageformer_srac_predsplit_v1.audit import metrics, size_masks
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import event


LABELS = ("Background", "Intact", "Damaged", "Destroyed")
B_LABELS = ("B_0_01", "B_01_04", "B_04_06", "B_06_09", "B_09_1")


def _labels(value, name):
    if not torch.is_tensor(value) or value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(name + " must be a batch-one [1,H,W] tensor")
    if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
        raise ValueError(name + " must contain integer labels")
    return value[0].detach().long().cpu()


class Evidence:
    """Accumulate full-class confusion and paired errors with configurable names.

    Heavy group diagnostics are intended for scheduled full-image evaluation,
    never minibatch training. Every count is int64 and ignore pixels are excluded
    before encoding confusion/transition indices.
    """

    def __init__(self, modes, pairs, threshold=0.5):
        if isinstance(modes, str):
            raise ValueError("modes must be a sequence of distinct names")
        self.modes = tuple(modes)
        if not self.modes or any(not isinstance(x, str) or not x or "__" in x for x in self.modes):
            raise ValueError("modes must be nonempty names without '__'")
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("Duplicate prediction modes")
        if not isinstance(threshold, (float, int)) or not 0 < threshold < 1:
            raise ValueError("Predicted building threshold must be in (0,1)")
        self.threshold = float(threshold)
        self.pairs = []
        for pair in pairs:
            if isinstance(pair, str) or len(pair) != 2:
                raise ValueError("Every pair must contain two mode names")
            a, b = pair
            if a not in self.modes or b not in self.modes or a == b:
                raise ValueError("Every pair must name two different existing modes")
            self.pairs.append((a, b))
        if len(set(self.pairs)) != len(self.pairs):
            raise ValueError("Duplicate prediction pairs")
        self.pairs = tuple(self.pairs)
        self.total = self._empty()
        self.groups, self.events, self.images = {}, {}, []
        self.component_counts = dict(size_small=0, size_medium=0, size_large=0)

    def _empty(self):
        return dict(
            cm={m: np.zeros((4, 4), dtype=np.int64) for m in self.modes},
            joint={a + "__" + b: np.zeros((4, 4, 4), dtype=np.int64) for a, b in self.pairs},
        )

    @staticmethod
    def _finish(bucket):
        return dict(
            metrics={m: metrics(torch.from_numpy(cm)) for m, cm in bucket["cm"].items()},
            paired={pair: paired_summary(joint) for pair, joint in bucket["joint"].items()},
        )

    def _category(self, categories, labels, confusion_codes, pair_codes):
        # One bincount per mode/pair, rather than scanning each condition's pixels.
        buckets = [self.groups.setdefault(label, self._empty()) for label in labels]
        for mode, code in confusion_codes.items():
            values = np.bincount(categories * 16 + code, minlength=len(labels) * 16).reshape(-1, 4, 4)
            for bucket, matrix in zip(buckets, values):
                bucket["cm"][mode] += matrix
        for pair, code in pair_codes.items():
            values = np.bincount(categories * 64 + code, minlength=len(labels) * 64).reshape(-1, 4, 4, 4)
            for bucket, matrix in zip(buckets, values):
                bucket["joint"][pair] += matrix

    def update(self, gt, predictions, building, name):
        y = _labels(gt, "Ground truth")
        if not isinstance(predictions, Mapping) or set(predictions) != set(self.modes):
            raise ValueError("Prediction modes changed during evaluation")
        truth = y.numpy().reshape(-1)
        valid = (truth >= 0) & (truth < 4)
        truth = truth[valid]
        pred = {}
        for mode, value in predictions.items():
            labels = _labels(value, "Prediction " + mode)
            if labels.shape != y.shape:
                raise ValueError("Prediction and ground-truth shapes differ")
            values = labels.numpy().reshape(-1)[valid]
            if np.any((values < 0) | (values >= 4)):
                raise ValueError("Predictions on valid pixels must be in [0,3]")
            pred[mode] = values
        if not torch.is_tensor(building) or building.ndim != 4 or building.shape[:2] != (1, 1):
            raise ValueError("Predicted building probability must be [1,1,h,w]")
        b_tensor = building.detach().float().cpu()
        if not bool(torch.isfinite(b_tensor).all()) or bool(((b_tensor < 0) | (b_tensor > 1)).any()):
            raise ValueError("Predicted building probabilities must be finite and in [0,1]")
        b = F.interpolate(b_tensor, y.shape, mode="nearest")[0, 0].numpy().reshape(-1)[valid]
        confusion_codes = {m: truth * 4 + value for m, value in pred.items()}
        pair_codes = {a + "__" + b: truth * 16 + pred[a] * 4 + pred[b] for a, b in self.pairs}
        image = self._empty()
        ev = event(name)
        event_bucket = self.events.setdefault(ev, self._empty())
        for mode, code in confusion_codes.items():
            image["cm"][mode] = np.bincount(code, minlength=16).reshape(4, 4)
        for pair, code in pair_codes.items():
            image["joint"][pair] = np.bincount(code, minlength=64).reshape(4, 4, 4)
        for bucket in (self.total, event_bucket):
            for category in ("cm", "joint"):
                for key, matrix in image[category].items():
                    bucket[category][key] += matrix
        self.images.append(dict(
            image=str(name), event=ev,
            confusion={key: value.tolist() for key, value in image["cm"].items()},
            joint_gt_reference_candidate={key: value.tolist() for key, value in image["joint"].items()},
        ))
        edge = boundary_mask(y).numpy().reshape(-1)[valid].astype(np.int64)
        foreground = truth > 0
        roi = foreground.astype(np.int64) * 2 + (b >= self.threshold).astype(np.int64)
        sizes, component_counts = size_masks(y)
        size_group = np.zeros(y.shape, dtype=np.int64)
        for i, key in enumerate(("size_small", "size_medium", "size_large"), 1):
            size_group[sizes[key].numpy()] = i
            self.component_counts[key] += component_counts[key]
        categories = (
            (roi, ("roi_true_negative", "roi_false_positive", "roi_false_negative", "roi_true_positive")),
            (np.digitize(b, np.asarray([0.1, 0.4, 0.6, 0.9], dtype=b.dtype)), B_LABELS),
            (edge, ("interior", "boundary")),
            (truth * 2 + edge, tuple(label + suffix for label in LABELS for suffix in ("__interior", "__boundary"))),
            (foreground.astype(np.int64) * 2 + edge, ("background_interior", "background_boundary", "foreground_interior", "foreground_boundary")),
            (size_group.reshape(-1)[valid], ("size_background", "proxy_small", "proxy_medium", "proxy_large")),
        )
        for values, labels in categories:
            self._category(values, labels, confusion_codes, pair_codes)

    def finalize(self):
        return dict(
            image_count=len(self.images), modes=list(self.modes), pairs=[list(x) for x in self.pairs],
            **self._finish(self.total),
            groups={name: self._finish(bucket) for name, bucket in self.groups.items()},
            per_event={name: self._finish(bucket) for name, bucket in self.events.items()},
            per_image=self.images, component_counts=dict(self.component_counts),
            definitions=dict(
                labels=list(LABELS), confusion="rows GT, columns prediction; all four classes retained",
                paired="joint[GT][reference prediction][candidate prediction]; harm/rescue use the same valid pixels",
                conditional="Group scores are conditional pixel scores, not full-image foreground mIoU",
                missing_class="Undefined ratios are null; valid_miou uses fixed three foreground classes, absent classes zero",
                boundary="GT four-class transitions with both sides and Chebyshev radius 2 original pixels; ignores excluded",
                building="Predicted B on Q grid, nearest-expanded for groups; bins [0,.1),[.1,.4),[.4,.6),[.6,.9),[.9,1]",
                roi_threshold=self.threshold,
                object_size="8-connected GT foreground proxies, not instances; area small<256, medium<1024, else large",
                uncertainty="Per-image and per-event confusion and paired matrices retained; pixels are not independent replicates",
                use_of_gt="Offline error grouping only; no GT is used to generate predictions",
            ),
        )
