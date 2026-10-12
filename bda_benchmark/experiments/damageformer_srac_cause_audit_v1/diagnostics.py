"""Paired pixel errors; GT is used only for offline evaluation."""
import numpy as np
import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_srac_approx_v1.audit import boundary_mask
from bda_benchmark.experiments.damageformer_srac_predsplit_v1.audit import metrics, size_masks
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import event


PAIRS = (
    ("a1", "region"), ("a1_resize", "a1_q"),
    ("a1_q", "a1_q_sharp"), ("a1_q", "a1_q_split"),
    ("region", "region_read_sharp"), ("region", "region_pool_sharp"),
    ("region", "region_both_sharp"), ("region", "region_split"),
)
GROUP_PAIRS = (PAIRS[0], PAIRS[1], PAIRS[4], PAIRS[6], PAIRS[7])


def paired_summary(joint):
    harm, rescue = np.zeros((4, 4), dtype=np.int64), np.zeros((4, 4), dtype=np.int64)
    correct = sum(int(joint[c, c].sum()) for c in range(4))
    count = int(joint.sum())
    for c in range(4):
        for other in range(4):
            if c != other:
                harm[c, other] = joint[c, c, other]
                rescue[c, other] = joint[c, other, c]
    return dict(
        count=count, reference_correct=correct, reference_wrong=count-correct,
        harmed=int(harm.sum()), rescued=int(rescue.sum()), net=int(rescue.sum()-harm.sum()),
        harm_rate=float(harm.sum()/correct) if correct else None,
        rescue_rate=float(rescue.sum()/(count-correct)) if count > correct else None,
        damaged_to_intact_harm=int(harm[2, 1]),
        damaged_from_intact_rescue=int(rescue[2, 1]),
        harm_gt_to_wrong=harm.tolist(), rescue_gt_from_wrong=rescue.tolist(),
        joint_gt_reference_candidate=joint.tolist(),
    )


class DiagnosticAccumulator:
    def __init__(self, modes, threshold=0.5):
        self.modes, self.threshold = tuple(modes), threshold
        self.cm = {m: np.zeros((4, 4), dtype=np.int64) for m in modes}
        self.joint = {a+"__"+b: np.zeros((4, 4, 4), dtype=np.int64) for a, b in PAIRS}
        self.groups, self.events, self.images = {}, {}, []
        self.mixing, self.routing = {}, []

    def update(self, gt, predictions, building, entropy, purity_buckets, name, mixing, routing):
        if len(gt) != 1:
            raise ValueError("Only one full image at a time is supported")
        y = gt[0].detach().long().cpu()
        truth = y.numpy().reshape(-1)
        valid = (truth >= 0) & (truth < 4)
        selected_truth = truth[valid]
        pred = {k: v[0].detach().long().cpu().numpy().reshape(-1)[valid] for k, v in predictions.items()}
        if set(pred) != set(self.modes):
            raise ValueError("Prediction modes changed during audit")
        image_cm = {}
        ev = event(name)
        event_cm = self.events.setdefault(ev, {k: np.zeros((4, 4), dtype=np.int64) for k in self.modes})
        for mode, values in pred.items():
            cm = np.bincount(selected_truth*4+values, minlength=16).reshape(4, 4)
            self.cm[mode] += cm
            event_cm[mode] += cm
            image_cm[mode] = cm.tolist()
        self.images.append(dict(image=str(name), event=ev, confusion=image_cm))
        pair_codes = {}
        for a, b in PAIRS:
            key = a+"__"+b
            code = selected_truth*16+pred[a]*4+pred[b]
            self.joint[key] += np.bincount(code, minlength=64).reshape(4, 4, 4)
            pair_codes[key] = code

        def native_expand(value):
            return F.interpolate(value.detach().float().cpu(), y.shape, mode="nearest")[0, 0].numpy().reshape(-1)[valid]
        b = native_expand(building)
        e = native_expand(entropy)
        p = native_expand(purity_buckets).astype(np.int64)
        foreground = selected_truth > 0
        roi = foreground.astype(np.int64)*2 + (b >= self.threshold).astype(np.int64)
        edge = boundary_mask(y).numpy().reshape(-1)[valid].astype(np.int64)
        sizes, _ = size_masks(y)
        size_group = np.zeros(y.shape, dtype=np.int64)
        for i, key in enumerate(("size_small", "size_medium", "size_large"), 1):
            size_group[sizes[key].numpy()] = i
        categories = (
            (roi, ("roi_true_negative", "roi_false_positive", "roi_false_negative", "roi_true_positive")),
            (np.digitize(b, [0.1, 0.4, 0.6, 0.9]), ("B_0_01", "B_01_04", "B_04_06", "B_06_09", "B_09_1")),
            (edge, ("interior", "boundary")),
            (size_group.reshape(-1)[valid], ("size_background", "proxy_small", "proxy_medium", "proxy_large")),
            (np.digitize(e, [0.5, 0.8]), ("entropy_lt05", "entropy_05_08", "entropy_ge08")),
            (p, ("GT_purity_lt080", "GT_purity_080_095", "GT_purity_ge095")),
        )
        for category, labels in categories:
            for a, bname in GROUP_PAIRS:
                key = a+"__"+bname
                combined = category*64 + pair_codes[key]
                values = np.bincount(combined, minlength=len(labels)*64).reshape(len(labels), 4, 4, 4)
                for label, matrix in zip(labels, values):
                    dest = self.groups.setdefault(label, {})
                    dest.setdefault(key, np.zeros((4, 4, 4), dtype=np.int64))[:] += matrix
        self.routing.append(routing)
        for route_name, route_mix in mixing.items():
            dest = self.mixing.setdefault(route_name, {})
            for label, values in route_mix["per_class"].items():
                row = dest.setdefault(label, {key: 0.0 for key in values})
                for key, value in values.items():
                    row[key] += value

    def finalize(self):
        mixing = {}
        for route, classes in self.mixing.items():
            mixing[route] = {}
            for label, sums in classes.items():
                mass = sums["label_mass_feature_pixels"]
                mixing[route][label] = dict(
                    **sums,
                    mean_GT_purity=sums["purity_weighted_sum"]/mass if mass else None,
                    mean_GT_binary_impurity=sums["binary_impurity_weighted_sum"]/mass if mass else None,
                    mean_assignment_entropy=sums["entropy_weighted_sum"]/sums["entropy_weight"] if sums["entropy_weight"] else None,
                )
        route_means = {}
        if self.routing:
            for route, values in self.routing[0].items():
                route_means[route] = {key: float(np.mean([row[route][key] for row in self.routing])) for key in values}
        return dict(
            image_count=len(self.images),
            metrics={k: metrics(torch.from_numpy(v)) for k, v in self.cm.items()},
            paired={k: paired_summary(v) for k, v in self.joint.items()},
            groups={g: {p: paired_summary(v) for p, v in pairs.items()} for g, pairs in self.groups.items()},
            per_event={ev: {k: metrics(torch.from_numpy(v)) for k, v in values.items()} for ev, values in self.events.items()},
            per_image=self.images, mixing_by_GT_class=mixing, mean_routing_per_image=route_means,
            definitions=dict(
                confusion="rows GT, columns prediction; transitions include paired predictions on identical pixels",
                conditional="Group error counts/rates are conditional, not full-image foreground mIoU",
                boundary="GT four-class transitions, radius 2 original pixels, ignores excluded",
                object_size="8-connected GT foreground proxies, not instances; small<256 medium<1024 else large original pixels",
                building="Predicted B bilinear-mapped to Q grid, nearest-expanded to output for grouping",
                purity="Area-resampled GT fractions; Q-weighted region purity; bucket assigned by native hard argmax Q",
                entropy="Divide by log(9); class-conditional area weights; no purity/entropy target assumed optimal",
                uncertainty="Per-image and per-event confusion retained; pixels are not independent replicates",
            ),
        )
