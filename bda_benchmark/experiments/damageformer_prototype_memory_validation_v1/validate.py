#!/usr/bin/env python3
"""Causal controls, event-balanced prototypes, pixel impacts and bootstrap."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from bda_benchmark.experiments.damageformer_late_decision_correction_v1.late_core import (
    BG, D, DAMAGED_PX, DESTROYED_PX, I, INTACT_PX, candidate_mask, metrics_from_cm,
    sha256_file, write_json,
)


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def centroid(x: torch.Tensor) -> torch.Tensor:
    if len(x) == 0:
        raise RuntimeError("empty prototype support")
    return normalize(x.mean(0, keepdim=True))[0]


def event_centroids(x, y, events, cls: int) -> torch.Tensor:
    rows = []
    for event in sorted(set(events)):
        index = torch.tensor([(e == event) for e in events], dtype=torch.bool) & (y == cls)
        if bool(index.any()):
            rows.append(centroid(x[index]))
    if not rows:
        raise RuntimeError(f"no event centroid for class={cls}")
    return torch.stack(rows)


def tiny_cosine_kmeans(points: torch.Tensor, k: int, seed: int, iterations: int = 30) -> torch.Tensor:
    points = normalize(points.float())
    k = min(int(k), len(points))
    g = torch.Generator().manual_seed(seed)
    centers = points[torch.randperm(len(points), generator=g)[:k]].clone()
    for _ in range(iterations):
        assignment = (points @ centers.T).argmax(1)
        updated = []
        for j in range(k):
            rows = points[assignment == j]
            updated.append(centroid(rows) if len(rows) else centers[j])
        new = torch.stack(updated)
        if float((new - centers).abs().max()) < 1e-6:
            centers = new; break
        centers = new
    return normalize(centers)


def score_single(xq, xs, ys) -> torch.Tensor:
    return xq @ centroid(xs[ys == D]) - xq @ centroid(xs[ys == I])


def score_balanced_single(xq, xs, ys, events) -> torch.Tensor:
    pd = centroid(event_centroids(xs, ys, events, D))
    pi = centroid(event_centroids(xs, ys, events, I))
    return xq @ pd - xq @ pi


def score_balanced_multi(xq, xs, ys, events, k: int, seed: int) -> torch.Tensor:
    pd = tiny_cosine_kmeans(event_centroids(xs, ys, events, D), k, seed + 17)
    pi = tiny_cosine_kmeans(event_centroids(xs, ys, events, I), k, seed + 31)
    return (xq @ pd.T).max(1).values - (xq @ pi.T).max(1).values


def leave_event_scores(train, val, support_mask, mode: str, *, seed: int, k: int = 4,
                       shuffled_labels: bool = False, random_prototypes: bool = False) -> torch.Tensor:
    xtr = normalize(train["features"].float()); xva = normalize(val["features"].float())
    ytr = train["labels"].long(); train_events = [str(e) for e in train["events"]]
    val_events = [str(e) for e in val["events"]]
    output = torch.zeros(len(xva), dtype=torch.float32)
    for event_index, event in enumerate(sorted(set(val_events))):
        query = torch.tensor([e == event for e in val_events], dtype=torch.bool)
        support = support_mask & torch.tensor([e != event for e in train_events], dtype=torch.bool)
        xs, ys = xtr[support], ytr[support].clone()
        es = [train_events[i] for i in torch.nonzero(support, as_tuple=False).flatten().tolist()]
        g = torch.Generator().manual_seed(seed + 1009 * event_index)
        if shuffled_labels:
            ys = ys[torch.randperm(len(ys), generator=g)]
        if random_prototypes:
            pi = normalize(torch.randn(1, xtr.shape[1], generator=g))[0]
            pd = normalize(torch.randn(1, xtr.shape[1], generator=g))[0]
            output[query] = xva[query] @ pd - xva[query] @ pi
        elif mode == "single_global":
            output[query] = score_single(xva[query], xs, ys)
        elif mode == "single_event_balanced":
            output[query] = score_balanced_single(xva[query], xs, ys, es)
        elif mode == "multi_event_balanced":
            output[query] = score_balanced_multi(xva[query], xs, ys, es, k, seed + event_index)
        else:
            raise ValueError(mode)
    return output


def selected_indices(score: torch.Tensor, candidate: torch.Tensor, coverage: float) -> torch.Tensor:
    ids = torch.nonzero(candidate, as_tuple=False).flatten()
    k = min(len(ids), max(1, int(np.ceil(coverage * len(ids))))) if len(ids) else 0
    return ids[torch.argsort(score[ids], descending=True)[:k]] if k else ids


def apply_one(cm: torch.Tensor, building_cm: torch.Tensor) -> torch.Tensor:
    out = cm.clone()
    moved = building_cm[:, INTACT_PX].long()
    out[:, INTACT_PX] -= moved
    out[:, DAMAGED_PX] += moved
    return out


def image_adjusted_cms(val, image_data, selected: torch.Tensor) -> torch.Tensor:
    names = list(image_data["names"])
    lookup = {name: i for i, name in enumerate(names)}
    output = image_data["image_pixel_cm"].long().clone()
    for row in selected.tolist():
        image_index = lookup[str(val["names"][row])]
        output[image_index] = apply_one(output[image_index], val["building_pixel_cm"][row])
    return output


def size_impact(train, val, selected, purity_threshold: float) -> Dict[str, object]:
    train_area = train["building_pixel_cm"].sum((1, 2)).float()
    valid = (train["label_purity"].float() >= purity_threshold) & (train["labels"].long() >= 0)
    q1, q2 = torch.quantile(train_area[valid], torch.tensor([1/3, 2/3])).tolist()
    output = {}
    for label, lo, hi in (("small", -1, q1), ("medium", q1, q2), ("large", q2, float("inf"))):
        ids = [i for i in selected.tolist() if lo < float(val["building_pixel_cm"][i].sum()) <= hi]
        cm = val["building_pixel_cm"][ids].sum(0) if ids else torch.zeros(4, 4, dtype=torch.long)
        y = val["labels"][ids].long() if ids else torch.empty(0, dtype=torch.long)
        clean = val["label_purity"][ids].float() >= purity_threshold if ids else torch.empty(0, dtype=torch.bool)
        output[label] = {
            "selected_buildings": len(ids),
            "clean_damaged_buildings": int(((y == D) & clean).sum()),
            "clean_intact_buildings": int(((y == I) & clean).sum()),
            "rescue_pixels_D_to_I_reversed": int(cm[DAMAGED_PX, INTACT_PX]),
            "harm_pixels_I_correct_to_D": int(cm[INTACT_PX, INTACT_PX]),
            "destroyed_pixels_X_pred_I_to_D": int(cm[DESTROYED_PX, INTACT_PX]),
        }
    return {"train_area_tertiles": [q1, q2], "bins": output}


def bootstrap(base_images, modified_images, events: Sequence[str], iterations: int, seed: int) -> Dict[str, float]:
    if iterations <= 0:
        return {"iterations": 0, "skipped": True}
    groups = [[i for i, e in enumerate(events) if e == event] for event in sorted(set(events))]
    rng = np.random.default_rng(seed); micro, macro = [], []
    for _ in range(iterations):
        btotal = torch.zeros(4, 4, dtype=torch.long); mtotal = torch.zeros_like(btotal)
        event_delta = []
        for group in groups:
            sampled = rng.choice(group, size=len(group), replace=True).tolist()
            b = base_images[sampled].sum(0); m = modified_images[sampled].sum(0)
            btotal += b; mtotal += m
            event_delta.append(metrics_from_cm(m)["valid_miou"] - metrics_from_cm(b)["valid_miou"])
        micro.append(100 * (metrics_from_cm(mtotal)["valid_miou"] - metrics_from_cm(btotal)["valid_miou"]))
        macro.append(100 * float(np.mean(event_delta)))
    def summary(values):
        a = np.asarray(values)
        return {"mean_pp": float(a.mean()), "ci95_low_pp": float(np.percentile(a, 2.5)),
                "ci95_high_pp": float(np.percentile(a, 97.5)), "probability_gt_zero": float((a > 0).mean())}
    return {"iterations": iterations, "seed": seed, "event_stratified_image_micro": summary(micro),
            "equal_event_macro": summary(macro)}


def evaluate_strategy(name, score, train, val, image_data, candidate, coverage, purity, bootstrap_iterations, seed):
    selected = selected_indices(score, candidate, coverage)
    modified_images = image_adjusted_cms(val, image_data, selected)
    base_images = image_data["image_pixel_cm"].long()
    base_metrics = metrics_from_cm(base_images.sum(0)); metrics = metrics_from_cm(modified_images.sum(0))
    per_event = {}
    for event in sorted(set(image_data["events"])):
        ids = [i for i, e in enumerate(image_data["events"]) if e == event]
        b, m = base_images[ids].sum(0), modified_images[ids].sum(0)
        bm, mm = metrics_from_cm(b), metrics_from_cm(m)
        per_event[event] = {"base_valid_miou": bm["valid_miou"], "valid_miou": mm["valid_miou"],
                            "delta_valid_miou_pp": 100*(mm["valid_miou"]-bm["valid_miou"]),
                            "base_damaged_f1": bm["damaged_f1"], "damaged_f1": mm["damaged_f1"]}
    selected_cm = val["building_pixel_cm"][selected].sum(0)
    labels = val["labels"][selected].long(); clean = val["label_purity"][selected].float() >= purity
    return {
        "name": name, "coverage": coverage, "candidate_count": int(candidate.sum()), "selected": len(selected),
        "base_metrics": base_metrics, "metrics": metrics,
        "delta_valid_miou_pp": 100*(metrics["valid_miou"]-base_metrics["valid_miou"]),
        "delta_miou4_pp": 100*(metrics["miou4"]-base_metrics["miou4"]),
        "delta_damaged_f1_pp": 100*(metrics["damaged_f1"]-base_metrics["damaged_f1"]),
        "clean_rescue_buildings": int(((labels == D) & clean).sum()),
        "clean_harm_buildings": int(((labels == I) & clean).sum()),
        "rescue_pixels_D_to_I_reversed": int(selected_cm[DAMAGED_PX, INTACT_PX]),
        "harm_pixels_I_correct_to_D": int(selected_cm[INTACT_PX, INTACT_PX]),
        "destroyed_pixels_X_pred_I_to_D": int(selected_cm[DESTROYED_PX, INTACT_PX]),
        "event_nonnegative_count": sum(v["delta_valid_miou_pp"] >= 0 for v in per_event.values()),
        "event_positive_count": sum(v["delta_valid_miou_pp"] > 0 for v in per_event.values()),
        "event_count": len(per_event), "per_event": per_event,
        "building_size_impact": size_impact(train, val, selected, purity),
        "bootstrap": bootstrap(base_images, modified_images, image_data["events"], bootstrap_iterations, seed),
        "selected_indices": selected.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_cache", required=True); p.add_argument("--val_cache", required=True)
    p.add_argument("--image_cms", required=True); p.add_argument("--output", required=True)
    p.add_argument("--coverage", type=float, default=0.02)
    p.add_argument("--min_train_purity", type=float, default=0.70)
    p.add_argument("--multi_k", type=int, default=4); p.add_argument("--control_repeats", type=int, default=10)
    p.add_argument("--bootstrap_iterations", type=int, default=10000); p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()
    if "test" in Path(args.output).name.lower(): raise RuntimeError("TEST ACCESS IS FORBIDDEN")
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False, mmap=True)
    val = torch.load(args.val_cache, map_location="cpu", weights_only=False, mmap=True)
    image_data = torch.load(args.image_cms, map_location="cpu", weights_only=False)
    if any(bool(x.get("test_accessed", False)) for x in (train, val, image_data)):
        raise RuntimeError("test-derived input is forbidden")
    if image_data["val_cache_sha256"] != sha256_file(args.val_cache):
        raise RuntimeError("image-CM/val-cache identity mismatch")
    y = train["labels"].long(); purity = train["label_purity"].float()
    support = (purity >= args.min_train_purity) & ((y == I) | (y == D))
    candidate = candidate_mask(val, "base_intact")
    scores = {
        "P2_single_global_primary": leave_event_scores(train, val, support, "single_global", seed=args.seed),
        "P2b_single_event_balanced": leave_event_scores(train, val, support, "single_event_balanced", seed=args.seed),
        "P7_multi_event_balanced": leave_event_scores(train, val, support, "multi_event_balanced", seed=args.seed, k=args.multi_k),
    }
    scores["P4_swapped_labels"] = -scores["P2_single_global_primary"]
    primary = evaluate_strategy("P2_single_global_primary", scores["P2_single_global_primary"], train, val,
                                image_data, candidate, args.coverage, args.min_train_purity,
                                args.bootstrap_iterations, args.seed)
    if abs(primary["base_metrics"]["valid_miou"] - 0.6168659) > 5e-6:
        raise RuntimeError(
            "frozen Base identity failed: expected approximately 0.616866, "
            f"observed={primary['base_metrics']['valid_miou']:.8f}"
        )
    results = {"P2_single_global_primary": primary}
    for name in ("P2b_single_event_balanced", "P7_multi_event_balanced", "P4_swapped_labels"):
        results[name] = evaluate_strategy(name, scores[name], train, val, image_data, candidate, args.coverage,
                                          args.min_train_purity, args.bootstrap_iterations, args.seed)
    controls = {"shuffled_labels": [], "random_prototypes": []}
    for repeat in range(args.control_repeats):
        s = leave_event_scores(train, val, support, "single_global", seed=args.seed + repeat*97,
                               shuffled_labels=True)
        controls["shuffled_labels"].append(evaluate_strategy(
            f"P3_shuffled_labels_r{repeat}", s, train, val, image_data, candidate, args.coverage,
            args.min_train_purity, 0, args.seed + repeat)["delta_valid_miou_pp"])
        s = leave_event_scores(train, val, support, "single_global", seed=args.seed + repeat*193,
                               random_prototypes=True)
        controls["random_prototypes"].append(evaluate_strategy(
            f"P5_random_prototypes_r{repeat}", s, train, val, image_data, candidate, args.coverage,
            args.min_train_purity, 0, args.seed + repeat)["delta_valid_miou_pp"])
    control_summary = {k: {"values_pp": v, "mean_pp": float(np.mean(v)), "std_pp": float(np.std(v))}
                       for k, v in controls.items()}
    output = {
        "protocol": "DAMAGEFORMER_PROTOTYPE_MEMORY_VALIDATION_V1",
        "frozen_primary_candidate": "P2_single_global_primary",
        "frozen_coverage": args.coverage,
        "base_valid_miou_expected": 0.616866,
        "base_identity_observed": primary["base_metrics"],
        "results": results, "random_controls": control_summary,
        "comparisons_pp": {
            "primary_minus_shuffled_mean": primary["delta_valid_miou_pp"] - control_summary["shuffled_labels"]["mean_pp"],
            "primary_minus_random_mean": primary["delta_valid_miou_pp"] - control_summary["random_prototypes"]["mean_pp"],
            "multi_minus_primary": results["P7_multi_event_balanced"]["delta_valid_miou_pp"] - primary["delta_valid_miou_pp"],
            "balanced_single_minus_primary": results["P2b_single_event_balanced"]["delta_valid_miou_pp"] - primary["delta_valid_miou_pp"],
        },
        "specialist_fixed_lambda_status": "stopped; historical negative/non-complementary control only",
        "threshold_policy": "2% selected on Val and frozen before any Test access",
        "test_accessed": False,
    }
    primary_ci = primary["bootstrap"]["event_stratified_image_micro"]
    output["admission"] = {
        "rules": {
            "primary_gain_ge_0.30pp": 0.30,
            "bootstrap_ci_low_gt_zero": 0.0,
            "primary_minus_random_controls_ge_0.15pp": 0.15,
            "event_nonnegative_ge_10_of_14": 10,
            "multi_promotion_margin_ge_0.10pp": 0.10,
        },
        "primary_gain_ge_0.30pp": primary["delta_valid_miou_pp"] >= 0.30,
        "bootstrap_ci_low_gt_zero": primary_ci["ci95_low_pp"] > 0.0,
        "primary_minus_shuffled_ge_0.15pp": output["comparisons_pp"]["primary_minus_shuffled_mean"] >= 0.15,
        "primary_minus_random_ge_0.15pp": output["comparisons_pp"]["primary_minus_random_mean"] >= 0.15,
        "event_nonnegative_ge_10_of_14": primary["event_nonnegative_count"] >= 10,
        "multi_promote_over_primary": output["comparisons_pp"]["multi_minus_primary"] >= 0.10,
    }
    output["admission"]["primary_pass"] = all(
        output["admission"][key] for key in (
            "primary_gain_ge_0.30pp", "bootstrap_ci_low_gt_zero",
            "primary_minus_shuffled_ge_0.15pp", "primary_minus_random_ge_0.15pp",
            "event_nonnegative_ge_10_of_14",
        )
    )
    write_json(args.output, output)
    torch.save({"scores": {k: v.half() for k, v in scores.items()}, "candidate_mask": candidate,
                "coverage": args.coverage, "test_accessed": False}, str(Path(args.output).with_suffix(".pt")))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
