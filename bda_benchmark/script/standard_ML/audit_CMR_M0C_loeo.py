# -*- coding: utf-8 -*-
"""CMR-M0C: leave-one-event-out transfer audit for admitted Damaged prototypes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model.cmr.multiprototype import assign_to_centres, event_capped_indices, spherical_kmeans
from script.standard_ML.cmr_metric_utils import load_cmr_cache
from script.standard_ML.rm_region_retrieval_runtime import binary_auroc


def main() -> None:
    parser = argparse.ArgumentParser("CMR-M0C LOEO prototype transfer audit")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--m0b_result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cap_per_event", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--minimum_event_auc_gain", type=float, default=0.02)
    parser.add_argument("--minimum_nondecreasing_events", type=int, default=10)
    parser.add_argument("--minimum_multimode_events", type=int, default=10)
    args = parser.parse_args()

    m0b = json.load(open(args.m0b_result, encoding="utf-8"))
    if m0b.get("protocol") != "CMR_M0B_event_capped_fit_balanced_leakage_natural_utility_v1":
        raise RuntimeError("M0C refuses an unknown M0B result")
    if not bool(m0b.get("passed", False)) or m0b.get("selected_clusters") is None:
        raise RuntimeError("M0C is blocked because M0B did not admit a prototype count")
    k = int(m0b["selected_clusters"])
    feature, labels, events, event_map, cache = load_cmr_cache(
        args.train_cache, "mean", expected_role="train", print_schema=True
    )
    x = F.normalize(feature.float(), dim=1, eps=1e-8)
    rows = []
    for held_event in sorted(int(value) for value in torch.unique(events).tolist()):
        train_d = (labels == 1) & (events != held_event)
        train_i = (labels == 0) & (events != held_event)
        held = (events == held_event) & ((labels == 0) | (labels == 1))
        if int((held & (labels == 1)).sum()) == 0 or int((held & (labels == 0)).sum()) == 0:
            continue
        d_indices_local = event_capped_indices(
            events[train_d], cap_per_event=args.cap_per_event, seed=args.seed
        )
        i_indices_local = event_capped_indices(
            events[train_i], cap_per_event=args.cap_per_event, seed=args.seed
        )
        train_d_indices = torch.nonzero(train_d, as_tuple=False).flatten()[d_indices_local]
        train_i_indices = torch.nonzero(train_i, as_tuple=False).flatten()[i_indices_local]
        _, d_centres = spherical_kmeans(x[train_d_indices], k, seed=args.seed)
        d_one = F.normalize(x[train_d_indices].mean(dim=0), dim=0, eps=1e-8)[None]
        i_centre = F.normalize(x[train_i_indices].mean(dim=0), dim=0, eps=1e-8)
        query = x[held]
        target = labels[held] == 1
        score_k = (query @ d_centres.t()).max(dim=1).values - query @ i_centre
        score_one = (query @ d_one.t()).flatten() - query @ i_centre
        held_d = x[(events == held_event) & (labels == 1)]
        assignment = assign_to_centres(held_d, d_centres)
        histogram = torch.bincount(assignment, minlength=k)
        rows.append({
            "event_code": held_event,
            "event": next(name for name, code in event_map.items() if code == held_event),
            "queries": int(target.numel()),
            "damaged_queries": int(target.sum()),
            "k1_auroc": binary_auroc(score_one, target),
            "multi_prototype_auroc": binary_auroc(score_k, target),
            "auc_gain": binary_auroc(score_k, target) - binary_auroc(score_one, target),
            "active_prototypes": int((histogram > 0).sum()),
            "maximum_assignment_share": float(histogram.max() / histogram.sum().clamp_min(1)),
            "assignment_counts": histogram.tolist(),
        })
    macro_k1 = sum(row["k1_auroc"] for row in rows) / max(len(rows), 1)
    macro_multi = sum(row["multi_prototype_auroc"] for row in rows) / max(len(rows), 1)
    nondecreasing = sum(row["auc_gain"] >= 0 for row in rows)
    multimode = sum(row["active_prototypes"] >= 2 for row in rows)
    checks = {
        "event_macro_auc_gain_at_least_declared": macro_multi - macro_k1 >= args.minimum_event_auc_gain,
        "nondecreasing_events_at_least_declared": nondecreasing >= args.minimum_nondecreasing_events,
        "held_out_events_use_multiple_prototypes": multimode >= args.minimum_multimode_events,
    }
    passed = all(checks.values())
    payload = {
        "protocol": "CMR_M0C_leave_one_event_out_damaged_prototype_transfer_v1",
        "selected_clusters": k,
        "events": rows,
        "event_macro_k1_auroc": macro_k1,
        "event_macro_multi_prototype_auroc": macro_multi,
        "event_macro_auc_gain": macro_multi - macro_k1,
        "nondecreasing_events": nondecreasing,
        "multimode_events": multimode,
        "checks": checks,
        "passed": passed,
        "decision": (
            "ADMIT_TO_TRAIN_ONLY_SOFT_DAMAGED_PROTOTYPE_AUXILIARY"
            if passed else "STOP_MULTIPROTOTYPE_LOEO_TRANSFER_INSUFFICIENT"
        ),
        "fuses_p3": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 112)
    print("CMR-M0C leave-one-event-out prototype transfer audit")
    print(f"K1/Multi event AUC : {macro_k1:.4f}/{macro_multi:.4f} ({macro_multi-macro_k1:+.4f})")
    print(f"Nondecreasing      : {nondecreasing}/{len(rows)}")
    print(f"Multimode events   : {multimode}/{len(rows)}")
    print(f"Checks             : {checks}")
    print(f"Decision           : {payload['decision']}")
    print(f"Saved              : {output}")
    print("Test accessed      : False")


if __name__ == "__main__":
    main()

