# -*- coding: utf-8 -*-
"""CMR-M0B: event-capped fitting with balanced leakage and natural utility views."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model.cmr.multiprototype import (
    assign_to_centres,
    cross_event_similarity_gain,
    event_capped_indices,
    mean_pairwise_ari,
    normalised_mutual_information,
    prototype_event_diagnostics,
    spherical_kmeans,
)
from script.standard_ML.cmr_metric_utils import load_cmr_cache


def main() -> None:
    parser = argparse.ArgumentParser("CMR-M0B event-capped multi-prototype audit")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clusters", default="2,3,4")
    parser.add_argument("--seeds", default="3407,3408,3409")
    parser.add_argument("--cap_per_event", type=int, default=200)
    parser.add_argument("--relation_pairs", type=int, default=50000)
    parser.add_argument("--minimum_prototype_samples", type=int, default=100)
    parser.add_argument("--minimum_events_per_prototype", type=int, default=5)
    parser.add_argument("--maximum_event_share", type=float, default=0.30)
    parser.add_argument("--minimum_stability_ari", type=float, default=0.60)
    parser.add_argument("--maximum_event_nmi", type=float, default=0.35)
    parser.add_argument("--minimum_similarity_gain", type=float, default=0.03)
    args = parser.parse_args()

    feature, labels, events, event_map, cache = load_cmr_cache(
        args.train_cache, "mean", expected_role="train", print_schema=True
    )
    damaged = labels == 1
    x, e = feature[damaged], events[damaged]
    fit_indices = event_capped_indices(e, cap_per_event=args.cap_per_event, seed=3407)
    fit_x, fit_e = x[fit_indices], e[fit_indices]
    clusters = [int(value) for value in args.clusters.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    results = {}
    for k in clusters:
        natural_assignments = []
        centres_by_seed = []
        for seed in seeds:
            _, centres = spherical_kmeans(fit_x, k, seed=seed)
            centres_by_seed.append(centres)
            natural_assignments.append(assign_to_centres(x, centres))
        natural_assignment = natural_assignments[0]
        balanced_assignment = assign_to_centres(fit_x, centres_by_seed[0])
        balanced_prototypes = prototype_event_diagnostics(
            balanced_assignment, fit_e, clusters=k
        )
        natural_prototypes = prototype_event_diagnostics(
            natural_assignment, e, clusters=k
        )
        stability = mean_pairwise_ari(natural_assignments)
        nmi = normalised_mutual_information(balanced_assignment, fit_e)
        relation = cross_event_similarity_gain(
            x, natural_assignment, e, seed=3407, pairs=args.relation_pairs
        )
        checks = {
            "balanced_prototypes_have_minimum_samples": all(
                row["samples"] >= args.minimum_prototype_samples for row in balanced_prototypes
            ),
            "balanced_prototypes_cover_multiple_events": all(
                row["events"] >= args.minimum_events_per_prototype for row in balanced_prototypes
            ),
            "balanced_prototypes_not_event_dominated": all(
                row["max_event_share"] <= args.maximum_event_share for row in balanced_prototypes
            ),
            "natural_assignment_multi_seed_ari_at_least_declared": stability >= args.minimum_stability_ari,
            "balanced_cluster_event_nmi_at_most_declared": nmi <= args.maximum_event_nmi,
            "natural_cross_event_similarity_gain_at_least_declared": relation["similarity_gain"] >= args.minimum_similarity_gain,
        }
        results[str(k)] = {
            "clusters": k,
            "mean_pairwise_ari_on_natural_assignment": stability,
            "balanced_cluster_event_nmi": nmi,
            "balanced_prototypes": balanced_prototypes,
            "natural_prototypes_diagnostic": natural_prototypes,
            "natural_cross_event_relation": relation,
            "canonical_centres": centres_by_seed[0].tolist(),
            "checks": checks,
            "passed": all(checks.values()),
        }
    admissible = [row for row in results.values() if row["passed"]]
    # Prefer the smallest structure that passes every predeclared check.  A
    # larger K almost always increases within-cluster similarity and must not
    # win merely because it fragments the class more aggressively.
    selected = min(admissible, key=lambda row: row["clusters"], default=None)
    payload = {
        "protocol": "CMR_M0B_event_capped_fit_balanced_leakage_natural_utility_v1",
        "fit_contract": "Train Damaged only; <=cap/event; no replacement",
        "cap_per_event": args.cap_per_event,
        "fit_regions": int(fit_indices.numel()),
        "natural_damaged_regions": int(x.shape[0]),
        "cluster_candidates": results,
        "selected_clusters": None if selected is None else selected["clusters"],
        "passed": selected is not None,
        "decision": (
            "ADMIT_TO_M0C_LEAVE_ONE_EVENT_OUT"
            if selected is not None else "STOP_MULTIPROTOTYPE_AFTER_EVENT_CAPPED_FIT"
        ),
        "p3_checkpoint_sha256": cache["metadata"].get("p3_checkpoint_sha256"),
        "fuses_p3": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 112)
    print("CMR-M0B event-capped prototype fitting audit")
    print(f"Natural/Fit D : {x.shape[0]}/{fit_indices.numel()}")
    for name, row in results.items():
        failed = [key for key, value in row["checks"].items() if not value]
        print(
            f"K={name} | ARI={row['mean_pairwise_ari_on_natural_assignment']:.4f} | "
            f"balanced-NMI={row['balanced_cluster_event_nmi']:.4f} | "
            f"gain={row['natural_cross_event_relation']['similarity_gain']:+.4f} | passed={row['passed']}"
        )
        if failed:
            print(f"      failed: {', '.join(failed)}")
    print(f"Selected K    : {payload['selected_clusters']}")
    print(f"Decision      : {payload['decision']}")
    print(f"Saved         : {output}")
    print("Test accessed : False")


if __name__ == "__main__":
    main()
