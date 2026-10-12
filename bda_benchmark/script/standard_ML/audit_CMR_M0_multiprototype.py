# -*- coding: utf-8 -*-
"""CMR-M0: no-training audit of cross-event Damaged multi-prototypes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model.cmr.multiprototype import (
    cross_event_similarity_gain,
    mean_pairwise_ari,
    normalised_mutual_information,
    prototype_event_diagnostics,
    spherical_kmeans,
)
from script.standard_ML.cmr_metric_utils import load_cmr_cache


def main() -> None:
    parser = argparse.ArgumentParser("CMR-M0 frozen Damaged multi-prototype audit")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clusters", default="2,3,4,6")
    parser.add_argument("--seeds", default="3407,3408,3409")
    parser.add_argument("--iterations", type=int, default=50)
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
    clusters = [int(value) for value in args.clusters.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 2:
        raise ValueError("M0 needs at least two seeds to audit stability")

    results = {}
    for cluster_count in clusters:
        assignments = []
        seed_rows = []
        for seed in seeds:
            assignment, centres = spherical_kmeans(
                x, cluster_count, seed=seed, iterations=args.iterations
            )
            assignments.append(assignment)
            seed_rows.append({
                "seed": seed,
                "prototype_cosine": (centres @ centres.t()).tolist(),
                "prototypes": prototype_event_diagnostics(
                    assignment, e, clusters=cluster_count
                ),
            })
        canonical = assignments[0]
        prototype_rows = seed_rows[0]["prototypes"]
        relation = cross_event_similarity_gain(
            x, canonical, e, seed=seeds[0], pairs=args.relation_pairs
        )
        stability = mean_pairwise_ari(assignments)
        event_nmi = normalised_mutual_information(canonical, e)
        checks = {
            "all_prototypes_have_minimum_samples": all(
                row["samples"] >= args.minimum_prototype_samples for row in prototype_rows
            ),
            "all_prototypes_cover_multiple_events": all(
                row["events"] >= args.minimum_events_per_prototype for row in prototype_rows
            ),
            "no_prototype_is_event_dominated": all(
                row["max_event_share"] <= args.maximum_event_share for row in prototype_rows
            ),
            "multi_seed_ari_at_least_declared": stability >= args.minimum_stability_ari,
            "cluster_event_nmi_at_most_declared": event_nmi <= args.maximum_event_nmi,
            "within_prototype_similarity_gain_at_least_declared": (
                relation["similarity_gain"] >= args.minimum_similarity_gain
            ),
        }
        results[str(cluster_count)] = {
            "clusters": cluster_count,
            "seeds": seed_rows,
            "mean_pairwise_ari": stability,
            "cluster_event_nmi": event_nmi,
            "cross_event_relation": relation,
            "checks": checks,
            "passed": all(checks.values()),
        }

    admissible = [row for row in results.values() if row["passed"]]
    selected = max(
        admissible,
        key=lambda row: (row["mean_pairwise_ari"], row["cross_event_relation"]["similarity_gain"]),
        default=None,
    )
    payload = {
        "protocol": "CMR_M0_frozen_damaged_multiprototype_audit_v1",
        "scope": "train-only frozen Mean key; no prediction fusion",
        "damaged_regions": int(x.shape[0]),
        "events": len(event_map),
        "cluster_candidates": results,
        "selected_clusters": None if selected is None else selected["clusters"],
        "passed": selected is not None,
        "decision": (
            "ADMIT_MULTIPROTOTYPE_TO_V0_SUPPORT_STRATIFICATION_DO_NOT_FUSE"
            if selected is not None
            else "STOP_MULTIPROTOTYPE_EVENT_OR_STABILITY_CONTRACT_FAILED"
        ),
        "fuses_p3": False,
        "test_accessed": False,
        "p3_checkpoint_sha256": cache["metadata"].get("p3_checkpoint_sha256"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 104)
    print("CMR-M0 frozen Damaged multi-prototype audit")
    for name, row in results.items():
        relation = row["cross_event_relation"]
        print(
            f"K={name:<2} | ARI={row['mean_pairwise_ari']:.4f} | "
            f"event-NMI={row['cluster_event_nmi']:.4f} | "
            f"cross-event gain={relation['similarity_gain']:+.4f} | passed={row['passed']}"
        )
        failed = [check for check, value in row["checks"].items() if not value]
        if failed:
            print(f"       failed checks: {', '.join(failed)}")
    print(f"Selected K      : {payload['selected_clusters']}")
    print(f"Decision        : {payload['decision']}")
    print(f"Saved           : {output}")
    print("Test accessed   : False")


if __name__ == "__main__":
    main()
