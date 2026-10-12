#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""O-K0: train-locked cross-event Intact/Affected key audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

from script.standard_ML.cmr_metric_utils import relation_statistics
from script.standard_ML.hm_occurrence_runtime import (
    binary_centre_scores,
    event_class_capped_indices,
    event_probe_report,
    fit_binary_centres,
    loeo_occurrence_scores,
    occurrence_auc_report,
)


def load_cache(path: str, role: str) -> Mapping[str, object]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("protocol") != "HM_OK0_exact_replay_s2o_occurrence_keys_v1":
        raise RuntimeError(f"not an O-K0 key cache: {path}")
    if cache.get("role") != role:
        raise RuntimeError(f"O-K0 cache role mismatch: {cache.get('role')} != {role}")
    if bool(cache.get("metadata", {}).get("test_accessed", True)):
        raise RuntimeError("O-K0 refuses Test-derived keys")
    return cache


def load_p0(path: str, role: str) -> Mapping[str, object]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    metadata = cache.get("metadata", {})
    if metadata.get("protocol") != "CMR_P0_frozen_pooling_audit_v1":
        raise RuntimeError("O-K0 baseline must be a frozen CMR-P0 cache")
    expected = "train" if role == "train" else "query"
    if metadata.get("role") != expected or bool(metadata.get("test_accessed", True)):
        raise RuntimeError("O-K0 P0 baseline role or split contract failed")
    return cache


def assert_alignment(hm: Mapping[str, object], p0: Mapping[str, object]) -> None:
    if list(hm["image_ids"]) != list(p0["image_ids"]):
        raise RuntimeError("HM and P0 image rows differ")
    for name in ("labels", "bboxes", "areas"):
        if not torch.equal(torch.as_tensor(hm[name]), torch.as_tensor(p0[name])):
            raise RuntimeError(f"HM and P0 {name} rows differ")
    if list(hm["events"]) != list(p0["events"]):
        raise RuntimeError("HM and P0 event rows differ")


def event_codes(train_events: list[str], query_events: list[str]) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    names = sorted(set(train_events))
    if not set(query_events).issubset(names):
        raise RuntimeError("Validation contains an event absent from Train")
    mapping = {name: index for index, name in enumerate(names)}
    return (
        torch.tensor([mapping[name] for name in train_events], dtype=torch.long),
        torch.tensor([mapping[name] for name in query_events], dtype=torch.long),
        mapping,
    )


def main() -> None:
    parser = argparse.ArgumentParser("HM O-K0 cross-event Occurrence key audit")
    parser.add_argument("--train_key_cache", required=True)
    parser.add_argument("--validation_key_cache", required=True)
    parser.add_argument("--train_p0_cache", required=True)
    parser.add_argument("--validation_p0_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cap_per_event_class", type=int, default=500)
    parser.add_argument("--event_probe_cap", type=int, default=1000)
    parser.add_argument("--relation_pairs", type=int, default=60000)
    parser.add_argument("--minimum_global_auc", type=float, default=0.70)
    parser.add_argument("--minimum_event_auc", type=float, default=0.60)
    parser.add_argument("--minimum_consistent_events", type=int, default=10)
    parser.add_argument("--minimum_auc_gain", type=float, default=0.05)
    parser.add_argument("--maximum_event_probe_increase", type=float, default=0.02)
    args = parser.parse_args()

    train_hm = load_cache(args.train_key_cache, "train")
    val_hm = load_cache(args.validation_key_cache, "validation")
    train_p0 = load_p0(args.train_p0_cache, "train")
    val_p0 = load_p0(args.validation_p0_cache, "validation")
    assert_alignment(train_hm, train_p0)
    assert_alignment(val_hm, val_p0)
    for key in ("p3_checkpoint_sha256", "s2o_checkpoint_sha256", "s2o_step"):
        if train_hm["metadata"].get(key) != val_hm["metadata"].get(key):
            raise RuntimeError(f"Train/Validation O-K0 {key} differs")

    train_labels = (torch.as_tensor(train_hm["labels"]).long() > 0).long()
    val_labels = (torch.as_tensor(val_hm["labels"]).long() > 0).long()
    train_names = list(map(str, train_hm["events"]))
    val_names = list(map(str, val_hm["events"]))
    train_events, val_events, mapping = event_codes(train_names, val_names)

    train_variants = {
        "old_p3_mean": train_p0["key_variants"]["mean"].float(),
        **{name: value.float() for name, value in train_hm["key_variants"].items()},
    }
    val_variants = {
        "old_p3_mean": val_p0["key_variants"]["mean"].float(),
        **{name: value.float() for name, value in val_hm["key_variants"].items()},
    }
    if tuple(train_variants) != tuple(val_variants):
        raise RuntimeError("Train/Validation O-K0 variants differ")

    train_results = {}
    validation_results = {}
    for variant, train_feature in train_variants.items():
        val_feature = val_variants[variant]
        oof_scores = loeo_occurrence_scores(
            train_feature,
            train_labels,
            train_events,
            cap_per_event_class=args.cap_per_event_class,
            seed=args.seed,
        )
        train_auc = occurrence_auc_report(oof_scores, train_labels, train_names)
        fit_indices = event_class_capped_indices(
            train_labels,
            train_events,
            cap_per_event_class=args.cap_per_event_class,
            seed=args.seed,
        )
        centres = fit_binary_centres(train_feature, train_labels, fit_indices)
        val_scores = binary_centre_scores(val_feature, centres)
        val_auc = occurrence_auc_report(val_scores, val_labels, val_names)
        event_probe = event_probe_report(
            train_feature,
            train_events,
            val_feature,
            val_events,
            cap_per_event=args.event_probe_cap,
            seed=args.seed,
        )
        train_relation = relation_statistics(
            train_feature,
            train_labels,
            train_events,
            num_pairs=args.relation_pairs,
            seed=args.seed,
        )
        validation_relation = relation_statistics(
            val_feature,
            val_labels,
            val_events,
            num_pairs=args.relation_pairs,
            seed=args.seed,
        )
        train_results[variant] = {
            "out_of_event_auc": train_auc,
            "class_over_event_geometry": train_relation,
        }
        validation_results[variant] = {
            "auc": val_auc,
            "event_probe": event_probe,
            "class_over_event_geometry": validation_relation,
        }

    candidates = [name for name in train_variants if name != "old_p3_mean"]
    selected = max(
        candidates,
        key=lambda name: (
            train_results[name]["out_of_event_auc"]["event_macro_auroc"],
            train_results[name]["out_of_event_auc"]["global_auroc"],
            train_results[name]["out_of_event_auc"]["direction_consistent_events"],
            train_results[name]["class_over_event_geometry"]["delta_cross_event"],
        ),
    )
    baseline = validation_results["old_p3_mean"]
    chosen = validation_results[selected]
    auc_gain = chosen["auc"]["global_auroc"] - baseline["auc"]["global_auroc"]
    event_probe_increase = (
        chosen["event_probe"]["balanced_accuracy"]
        - baseline["event_probe"]["balanced_accuracy"]
    )
    checks = {
        "global_occurrence_auroc_at_least_declared": (
            chosen["auc"]["global_auroc"] >= args.minimum_global_auc
        ),
        "event_occurrence_auroc_at_least_declared": (
            chosen["auc"]["event_macro_auroc"] >= args.minimum_event_auc
        ),
        "direction_consistent_events_at_least_declared": (
            chosen["auc"]["direction_consistent_events"]
            >= args.minimum_consistent_events
        ),
        "global_auc_gain_over_old_mean_at_least_declared": (
            auc_gain >= args.minimum_auc_gain
        ),
        "event_probe_does_not_increase_beyond_tolerance": (
            event_probe_increase <= args.maximum_event_probe_increase
        ),
    }
    passed = all(checks.values())
    payload = {
        "protocol": "HM_OK0_train_loeo_locked_stage2_occurrence_key_audit_v1",
        "selection_contract": (
            "S2O pooling/temperature selected by Train leave-one-event-out "
            "Event AUROC; Validation never selects a variant"
        ),
        "selected_on_train": selected,
        "event_map": mapping,
        "train_results": train_results,
        "validation_results": validation_results,
        "validation_diagnostics": {
            "global_auc_gain_vs_old_p3_mean": auc_gain,
            "event_probe_balanced_accuracy_increase_vs_old_p3_mean": event_probe_increase,
        },
        "thresholds": {
            "minimum_global_auc": args.minimum_global_auc,
            "minimum_event_auc": args.minimum_event_auc,
            "minimum_consistent_events": args.minimum_consistent_events,
            "minimum_auc_gain": args.minimum_auc_gain,
            "maximum_event_probe_increase": args.maximum_event_probe_increase,
        },
        "checks": checks,
        "passed": passed,
        "decision": (
            "ADMIT_TO_HM_OM0_EVENT_DEBIASED_PROTOTYPES_NO_FUSION"
            if passed
            else "STOP_HIERARCHICAL_MEMORY_OCCURRENCE_KEY_INSUFFICIENT"
        ),
        "p3_checkpoint_sha256": train_hm["metadata"]["p3_checkpoint_sha256"],
        "s2o_checkpoint_sha256": train_hm["metadata"]["s2o_checkpoint_sha256"],
        "s2o_step": train_hm["metadata"]["s2o_step"],
        "validation_selects_key": False,
        "fuses_p3": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 120)
    print("HM O-K0 train-locked Stage2 Occurrence-key audit")
    print(f"Selected on Train : {selected}")
    for variant, result in validation_results.items():
        auc = result["auc"]
        probe = result["event_probe"]
        delta = result["class_over_event_geometry"]["delta_cross_event"]
        print(
            f"{variant:<24} | global={auc['global_auroc']:.4f} | "
            f"event={auc['event_macro_auroc']:.4f} | "
            f"consistent={auc['direction_consistent_events']}/{auc['comparable_events']} | "
            f"event-probe={probe['balanced_accuracy']:.4f} | DeltaCE={delta:+.4f}"
        )
    print(f"Checks            : {checks}")
    print(f"Decision          : {payload['decision']}")
    print(f"Saved             : {output}")
    print("Test accessed     : False")


if __name__ == "__main__":
    main()
