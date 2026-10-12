# -*- coding: utf-8 -*-
"""CMR-H0B: audit a capped, rotating Train-only anchor schedule."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from script.standard_ML.cmr_event_capped_sampler import (
    audit_event_capped_schedules,
    event_anchor_pools,
    event_occurrence_statistics,
    h0b_gate,
)
from script.standard_ML.cmr_metric_utils import load_cmr_cache


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser("CMR-H0B event-capped Train sampler audit")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--h0_pairs", required=True)
    parser.add_argument("--manifest_output", required=True)
    parser.add_argument("--audit_output", required=True)
    parser.add_argument("--cap_per_event", type=int, default=64)
    parser.add_argument("--audit_epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    _, labels, events, event_map, cache = load_cmr_cache(
        args.train_cache, "mean", expected_role="train", print_schema=True
    )
    if "base_predictions" not in cache:
        raise KeyError("CMR-H0B requires frozen P3 base_predictions")
    base_predictions = cache["base_predictions"].long().flatten().cpu()
    pairs = torch.load(args.h0_pairs, map_location="cpu", weights_only=False)
    if pairs.get("protocol") != "CMR_H0_train_only_occurrence_hard_pairs_v1":
        raise RuntimeError("CMR-H0B refuses an unknown H0 pair protocol")
    if pairs.get("source_split") != "train":
        raise RuntimeError("CMR-H0B requires Train-only H0 pairs")
    if bool(pairs.get("validation_accessed", True)) or bool(pairs.get("test_accessed", True)):
        raise RuntimeError("CMR-H0B refuses pairs that accessed Validation/Test")
    cache_sha = cache["metadata"].get("p3_checkpoint_sha256")
    if pairs.get("p3_checkpoint_sha256") != cache_sha:
        raise RuntimeError("H0 pairs and Train cache use different P3 checkpoints")
    if pairs.get("event_map") != event_map:
        raise RuntimeError("H0 pairs and Train cache use different event maps")

    anchors = pairs["anchor_indices"].long().flatten().cpu()
    anchor_events = pairs["anchor_events"].long().flatten().cpu()
    if anchors.numel() != anchor_events.numel():
        raise RuntimeError("H0 anchor indices/events do not align")
    if not torch.equal(events[anchors], anchor_events):
        raise RuntimeError("H0 anchor event codes do not match Train cache")

    old_checks = dict(pairs.get("checks", {}))
    raw_share_check = "maximum_anchor_event_share_at_most_0_35"
    core_h0_checks = {
        name: bool(value)
        for name, value in old_checks.items()
        if name != raw_share_check
    }
    if not core_h0_checks:
        raise RuntimeError("H0 pair file does not contain declared core checks")

    occurrence = event_occurrence_statistics(
        labels,
        base_predictions,
        events,
        event_map,
        anchors,
        cap_per_event=args.cap_per_event,
    )
    schedule = audit_event_capped_schedules(
        anchor_events,
        cap_per_event=args.cap_per_event,
        seed=args.seed,
        audit_epochs=args.audit_epochs,
    )
    preview_rows = schedule.pop("preview_epoch_rows")
    checks = h0b_gate(core_h0_checks, occurrence, schedule)
    passed = all(checks.values())
    decision = (
        "ADMIT_CMR_H0B_TO_P3_H1_TRAIN_ONLY_DO_NOT_FUSE_OR_TEST"
        if passed
        else "STOP_OR_REVISE_EVENT_CAPPED_SAMPLER_DO_NOT_TRAIN_FUSE_OR_TEST"
    )

    pools = event_anchor_pools(anchor_events)
    manifest = {
        "protocol": "CMR_H0B_event_capped_rotating_train_manifest_v1",
        "source_split": "train",
        "train_cache": str(Path(args.train_cache).resolve()),
        "h0_pairs": str(Path(args.h0_pairs).resolve()),
        "h0_pairs_sha256": sha256(args.h0_pairs),
        "p3_checkpoint_sha256": cache_sha,
        "event_map": event_map,
        "event_anchor_rows": pools,
        "cap_per_event": int(args.cap_per_event),
        "seed": int(args.seed),
        "sampler_function": "build_event_capped_epoch",
        "large_event_rotation": "fixed_permutation_circular_blocks",
        "no_replacement_within_epoch": True,
        "preview_epoch": 0,
        "preview_epoch_rows": preview_rows,
        "preview_epoch_train_indices": anchors[preview_rows],
        "occurrence_audit": occurrence,
        "schedule_audit": schedule,
        "core_h0_checks": core_h0_checks,
        "checks": checks,
        "passed": passed,
        "inference_uses_memory": False,
        "validation_accessed": False,
        "test_accessed": False,
    }
    manifest_path = Path(args.manifest_output)
    atomic_torch_save(manifest, manifest_path)

    result = {
        "protocol": "CMR_H0B_event_capped_train_distribution_audit_v1",
        "manifest": str(manifest_path),
        "h0_pairs": str(Path(args.h0_pairs).resolve()),
        "occurrence_audit": occurrence,
        "schedule_audit": schedule,
        "core_h0_checks": core_h0_checks,
        "checks": checks,
        "passed": passed,
        "decision": decision,
        "inference_uses_memory": False,
        "validation_accessed": False,
        "test_accessed": False,
    }
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 116)
    print("CMR-H0B event-capped Train-only anchor distribution audit")
    print(
        f"Raw/effective anchors : {occurrence['raw_anchor_count']} / "
        f"{occurrence['effective_epoch_anchor_count']}"
    )
    print(
        f"Raw/effective max     : {100*occurrence['raw_maximum_event_share']:.2f}% / "
        f"{100*occurrence['effective_maximum_event_share']:.2f}%"
    )
    print(
        f"Audit rotation        : {schedule['audit_epochs']} epochs | "
        f"unique={schedule['unique_anchors_seen_across_audit_epochs']} "
        f"({100*schedule['unique_anchor_coverage_across_audit_epochs']:.2f}%)"
    )
    print("Per-event D->I audit:")
    ordered = sorted(
        occurrence["per_event"].items(),
        key=lambda item: item[1]["damaged_to_intact"],
        reverse=True,
    )
    for name, row in ordered:
        print(
            f"  {name:<28} | D={row['damaged_regions']:>5} | "
            f"D->I={row['damaged_to_intact']:>4} | "
            f"rate={100*row['damaged_to_intact_rate']:>6.2f}% | "
            f"epoch={row['effective_epoch_anchors']:>3} | "
            f"share={100*row['effective_epoch_share']:>5.2f}%"
        )
    print(f"Checks                 : {checks}")
    print(f"Decision               : {decision}")
    print(f"Saved manifest         : {manifest_path}")
    print(f"Saved audit            : {audit_path}")
    print("Validation accessed    : False")
    print("Test accessed          : False")


if __name__ == "__main__":
    main()
