#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from bda_benchmark.script.standard_ML.train_S5_S8_GPU import metrics_from_hist

from . import PROTOCOL
from .core import (
    LogisticVerifier, apply_selected, matrix_from_rows, native, proposal_filter,
    select_proposals, valid_miou,
)


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate a frozen verifier on read-only Val450 targets")
    parser.add_argument("--targets", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    cache = torch.load(args.targets, map_location="cpu", weights_only=False)
    if cache.get("protocol") != PROTOCOL + "_METRIC_TARGETS":
        raise RuntimeError(f"unexpected target protocol: {cache.get('protocol')!r}")
    if cache.get("split") != "validation" or cache.get("allowed_for_verifier_training", True):
        raise RuntimeError("evaluation requires a read-only validation cache")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != PROTOCOL + "_CHECKPOINT":
        raise RuntimeError(f"unexpected verifier protocol: {checkpoint.get('protocol')!r}")
    rows = proposal_filter(cache["rows"], checkpoint["variant"])
    value = matrix_from_rows(rows, checkpoint["feature_names"])
    value = (value - checkpoint["mean"]) / checkpoint["std"]
    model = LogisticVerifier(value.shape[1]); model.load_state_dict(checkpoint["model"], strict=True); model.eval()
    with torch.no_grad(): scores = torch.sigmoid(model(value)).tolist()
    selected = select_proposals(
        rows, scores, threshold=float(checkpoint["threshold"]),
        conflict_margin=float(checkpoint["conflict_margin"]),
    )
    final_hist = apply_selected(cache["base_hist"], rows, selected)
    base_metrics = metrics_from_hist(cache["loc_hist"], cache["base_hist"])
    metrics = metrics_from_hist(cache["loc_hist"], final_hist)
    selected_set = set(selected)
    event_delta = defaultdict(lambda: torch.zeros(4, 4, dtype=torch.float64))
    source_selected = defaultdict(int)
    affected_recovered = harmed = rescued = changed = 0
    for index in selected:
        row = rows[index]
        event_delta[str(row["event"])] += torch.as_tensor(row["delta"], dtype=torch.float64)
        source_selected[str(row["source"])] += 1
        affected_recovered += int(row["affected_recovered"]); harmed += int(row["harmed"])
        rescued += int(row["rescued"]); changed += int(row["changed"])
    event_gain = {}
    event_final = {}
    for event, base in cache["event_base_hist"].items():
        updated = base.double() + event_delta[str(event)]
        event_gain[str(event)] = 100.0 * (valid_miou(updated) - valid_miou(base))
        event_final[str(event)] = valid_miou(updated)
    region_count = len({str(row["region_key"]) for row in rows})
    result = {
        "protocol": PROTOCOL + "_VAL450", "variant": checkpoint["variant"],
        "base_metrics": base_metrics, "metrics": metrics,
        "gain_pp": 100.0 * (valid_miou(final_hist) - valid_miou(cache["base_hist"])),
        "threshold": checkpoint["threshold"], "candidate_count": len(rows),
        "candidate_region_count": region_count, "accepted_count": len(selected),
        "acceptance_rate": len(selected) / max(region_count, 1),
        "fallback_rate": 1.0 - len(selected) / max(region_count, 1),
        "source_accepted": dict(source_selected), "rescued_exact_pixels": rescued,
        "harmed_exact_pixels": harmed, "changed_pixels": changed,
        "correction_precision": rescued / max(rescued + harmed, 1),
        "affected_miss_recovered": affected_recovered,
        "event_gain_pp": event_gain, "event_final_miou": event_final,
        "event_macro_miou": sum(event_final.values()) / max(len(event_final), 1),
        "worst_event_miou": min(event_final.values()) if event_final else 0.0,
        "nonnegative_events": sum(value >= 0.0 for value in event_gain.values()),
        "improved_events": sum(value > 0.0 for value in event_gain.values()),
        "event_count": len(event_gain),
        "checks": {
            "gain_ge_0_20pp": 100.0 * (valid_miou(final_hist) - valid_miou(cache["base_hist"])) >= 0.20,
            "damaged_f1_nonnegative": metrics.get("damaged_f1", 0.0) >= base_metrics.get("damaged_f1", 0.0),
            "acceptance_le_70pct": len(selected) / max(region_count, 1) <= 0.70,
            "events_nonnegative_ge_10": sum(value >= 0.0 for value in event_gain.values()) >= 10,
        },
        "data": cache["metadata"], "test_accessed": False,
    }
    result["pass"] = all(result["checks"].values())
    destination = Path(args.output_json); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(native(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{checkpoint['variant']} | mIoU={100*metrics['valid_miou']:.4f} "
        f"gain={result['gain_pp']:+.4f}pp D-F1={100*metrics.get('damaged_f1', 0):.4f} "
        f"events={result['nonnegative_events']}/{result['event_count']} accept={100*result['acceptance_rate']:.2f}%"
    )
    print(f"Saved: {destination}")


if __name__ == "__main__": main()
