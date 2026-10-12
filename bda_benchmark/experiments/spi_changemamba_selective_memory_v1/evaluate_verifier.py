#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from bda_benchmark.script.standard_ML.train_S5_S8_GPU import metrics_from_hist

from . import PROTOCOL
from .core import valid_miou
from .verifier_core import LogisticVerifier, apply, filter_rows, matrix, select


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate frozen verifier on read-only Val")
    parser.add_argument("--utility", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    cache = torch.load(args.utility, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if cache.get("split") != "validation" or cache.get("allowed_for_verifier_training", True):
        raise RuntimeError("Evaluation requires read-only Validation utility")
    if checkpoint.get("protocol") != PROTOCOL + "_VERIFIER":
        raise RuntimeError("unexpected verifier checkpoint")
    rows = filter_rows(cache["rows"], checkpoint["variant"])
    value = matrix(rows, checkpoint["feature_names"])
    value = (value - checkpoint["mean"]) / checkpoint["std"]
    model = LogisticVerifier(value.shape[1])
    model.load_state_dict(checkpoint["model"], strict=True)
    with torch.no_grad():
        scores = torch.sigmoid(model(value)).tolist()
    chosen = select(rows, scores, checkpoint["threshold"], checkpoint["conflict_margin"])
    final_hist = apply(cache["base_hist"], rows, chosen)
    base_metrics = metrics_from_hist(cache["loc_hist"], cache["base_hist"])
    metrics = metrics_from_hist(cache["loc_hist"], final_hist)
    event_delta = defaultdict(lambda: torch.zeros(4, 4, dtype=torch.float64))
    sources = defaultdict(int)
    for index in chosen:
        event_delta[str(rows[index]["event"])] += rows[index]["delta"].double()
        sources[str(rows[index]["source"])] += 1
    event_gain, event_final = {}, {}
    for event, hist in cache["event_base_hist"].items():
        updated = hist.double() + event_delta[str(event)]
        event_gain[str(event)] = 100.0 * (valid_miou(updated) - valid_miou(hist))
        event_final[str(event)] = valid_miou(updated)
    region_count = len({str(row["region_key"]) for row in rows})
    result = {
        "protocol": PROTOCOL + "_VAL", "variant": checkpoint["variant"],
        "base_metrics": base_metrics, "metrics": metrics,
        "gain_pp": 100.0 * (valid_miou(final_hist) - valid_miou(cache["base_hist"])),
        "threshold": checkpoint["threshold"], "candidate_count": len(rows),
        "accepted_count": len(chosen), "acceptance_rate": len(chosen) / max(region_count, 1),
        "fallback_rate": 1.0 - len(chosen) / max(region_count, 1),
        "source_accepted": dict(sources), "event_gain_pp": event_gain,
        "event_macro_miou": sum(event_final.values()) / max(len(event_final), 1),
        "worst_event_miou": min(event_final.values()) if event_final else 0.0,
        "nonnegative_events": sum(value >= 0.0 for value in event_gain.values()),
        "event_count": len(event_gain), "test_accessed": False,
    }
    result["checks"] = {
        "gain_ge_0_20pp": result["gain_pp"] >= 0.20,
        "damaged_f1_nonnegative": metrics.get("damaged_f1", 0.0) >= base_metrics.get("damaged_f1", 0.0),
        "events_nonnegative_ge_10": result["nonnegative_events"] >= 10,
        "acceptance_le_70pct": result["acceptance_rate"] <= 0.70,
    }
    result["pass"] = all(result["checks"].values())
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{checkpoint['variant']} mIoU={100*metrics['valid_miou']:.4f} "
        f"gain={result['gain_pp']:+.4f}pp D-F1={100*metrics.get('damaged_f1', 0):.4f} "
        f"events={result['nonnegative_events']}/{result['event_count']} "
        f"accept={100*result['acceptance_rate']:.2f}%"
    )
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
