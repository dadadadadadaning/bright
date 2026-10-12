#!/usr/bin/env python3
"""Compare two frozen audits and apply the declared module admission gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bda_benchmark.experiments.p5_ablation_validation_v1.metrics import (
    compare_events,
    paired_bootstrap_gain,
)


def load(path: str):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("metrics", "events", "image_rows")
    if any(key not in payload for key in required):
        raise ValueError(f"{path} is not a mechanism-audit JSON")
    return payload


def parse_args():
    parser = argparse.ArgumentParser("Compare two P5 controlled audits")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--gate", choices=("sar_filter", "filter_with_attention",
                                           "cross_attention", "hierarchical", "final"),
                        required=True)
    parser.add_argument("--reference_perturbed", default="")
    parser.add_argument("--candidate_perturbed", default="")
    parser.add_argument("--reference_shuffled", default="")
    parser.add_argument("--candidate_shuffled", default="")
    parser.add_argument("--minimum_shuffle_drop_pp", type=float, default=0.10)
    parser.add_argument("--bootstrap_repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    reference, candidate = load(args.reference), load(args.candidate)
    if reference.get("split") != candidate.get("split"):
        raise ValueError("reference and candidate use different splits")
    event = compare_events(candidate["events"], reference["events"])
    bootstrap = paired_bootstrap_gain(candidate["image_rows"], reference["image_rows"],
                                      args.bootstrap_repeats, args.seed)
    ref_m = reference["metrics"]
    cand_m = candidate["metrics"]
    gain_pp = 100.0 * (cand_m["foreground_miou"] - ref_m["foreground_miou"])
    damaged_f1_gain_pp = 100.0 * (cand_m["damaged_f1"] - ref_m["damaged_f1"])
    perturbation = None
    shuffle = None
    if bool(args.reference_perturbed) != bool(args.candidate_perturbed):
        raise ValueError("provide both perturbed audits or neither")
    if args.reference_perturbed:
        ref_p, cand_p = load(args.reference_perturbed), load(args.candidate_perturbed)
        ref_drop = 100.0 * (ref_m["foreground_miou"] - ref_p["metrics"]["foreground_miou"])
        cand_drop = 100.0 * (cand_m["foreground_miou"] - cand_p["metrics"]["foreground_miou"])
        perturbation = {
            "reference_drop_pp": ref_drop,
            "candidate_drop_pp": cand_drop,
            "candidate_is_more_stable": cand_drop <= ref_drop,
        }
    if bool(args.reference_shuffled) != bool(args.candidate_shuffled):
        raise ValueError("provide both shuffled audits or neither")
    if args.reference_shuffled:
        ref_s, cand_s = load(args.reference_shuffled), load(args.candidate_shuffled)
        shuffle = {
            "reference_drop_pp": 100.0 * (
                ref_m["foreground_miou"] - ref_s["metrics"]["foreground_miou"]
            ),
            "candidate_drop_pp": 100.0 * (
                cand_m["foreground_miou"] - cand_s["metrics"]["foreground_miou"]
            ),
        }

    checks = {
        "ordinary_accuracy_improved": gain_pp > 0.0,
        "damaged_f1_nonnegative": damaged_f1_gain_pp >= 0.0,
    }
    if args.gate == "sar_filter":
        checks["c1_above_c0"] = gain_pp > 0.0
        checks["filter_views_are_distinct"] = float(
            candidate.get("sar_view_diagnostics", {}).get("view_diversity_abs", 0.0)
        ) > float(reference.get("sar_view_diagnostics", {}).get("view_diversity_abs", 0.0))
    elif args.gate == "filter_with_attention":
        checks["filtered_attention_above_raw_attention"] = gain_pp > 0.0
        checks["filter_views_are_distinct"] = float(
            candidate.get("sar_view_diagnostics", {}).get("view_diversity_abs", 0.0)
        ) > float(reference.get("sar_view_diagnostics", {}).get("view_diversity_abs", 0.0))
    elif args.gate == "cross_attention":
        checks["c2_above_c1"] = gain_pp > 0.0
        checks["misregistration_more_stable"] = bool(
            perturbation and perturbation["candidate_is_more_stable"]
        )
        checks["aligned_relation_is_used"] = bool(
            shuffle and shuffle["candidate_drop_pp"] >= args.minimum_shuffle_drop_pp
        )
    elif args.gate == "hierarchical":
        checks["gain_at_least_0_30pp"] = gain_pp >= 0.30
        ref_hist, cand_hist = reference["confusion"], candidate["confusion"]
        ref_den = sum(sum(ref_hist[row]) for row in (2, 3))
        cand_den = sum(sum(cand_hist[row]) for row in (2, 3))
        ref_miss = (ref_hist[2][1] + ref_hist[3][1]) / max(ref_den, 1)
        cand_miss = (cand_hist[2][1] + cand_hist[3][1]) / max(cand_den, 1)
        checks["affected_to_intact_nonincrease"] = cand_miss <= ref_miss
    elif args.gate == "final":
        checks["at_least_10_events_nonnegative"] = event["nonnegative_events"] >= 10
        checks["bootstrap_ci_low_above_zero"] = bootstrap["ci95_low_pp"] > 0.0

    payload = {
        "protocol": "P5_ABLATION_ADMISSION_v1",
        "gate": args.gate,
        "reference": args.reference,
        "candidate": args.candidate,
        "gain_pp": gain_pp,
        "damaged_f1_gain_pp": damaged_f1_gain_pp,
        "event_macro_miou": sum(
            item["metrics"]["foreground_miou"] for item in candidate["events"].values()
        ) / max(len(candidate["events"]), 1),
        "worst_event_miou": min(
            (item["metrics"]["foreground_miou"] for item in candidate["events"].values()),
            default=0.0,
        ),
        "events": event,
        "bootstrap": bootstrap,
        "perturbation": perturbation,
        "shuffle": shuffle,
        "checks": checks,
        "pass": all(checks.values()),
        "test_accessed": False,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path = Path(args.output_md) if args.output_md else output.with_suffix(".md")
    lines = [
        f"# {args.gate} admission",
        "",
        f"- Decision: **{'PASS' if payload['pass'] else 'FAIL'}**",
        f"- Foreground mIoU gain: {gain_pp:+.4f} pp",
        f"- Damaged F1 gain: {damaged_f1_gain_pp:+.4f} pp",
        f"- Nonnegative events: {event['nonnegative_events']}/{event['event_count']}",
        f"- Bootstrap 95% CI: [{bootstrap['ci95_low_pp']:+.4f}, "
        f"{bootstrap['ci95_high_pp']:+.4f}] pp",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- {'PASS' if passed else 'FAIL'}: `{name}`"
                 for name, passed in checks.items())
    if perturbation:
        lines.extend([
            "", "## Perturbation stability", "",
            f"- Reference drop: {perturbation['reference_drop_pp']:.4f} pp",
            f"- Candidate drop: {perturbation['candidate_drop_pp']:.4f} pp",
        ])
    if shuffle:
        lines.extend([
            "", "## Wrong-pair sensitivity", "",
            f"- Reference drop: {shuffle['reference_drop_pp']:.4f} pp",
            f"- Candidate drop: {shuffle['candidate_drop_pp']:.4f} pp",
        ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Saved: {output}\nSaved: {md_path}", flush=True)


if __name__ == "__main__":
    main()
