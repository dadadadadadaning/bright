"""Summarize only a passed, immutable, full-Val450 bank/retrieval factorial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import FACTORIAL_ARMS, FORMAT, PAIRS


def read_formal(output):
    from bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.io_utils import (
        digest, safe_child)
    output = Path(output)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary.get("audit_file") != "audit.json":
        raise ValueError("Summary does not reference a completed formal audit")
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    if (summary.get("format") != FORMAT or audit.get("format") != FORMAT
            or summary.get("complete") is not True
            or audit.get("complete") is not True
            or summary.get("full_validation_evidence") is not True
            or audit.get("full_validation_evidence") is not True
            or summary.get("image_count") != 450
            or audit.get("image_count") != 450
            or audit.get("reproduction", {}).get("passed") is not True
            or audit.get("test_accessed") is not False
            or audit.get("training") is not False
            or not all(audit.get("immutable", {}).get(key) is True for key in
                       ("files", "inherited_code", "new_code", "frozen_models",
                        "runtime", "protocol", "new_bank"))):
        raise ValueError("The frozen Val450 factorial failed formal evidence gates")
    if audit.get("factorial_arms") != list(FACTORIAL_ARMS):
        raise ValueError("Audit is missing one of the four fixed factorial arms")
    if any(arm not in audit.get("metrics", {}) for arm in
           ("a1_pixel", "grid_probq", "j1", *FACTORIAL_ARMS)):
        raise ValueError("Metrics are incomplete")
    if any(pair not in audit.get("transitions", {}) for pair in PAIRS):
        raise ValueError("Paired transition evidence is incomplete")
    journal = output / "images.jsonl"
    if digest(journal) != audit.get("image_journal_sha256"):
        raise ValueError("Val image journal changed after the formal audit")
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != 450 or [row.get("index") for row in rows] != list(range(450)):
        raise ValueError("Val image journal is incomplete or out of order")
    for row in rows:
        path = safe_child(output, row.get("prediction_file"))
        if digest(path) != row.get("prediction_sha256"):
            raise ValueError("Val prediction/support artifact changed: " + str(path))
    return audit


def factorial_table(audit):
    rows = []
    for arm in ("a1_pixel", "grid_probq", "j1", *FACTORIAL_ARMS):
        score = audit["metrics"][arm]
        support = audit.get("support_diagnostics", {}).get(arm, {})
        source = audit.get("support_source_changes_vs_j3", {}).get(arm, {})
        rows.append(dict(
            arm=arm,
            foreground_miou_percent=100 * score["valid_miou"],
            delta_vs_a1_pp=100 * (score["valid_miou"] - audit["metrics"]["a1_pixel"]["valid_miou"]),
            delta_vs_grid_probq_pp=100 * (score["valid_miou"] - audit["metrics"]["grid_probq"]["valid_miou"]),
            delta_vs_old_j3_pp=100 * (score["valid_miou"] - audit["metrics"]["j3"]["valid_miou"]),
            intact_iou_percent=100 * score["iou"][1],
            damaged_iou_percent=100 * score["iou"][2],
            destroyed_iou_percent=100 * score["iou"][3],
            mean_valid_supports=support.get("mean_valid_slots_per_query"),
            unique_images=support.get("mean_unique_images_per_query"),
            unique_events=support.get("mean_unique_events_per_query"),
            mean_support_purity=support.get("mean_support_purity"),
            queries_with_rank_change_vs_j3=source.get("queries_with_rank_change"),
            queries_with_set_change_vs_j3=source.get("queries_with_set_change"),
        ))
    return rows


def paired_table(audit):
    rows = []
    for pair in PAIRS:
        before, after = pair.split("__")
        evidence = audit["transitions"][pair]
        damaged = evidence["per_class"]["Damaged"]
        intact = evidence["per_class"]["Intact"]
        joint = evidence["joint_gt_reference_candidate"]
        rows.append(dict(
            pair=pair,
            miou_delta_pp=100 * (audit["metrics"][after]["valid_miou"]
                                 - audit["metrics"][before]["valid_miou"]),
            damaged_rescue=damaged["rescue"],
            damaged_harm=damaged["harm"],
            damaged_net_false_positive=damaged["delta_fp"],
            intact_rescue=intact["rescue"],
            intact_harm=intact["harm"],
            intact_correct_to_damaged=int(joint[1][1][2]),
        ))
    return rows


def factor_effects(audit):
    score = {arm: audit["metrics"][arm]["valid_miou"] for arm in FACTORIAL_ARMS}
    old_normal = score["j3"]
    old_diverse = score["j3_old_diverse"]
    new_normal = score["j3_quality_top8"]
    new_diverse = score["j3_quality_diverse"]
    return dict(
        quality_bank_effect_at_normal_top8_pp=100 * (new_normal - old_normal),
        diverse_top8_effect_on_old_bank_pp=100 * (old_diverse - old_normal),
        diverse_top8_effect_on_quality_bank_pp=100 * (new_diverse - new_normal),
        bank_selection_x_retrieval_interaction_pp=100 * ((new_diverse - new_normal)
                                                           - (old_diverse - old_normal)),
    )


def compat_table(audit):
    rows = []
    for arm in FACTORIAL_ARMS:
        bins = audit["posthoc_region_label_compatibility"][arm]["soft_compatibility"]
        for name in ("all", "foreground_pure", "foreground_mixed",
                     "background_diluted", "mostly_background"):
            item = bins[name]
            n = item["supported_queries"]
            intact = item["intact_queries"]
            damaged = item["damaged_queries"]
            rows.append(dict(
                arm=arm, q_bin=name, supported_q=n,
                mean_soft_agreement=(item["compatibility_sum"] / n if n else None),
                dominant_agreement=(item["dominant_agree"] / n if n else None),
                intact_support_prefers_damaged=(item["intact_support_prefers_damaged"] / intact
                                                 if intact else None),
                damaged_support_prefers_intact=(item["damaged_support_prefers_intact"] / damaged
                                                if damaged else None),
            ))
    return rows


def event_wins(audit):
    reference = "j3"
    results = {}
    for arm in FACTORIAL_ARMS[1:]:
        values = []
        intact_correct_to_damaged = {}
        damaged_rescue = {}
        damaged_harm = {}
        for event_name, metrics in audit["per_event"].items():
            old = metrics[reference]["valid_miou"]
            new = metrics[arm]["valid_miou"]
            if old is not None and new is not None:
                values.append((event_name, 100 * (new - old)))
            transition = audit["per_event_transitions_vs_j3"][event_name][arm]
            intact_correct_to_damaged[event_name] = int(
                transition["joint_gt_reference_candidate"][1][1][2])
            damaged_rescue[event_name] = int(transition["per_class"]["Damaged"]["rescue"])
            damaged_harm[event_name] = int(transition["per_class"]["Damaged"]["harm"])
        results[arm] = dict(events=len(values), improved=sum(delta > 0 for _, delta in values),
                            worsened=sum(delta < 0 for _, delta in values),
                            unchanged=sum(delta == 0 for _, delta in values),
                            per_event_delta_pp=dict(values),
                            intact_correct_to_damaged=intact_correct_to_damaged,
                            damaged_rescue=damaged_rescue, damaged_harm=damaged_harm)
    return results


def print_table(rows, columns):
    print("\t".join(columns))
    for row in rows:
        print("\t".join("-" if row[key] is None else
                        f"{row[key]:.6f}" if isinstance(row[key], float)
                        else str(row[key]) for key in columns))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    audit = read_formal(args.output)
    print("Four frozen J3 bank x retrieval cells; percentages are pixel-level Val450.")
    rows = factorial_table(audit)
    print_table(rows, tuple(rows[0]))
    print("\nPaired pixel transitions; each pair uses the same image/front/Q/J3 Reader.")
    pairs = paired_table(audit)
    print_table(pairs, tuple(pairs[0]))
    print("\nFactor effects in foreground mIoU percentage points:")
    for key, value in factor_effects(audit).items():
        print(f"{key}={value:.6f}")
    print("\nPost-hoc Q soft-label compatibility; Q regions are not building instances.")
    print("Compare four arms within each Q stratum; pure/mixed means have different intrinsic maxima.")
    compat = compat_table(audit)
    print_table(compat, tuple(compat[0]))
    print("\nPer-event mIoU change vs original J3:")
    for arm, row in event_wins(audit).items():
        print(arm, "improved=", row["improved"], "worsened=", row["worsened"],
              "unchanged=", row["unchanged"], "events=", row["events"],
              "intact_correct_to_damaged=", sum(row["intact_correct_to_damaged"].values()))
    print("Full per-event metrics, support IDs, predictions and SHA256: audit.json/images.jsonl/predictions/*.npz")
    print("Interpretation: frozen Reader input interventions; new bank may shift its input distribution.")


if __name__ == "__main__":
    main()
