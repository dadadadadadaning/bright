# -*- coding: utf-8 -*-
"""CMR-E0: frozen-P3 Occurrence then Severity diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from script.standard_ML.cmr_hierarchy_runtime import binary_metrics, event_binary_metrics


def main() -> None:
    parser = argparse.ArgumentParser("CMR-E0 frozen P3 hierarchy audit")
    parser.add_argument("--validation_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    cache = torch.load(args.validation_cache, map_location="cpu", weights_only=False)
    metadata = cache["metadata"]
    if metadata.get("protocol") != "CMR_P0_frozen_pooling_audit_v1" or metadata.get("role") != "query":
        raise RuntimeError("E0 requires the frozen P0 Validation query cache")
    if bool(metadata.get("test_accessed", True)):
        raise RuntimeError("E0 refuses Test-derived input")
    label = cache["labels"].long().flatten()
    probability = cache["base_probabilities"].float()
    prediction = cache["base_predictions"].long().flatten()
    events = list(map(str, cache["events"]))
    if probability.shape != (label.numel(), 3):
        raise RuntimeError("E0 expects conditional I/D/X probabilities")

    affected_target = label > 0
    affected_score = probability[:, 1] + probability[:, 2]
    occurrence = {
        "global": binary_metrics(affected_score, affected_target, args.threshold),
        **event_binary_metrics(affected_score, affected_target, events, args.threshold),
    }
    affected_indices = torch.nonzero(affected_target, as_tuple=False).flatten()
    severity_target = label[affected_indices] == 2
    severity_score = probability[affected_indices, 2] / (
        probability[affected_indices, 1] + probability[affected_indices, 2]
    ).clamp_min(1e-8)
    severity_events = [events[index] for index in affected_indices.tolist()]
    severity = {
        "global": binary_metrics(severity_score, severity_target, args.threshold),
        **event_binary_metrics(severity_score, severity_target, severity_events, args.threshold),
    }

    damaged = label == 1
    destroyed = label == 2
    d_to_i = int((damaged & (prediction == 0)).sum())
    x_to_i = int((destroyed & (prediction == 0)).sum())
    conditional_severity_prediction = probability[:, 2] >= probability[:, 1]
    severity_confusion = torch.zeros(2, 2, dtype=torch.long)
    true_dx = (label[affected_indices] == 2).long()
    pred_dx = conditional_severity_prediction[affected_indices].long()
    severity_confusion += torch.bincount(true_dx * 2 + pred_dx, minlength=4).reshape(2, 2)

    occurrence_event_auc = float(occurrence["event_macro_auroc"])
    severity_event_auc = float(severity["event_macro_auroc"])
    if occurrence_event_auc + 0.05 < severity_event_auc:
        diagnosis = "OCCURRENCE_IS_PRIMARY_BOTTLENECK"
    elif severity_event_auc + 0.05 < occurrence_event_auc:
        diagnosis = "SEVERITY_IS_PRIMARY_BOTTLENECK"
    else:
        diagnosis = "OCCURRENCE_AND_SEVERITY_REQUIRE_JOINT_CAUSAL_AUDIT"
    payload = {
        "protocol": "CMR_E0_frozen_p3_occurrence_severity_audit_v1",
        "scope": "Validation diagnostic only; no fitting and no prediction changes",
        "regions": int(label.numel()),
        "class_counts": torch.bincount(label, minlength=3).tolist(),
        "occurrence_intact_vs_affected": occurrence,
        "severity_damaged_vs_destroyed_given_gt_affected": severity,
        "damaged_to_intact": {
            "count": d_to_i,
            "rate": d_to_i / max(int(damaged.sum()), 1),
        },
        "destroyed_to_intact": {
            "count": x_to_i,
            "rate": x_to_i / max(int(destroyed.sum()), 1),
        },
        "conditional_severity_confusion_D_X": severity_confusion.tolist(),
        "diagnosis": diagnosis,
        "trains_model": False,
        "fuses_p3": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 104)
    print("CMR-E0 frozen P3 Occurrence -> Severity audit")
    print(
        f"Occurrence | AUC={occurrence['global']['auroc']:.4f} | "
        f"event={occurrence['event_macro_auroc']:.4f} | F1={occurrence['global']['f1']:.4f}"
    )
    print(
        f"Severity   | AUC={severity['global']['auroc']:.4f} | "
        f"event={severity['event_macro_auroc']:.4f} | F1={severity['global']['f1']:.4f}"
    )
    print(f"D->I / X->I : {100*payload['damaged_to_intact']['rate']:.2f}% / {100*payload['destroyed_to_intact']['rate']:.2f}%")
    print(f"Diagnosis     : {diagnosis}")
    print(f"Saved         : {output}")
    print("Test accessed : False")


if __name__ == "__main__":
    main()

