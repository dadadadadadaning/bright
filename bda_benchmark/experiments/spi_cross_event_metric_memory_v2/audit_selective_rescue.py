from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import PROTOCOL
from .core import (
    CrossEventMetricEncoder,
    class_similarity_scores,
    classification_metrics,
    event_metrics,
    prepare_cache,
    probability_from_scores,
    selective_memory_probability,
)


ADVANTAGE_GRID = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25)
MARGIN_GRID = (0.05, 0.10, 0.15, 0.20, 0.30)


def evaluate_policy(
    data, parametric, memory, availability, topk, advantage, margin
):
    probability, intervene, diagnostics = selective_memory_probability(
        parametric,
        memory,
        availability,
        topk=topk,
        minimum_confidence_advantage=advantage,
        minimum_memory_margin=margin,
    )
    metrics = classification_metrics(data.hard_label, probability)
    events = event_metrics(data, probability)
    baseline = event_metrics(data, parametric)
    nonnegative = sum(
        events["events"][name]["present_class_macro_f1"]
        >= baseline["events"][name]["present_class_macro_f1"]
        for name in events["events"]
    )
    return {
        "minimum_confidence_advantage": advantage,
        "minimum_memory_margin": margin,
        "interventions": int(intervene.sum()),
        "intervention_fraction": float(intervene.float().mean()),
        "metrics": metrics,
        "event_metrics": events,
        "event_nonnegative_vs_parametric": (
            f"{nonnegative}/{len(events['events'])}"
        ),
        "mean_memory_margin_intervened": (
            float(diagnostics["memory_margin"][intervene].mean())
            if intervene.any()
            else 0.0
        ),
    }


def predictions(model, train, query, memory_indices, args, device):
    model.eval()
    with torch.no_grad():
        _, train_embedding = model(train.x.to(device))
        query_logits, query_embedding = model(query.x.to(device))
    parametric = query_logits.softmax(1).cpu()
    scores, availability = class_similarity_scores(
        query_embedding.cpu(),
        train_embedding.cpu()[memory_indices],
        train.hard_label[memory_indices],
        query.event,
        train.event[memory_indices],
        topk=args.topk,
        exclude_same_event=True,
        purity=train.purity[memory_indices],
        minimum_purity=args.minimum_memory_purity,
        chunk_size=args.chunk_size,
    )
    memory = probability_from_scores(scores, args.retrieval_temperature)
    return parametric, memory, availability


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("post-M1 selective Memory rescue audit")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--m1-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--retrieval-temperature", type=float, default=0.10)
    parser.add_argument("--minimum-memory-purity", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    report = json.loads(Path(args.m1_json).read_text(encoding="utf-8"))
    if report.get("decision") != "STOP_CROSS_EVENT_METRIC_M1":
        raise RuntimeError(
            "selective rescue is only the registered post-M1-failure audit"
        )
    checkpoint = torch.load(
        report["selected_checkpoint"], map_location="cpu", weights_only=False
    )
    if checkpoint.get("stage") != "M1":
        raise RuntimeError("unexpected selected checkpoint")

    train_raw = torch.load(
        args.train_cache, map_location="cpu", weights_only=False
    )
    calibration_raw = torch.load(
        args.calibration_cache, map_location="cpu", weights_only=False
    )
    for cache, split in (
        (train_raw, "meta_train"),
        (calibration_raw, "meta_calibration"),
    ):
        if cache.get("split") != split:
            raise RuntimeError(f"unexpected cache split: {cache.get('split')}")
        if any(
            cache.get(key, False)
            for key in (
                "meta_confirm_accessed",
                "val450_accessed",
                "test_accessed",
            )
        ):
            raise RuntimeError("forbidden split was accessed")

    mean, std = checkpoint["feature_mean"], checkpoint["feature_std"]
    train, _, _ = prepare_cache(train_raw, mean=mean, std=std)
    calibration, _, _ = prepare_cache(calibration_raw, mean=mean, std=std)
    model = CrossEventMetricEncoder(
        checkpoint["input_dim"], checkpoint["hidden"]
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    memory_indices = checkpoint["memory_indices"].long()

    train_parametric, train_memory, train_availability = predictions(
        model, train, train, memory_indices, args, device
    )
    cal_parametric, cal_memory, cal_availability = predictions(
        model, train, calibration, memory_indices, args, device
    )
    train_baseline = classification_metrics(
        train.hard_label, train_parametric
    )
    calibration_baseline = classification_metrics(
        calibration.hard_label, cal_parametric
    )

    candidates = []
    for advantage in ADVANTAGE_GRID:
        for margin in MARGIN_GRID:
            row = evaluate_policy(
                train,
                train_parametric,
                train_memory,
                train_availability,
                args.topk,
                advantage,
                margin,
            )
            gain = row["metrics"]["macro_f1"] - train_baseline["macro_f1"]
            rare_safe = all(
                row["metrics"]["class_recall"][index]
                >= train_baseline["class_recall"][index]
                for index in (1, 2)
            )
            nonnegative = int(
                row["event_nonnegative_vs_parametric"].split("/")[0]
            )
            fraction = row["intervention_fraction"]
            row["gain_vs_parametric_macro_f1"] = gain
            row["train_constraints"] = {
                "gain_ge_0p005": gain >= 0.005,
                "damaged_destroyed_recall_safe": rare_safe,
                "event_nonnegative_ge_10": nonnegative
                >= min(10, len(row["event_metrics"]["events"])),
                "intervention_fraction_ge_0p005": fraction >= 0.005,
                "intervention_fraction_le_0p20": fraction <= 0.20,
            }
            candidates.append(row)

    feasible = [
        row for row in candidates if all(row["train_constraints"].values())
    ]
    selected = max(
        feasible,
        key=lambda row: (
            row["event_metrics"]["event_macro_present_f1"],
            row["metrics"]["macro_f1"],
            -row["intervention_fraction"],
        ),
        default=None,
    )
    if selected is None:
        decision = "STOP_MEMORY_NO_IDENTIFIABLE_SAFE_SUBSET"
        calibration_result = None
        checks = {"train_has_feasible_policy": False}
    else:
        calibration_result = evaluate_policy(
            calibration,
            cal_parametric,
            cal_memory,
            cal_availability,
            args.topk,
            selected["minimum_confidence_advantage"],
            selected["minimum_memory_margin"],
        )
        gain = (
            calibration_result["metrics"]["macro_f1"]
            - calibration_baseline["macro_f1"]
        )
        rare_safe = all(
            calibration_result["metrics"]["class_recall"][index]
            >= calibration_baseline["class_recall"][index]
            for index in (1, 2)
        )
        nonnegative = int(
            calibration_result["event_nonnegative_vs_parametric"].split("/")[0]
        )
        checks = {
            "train_has_feasible_policy": True,
            "calibration_macro_f1_gain_ge_0p005": gain >= 0.005,
            "calibration_damaged_destroyed_recall_safe": rare_safe,
            "calibration_event_nonnegative_ge_10": nonnegative
            >= min(10, len(calibration_result["event_metrics"]["events"])),
            "calibration_interventions_nonzero": (
                calibration_result["interventions"] > 0
            ),
        }
        calibration_result["gain_vs_parametric_macro_f1"] = gain
        decision = (
            "ADMIT_SELECTIVE_MEMORY_AS_AUXILIARY_E1"
            if all(checks.values())
            else "STOP_MEMORY_AFTER_SELECTIVE_RESCUE"
        )

    output = {
        "protocol": PROTOCOL,
        "stage": "M1R_selective_rescue",
        "source_m1_decision": report["decision"],
        "source_variant": report["selected_variant"],
        "train_parametric": train_baseline,
        "calibration_parametric": calibration_baseline,
        "candidate_count": len(candidates),
        "feasible_candidate_count": len(feasible),
        "selected_train_policy": selected,
        "calibration_result": calibration_result,
        "checks": checks,
        "decision": decision,
        "meta_confirm_accessed": False,
        "val450_accessed": False,
        "official_instances_accessed": False,
        "test_accessed": False,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "source_variant": report["selected_variant"],
                "feasible_candidate_count": len(feasible),
                "checks": checks,
                "decision": decision,
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"Saved: {path}", flush=True)
    print(
        "Meta-confirm/Val450/official instances/test accessed: "
        "NEVER / NEVER / NEVER / NEVER"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
