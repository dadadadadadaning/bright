from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import PROTOCOL
from .core import (
    CrossEventMetricEncoder, class_similarity_scores, classification_metrics,
    event_metrics, prepare_cache, probability_from_scores,
)


def hierarchical_probabilities(parametric: torch.Tensor, retrieved: torch.Tensor):
    eps = 1e-8
    param_affected = parametric[:, 1:].sum(1)
    retrieved_affected = retrieved[:, 1:].sum(1)
    param_severity = parametric[:, 1:] / param_affected[:, None].clamp_min(eps)
    retrieved_severity = retrieved[:, 1:] / retrieved_affected[:, None].clamp_min(eps)
    occurrence = torch.cat(((1 - retrieved_affected)[:, None],
                            retrieved_affected[:, None] * param_severity), 1)
    severity = torch.cat((parametric[:, :1], param_affected[:, None] * retrieved_severity), 1)
    return occurrence, severity, retrieved


def reliability_gate(retrieved: torch.Tensor, availability: torch.Tensor,
                     topk: int) -> torch.Tensor:
    top = retrieved.topk(2, 1).values
    margin = top[:, 0] - top[:, 1]
    entropy = -(retrieved.clamp_min(1e-8) * retrieved.clamp_min(1e-8).log()).sum(1)
    entropy = entropy / torch.log(torch.tensor(3.0))
    coverage = (availability >= topk).all(1).float()
    # Pre-registered monotonic reliability: no calibration threshold is fitted.
    return (margin / .25).clamp(0, 1) * (1 - entropy).clamp(0, 1) * coverage


def blend(base: torch.Tensor, candidate: torch.Tensor, gate: torch.Tensor):
    value = base * (1 - gate[:, None]) + candidate * gate[:, None]
    return value / value.sum(1, keepdim=True).clamp_min(1e-8)


def evaluate(data, probability, base_event):
    overall = classification_metrics(data.hard_label, probability)
    events = event_metrics(data, probability)
    nonnegative = sum(
        events["events"][name]["present_class_macro_f1"] >=
        base_event["events"][name]["present_class_macro_f1"]
        for name in events["events"]
    )
    return {"overall": overall, "event": events,
            "event_nonnegative": f"{nonnegative}/{len(events['events'])}"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("M2 occurrence/severity/confidence memory ablation")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--m1-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--retrieval-temperature", type=float, default=.10)
    parser.add_argument("--minimum-memory-purity", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    m1 = json.loads(Path(args.m1_json).read_text(encoding="utf-8"))
    if m1.get("decision") != "ADMIT_HIERARCHICAL_MEMORY_M2":
        raise RuntimeError("M2 refused: M1 cross-event metric space did not pass")
    checkpoint = torch.load(m1["selected_checkpoint"], map_location="cpu", weights_only=False)
    train_cache = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration_cache = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    for cache, split in ((train_cache, "meta_train"), (calibration_cache, "meta_calibration")):
        if cache.get("split") != split or cache.get("meta_confirm_accessed", False) or cache.get("val450_accessed", False) or cache.get("test_accessed", False):
            raise RuntimeError("M2 received unexpected or forbidden cache")
    train, _, _ = prepare_cache(train_cache, mean=checkpoint["feature_mean"], std=checkpoint["feature_std"])
    calibration, _, _ = prepare_cache(calibration_cache, mean=checkpoint["feature_mean"], std=checkpoint["feature_std"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = CrossEventMetricEncoder(checkpoint["input_dim"], checkpoint["hidden"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True); model.eval()
    with torch.no_grad():
        train_logits, train_embedding = model(train.x.to(device))
        cal_logits, cal_embedding = model(calibration.x.to(device))
    parametric = cal_logits.softmax(1).cpu()
    selected = checkpoint["memory_indices"].long()
    scores, availability = class_similarity_scores(
        cal_embedding.cpu(), train_embedding.cpu()[selected], train.hard_label[selected],
        calibration.event, train.event[selected], topk=args.topk, exclude_same_event=True,
        purity=train.purity[selected], minimum_purity=args.minimum_memory_purity,
        chunk_size=args.chunk_size,
    )
    retrieved = probability_from_scores(scores, args.retrieval_temperature)
    occurrence, severity, dual = hierarchical_probabilities(parametric, retrieved)
    gate = reliability_gate(retrieved, availability, args.topk)
    base_event = event_metrics(calibration, parametric)
    variants = {
        "b0_parametric": evaluate(calibration, parametric, base_event),
        "b1_occurrence_only": evaluate(calibration, occurrence, base_event),
        "b2_severity_only": evaluate(calibration, severity, base_event),
        "b3_dual": evaluate(calibration, dual, base_event),
        "b4_confident_occurrence": evaluate(calibration, blend(parametric, occurrence, gate), base_event),
        "b5_confident_severity": evaluate(calibration, blend(parametric, severity, gate), base_event),
        "b6_confident_dual": evaluate(calibration, blend(parametric, dual, gate), base_event),
    }
    candidate_names = ("b4_confident_occurrence", "b5_confident_severity", "b6_confident_dual")
    selected_name = max(candidate_names, key=lambda name: variants[name]["overall"]["macro_f1"])
    base = variants["b0_parametric"]; best = variants[selected_name]
    gain = best["overall"]["macro_f1"] - base["overall"]["macro_f1"]
    nonnegative = int(best["event_nonnegative"].split("/")[0])
    checks = {
        "macro_f1_gain_ge_0p01": gain >= .01,
        "damaged_recall_not_below_parametric": best["overall"]["class_recall"][1] >= base["overall"]["class_recall"][1],
        "destroyed_recall_not_below_parametric": best["overall"]["class_recall"][2] >= base["overall"]["class_recall"][2],
        "event_nonnegative_ge_10": nonnegative >= min(10, len(base_event["events"])),
        "gate_is_nontrivial": float((gate > 0).float().mean()) > .01,
        "gate_is_selective": float((gate < .999).float().mean()) > .01,
    }
    decision = "ADMIT_MEMORY_TO_P5_INTEGRATION_E1" if all(checks.values()) else "STOP_HIERARCHICAL_MEMORY_M2"
    report = {
        "protocol": PROTOCOL, "stage": "M2", "configuration": vars(args),
        "source_variant": m1["selected_variant"], "variants": variants,
        "selected_confident_variant": selected_name, "selected_macro_f1_gain": gain,
        "gate": {"mean": float(gate.mean()), "nonzero_fraction": float((gate > 0).float().mean()),
                 "q50": float(torch.quantile(gate, .5)), "q90": float(torch.quantile(gate, .9))},
        "checks": checks, "decision": decision,
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected_name, "gain": gain, "gate": report["gate"],
                      "checks": checks, "decision": decision}, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
