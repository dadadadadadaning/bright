from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from bda_benchmark.experiments.spi_complete_building_memory_v1.core import macro_f1
from . import PROTOCOL


FAMILY_NAMES = ("p3_correct", "affected_miss", "affected_false_alarm", "severity_confusion", "other_error")
SOURCE_NAMES = ("background", "intact", "damaged", "destroyed")


def class_metrics(target: torch.Tensor, prediction: torch.Tensor, cls: int):
    tp = ((target == cls) & (prediction == cls)).sum().float()
    fp = ((target != cls) & (prediction == cls)).sum().float()
    fn = ((target == cls) & (prediction != cls)).sum().float()
    precision = tp / (tp + fp).clamp_min(1); recall = tp / (tp + fn).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {"support": int((target == cls).sum()), "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def confusion(target: torch.Tensor, prediction: torch.Tensor):
    out = torch.zeros(5, 5, dtype=torch.long)
    for actual in range(5):
        for predicted in range(5):
            out[actual, predicted] = ((target == actual) & (prediction == predicted)).sum()
    return out.tolist()


def report(target: torch.Tensor, probability: torch.Tensor, source: torch.Tensor, events: list[str]):
    prediction = probability.argmax(1)
    family = {FAMILY_NAMES[c]: class_metrics(target, prediction, c) for c in range(5)}
    by_source, by_event = {}, {}
    for cls in range(4):
        mask = source == cls
        by_source[SOURCE_NAMES[cls]] = {"queries": int(mask.sum()), "macro_f1": macro_f1(target[mask], prediction[mask], classes=5),
                                         "family": {FAMILY_NAMES[c]: class_metrics(target[mask], prediction[mask], c) for c in range(5)}}
    for event in sorted(set(events)):
        mask = torch.tensor([x == event for x in events])
        by_event[event] = macro_f1(target[mask], prediction[mask], classes=5)
    return {"macro_f1": macro_f1(target, prediction, classes=5), "family": family,
            "by_p3_source": by_source, "events": by_event, "confusion": confusion(target, prediction)}


def main(argv=None):
    p = argparse.ArgumentParser("Read-only D2 attribution for frozen CRFM predictions")
    p.add_argument("--prediction-cache", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(argv); cache = torch.load(a.prediction_cache, map_location="cpu", weights_only=False)
    if cache.get("meta_confirm_accessed") or cache.get("val450_accessed") or cache.get("test_accessed"):
        raise RuntimeError("prediction cache records forbidden split access")
    target, source, events = cache["target"].long(), cache["p3_source"].long(), cache["events"]
    reports = {name: report(target, probability, source, events) for name, probability in cache["probabilities"].items()}
    true, base = reports["a1_true_memory"], reports["a0_p3_only"]
    family_delta = {name: true["family"][name]["f1"] - base["family"][name]["f1"] for name in FAMILY_NAMES}
    source_delta = {name: true["by_p3_source"][name]["macro_f1"] - base["by_p3_source"][name]["macro_f1"] for name in SOURCE_NAMES}
    event_delta = {name: true["events"][name] - base["events"][name] for name in true["events"]}
    payload = {"protocol": PROTOCOL + "_D2", "prediction_cache": str(a.prediction_cache), "reports": reports,
               "attribution": {"family_f1_delta_true_minus_p3": family_delta,
                               "source_macro_f1_delta_true_minus_p3": source_delta,
                               "event_macro_f1_delta_true_minus_p3": event_delta,
                               "best_family_gain": max(family_delta, key=family_delta.get),
                               "affected_miss_f1_delta": family_delta["affected_miss"],
                               "p3_correct_f1_delta": family_delta["p3_correct"]},
               "checks": {"affected_miss_f1_improved": family_delta["affected_miss"] > 0,
                          "affected_miss_precision_and_recall_improved": true["family"]["affected_miss"]["precision"] >= base["family"]["affected_miss"]["precision"] and true["family"]["affected_miss"]["recall"] >= base["family"]["affected_miss"]["recall"],
                          "gain_not_only_p3_correct": family_delta["affected_miss"] >= family_delta["p3_correct"],
                          "intact_source_macro_f1_improved": source_delta["intact"] > 0},
               "meta_confirm_accessed": False, "val450_accessed": False, "test_accessed": False}
    output = Path(a.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["attribution"], indent=2, ensure_ascii=False)); print("Checks:", payload["checks"])
    return 0


if __name__ == "__main__": raise SystemExit(main())
