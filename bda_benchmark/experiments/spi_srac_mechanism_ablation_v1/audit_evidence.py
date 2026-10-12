#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.audit_retrieval import report

from . import PROTOCOL


def partition(target, p3, direct, knn):
    p = p3 == target
    wrong = ~p
    output = {"p3_region_accuracy": float(p.float().mean()), "p3_wrong_count": int(wrong.sum())}
    for name, prediction in (("direct", direct), ("knn", knn)):
        correct = prediction == target
        output[f"{name}_recovery_rate"] = float((wrong & correct).sum() / wrong.sum().clamp_min(1))
        output[f"{name}_harm_rate"] = float((p & ~correct).sum() / p.sum().clamp_min(1))
    output["knn_only_rescue"] = int((wrong & (knn == target) & (direct != target)).sum())
    output["direct_only_rescue"] = int((wrong & (direct == target) & (knn != target)).sum())
    return output


def main() -> None:
    parser = argparse.ArgumentParser("Audit region pooler Direct and kNN evidence")
    parser.add_argument("--records", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    records = torch.load(args.records, map_location="cpu", weights_only=False)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache.get("protocol") != PROTOCOL + "_NEIGHBOURS":
        raise RuntimeError("unexpected neighbour cache")
    target = cache["labels"]
    direct = cache["direct_logits"].argmax(1) + 1
    knn = cache["knn_prediction"]
    payload = {
        "protocol": PROTOCOL + "_EVIDENCE_AUDIT",
        "direct": report(target, direct, cache["events"], include_event_class_metrics=True),
        "knn": report(target, knn, cache["events"], include_event_class_metrics=True),
        "complementarity": partition(target, records["p3_region_labels"], direct, knn),
        "diagnostics": {
            "mean_margin": float(cache["margin"].mean()),
            "mean_agreement": float(cache["agreement"].mean()),
            "mean_relative_gap": float(cache["relative_gap"].mean()),
            "mean_source_event_count": float(cache["source_event_count"].float().mean()),
        },
        "test_accessed": False,
    }
    payload["knn_minus_direct_macro_f1_pp"] = 100.0 * (
        payload["knn"]["macro_f1"] - payload["direct"]["macro_f1"]
    )
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
