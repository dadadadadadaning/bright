#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.audit_retrieval import report

from . import PROTOCOL
from .core import RetrievalAugmentedClassifier


def complementarity(target, p3, direct, knn, rag):
    p = p3 == target
    wrong = ~p
    result = {
        "p3_region_accuracy": float(p.float().mean()),
        "p3_wrong_count": int(wrong.sum()),
    }
    predictions = {"direct": direct, "knn": knn, "rag": rag}
    for name, prediction in predictions.items():
        correct = prediction == target
        result[f"{name}_recovery_rate"] = float((wrong & correct).sum() / wrong.sum().clamp_min(1))
        result[f"{name}_harm_rate"] = float((p & ~correct).sum() / p.sum().clamp_min(1))
    result["rag_only_over_direct"] = int((wrong & (rag == target) & (direct != target)).sum())
    result["direct_only_over_rag"] = int((wrong & (direct == target) & (rag != target)).sum())
    result["union_recovery_rate"] = float(
        (wrong & ((direct == target) | (rag == target))).sum() / wrong.sum().clamp_min(1)
    )
    return result


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("Audit Direct, kNN and retrieval-conditioned classifier")
    parser.add_argument("--records", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--rag", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    records = torch.load(args.records, map_location="cpu", weights_only=False)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.rag, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != PROTOCOL + "_RAG":
        raise RuntimeError("unexpected RAG checkpoint")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = RetrievalAugmentedClassifier(
        checkpoint["key_dim"], checkpoint["variant"], checkpoint["geometry_dim"],
        checkpoint["event_count"], checkpoint["hidden_dim"], checkpoint["heads"],
    ).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    rows = []
    for start in range(0, cache["labels"].numel(), args.batch_size):
        stop = min(start + args.batch_size, cache["labels"].numel())
        logits = model(
            cache["query_keys"][start:stop].to(device).float(),
            cache["neighbour_keys"][start:stop].to(device).float(),
            cache["neighbour_labels"][start:stop].to(device),
            cache["neighbour_geometry"][start:stop].to(device).float(),
            cache["neighbour_events"][start:stop].to(device),
        )
        rows.append(logits.cpu())
    rag_logits = torch.cat(rows)
    target = cache["labels"]
    direct = cache["direct_logits"].argmax(1) + 1
    knn = cache["knn_prediction"]
    rag = rag_logits.argmax(1) + 1
    result = {
        "protocol": PROTOCOL + "_AUDIT",
        "source_variant": Path(args.rag).stem,
        "direct": report(target, direct, cache["events"], include_event_class_metrics=True),
        "knn": report(target, knn, cache["events"], include_event_class_metrics=True),
        "rag": report(target, rag, cache["events"], include_event_class_metrics=True),
        "complementarity": complementarity(
            target, records["p3_region_labels"], direct, knn, rag
        ),
        "diagnostics": {
            "mean_knn_margin": float(cache["margin"].mean()),
            "mean_knn_agreement": float(cache["agreement"].mean()),
            "mean_relative_gap": float(cache["relative_gap"].mean()),
            "mean_source_event_count": float(cache["source_event_count"].float().mean()),
            "mean_source_event_entropy": float(cache["source_event_entropy"].mean()),
        },
        "test_accessed": False,
    }
    result["comparisons_pp"] = {
        "rag_minus_direct_macro_f1": 100.0 * (result["rag"]["macro_f1"] - result["direct"]["macro_f1"]),
        "rag_minus_knn_macro_f1": 100.0 * (result["rag"]["macro_f1"] - result["knn"]["macro_f1"]),
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
