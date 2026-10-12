#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import sha256_file
from bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.audit_retrieval import report

from . import PROTOCOL
from .core import (
    ChangeMambaTokenEncoder, groups_at, retrieval_diverse_topk,
    retrieval_relative_gap,
)


def partition(target, p3, direct, memory):
    p, d, m = p3 == target, direct == target, memory == target
    wrong = ~p
    return {
        "p3_region_accuracy": float(p.float().mean()),
        "direct_region_accuracy": float(d.float().mean()),
        "memory_region_accuracy": float(m.float().mean()),
        "p3_wrong_count": int(wrong.sum()),
        "direct_recovery_rate": float((wrong & d).sum() / wrong.sum().clamp_min(1)),
        "memory_recovery_rate": float((wrong & m).sum() / wrong.sum().clamp_min(1)),
        "direct_only_rescue": int((wrong & d & ~m).sum()),
        "memory_only_rescue": int((wrong & ~d & m).sum()),
        "union_recovery_rate": float((wrong & (d | m)).sum() / wrong.sum().clamp_min(1)),
        "direct_harm_rate": float((p & ~d).sum() / p.sum().clamp_min(1)),
        "memory_harm_rate": float((p & ~m).sum() / p.sum().clamp_min(1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser("Audit ChangeMamba Direct and Memory evidence")
    parser.add_argument("--records", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--max_per_event", type=int, default=2)
    parser.add_argument("--exclude_same_image", action="store_true")
    parser.add_argument("--exclude_same_event", action="store_true")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    records = torch.load(args.records, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.metric, map_location="cpu", weights_only=False)
    memory = torch.load(args.memory, map_location="cpu", weights_only=False)
    if memory["metadata"]["metric_sha256"] != sha256_file(args.metric):
        raise RuntimeError("Memory and metric checkpoint do not match")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ChangeMambaTokenEncoder(
        checkpoint["group_dims"], checkpoint["variant"],
        checkpoint["key_dim"], checkpoint["hidden_dim"],
    ).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    keys = memory["keys"].to(device).float()
    labels = memory["labels"].to(device)
    direct_rows, memory_rows = [], []
    margins, agreements, max_similarity, gaps, density = [], [], [], [], []
    with torch.inference_mode():
        total = records["labels"].numel()
        for start in range(0, total, args.batch_size):
            stop = min(start + args.batch_size, total)
            index = torch.arange(start, stop)
            query, logits = model(groups_at(records, index, device))
            result = retrieval_diverse_topk(
                query, keys, labels,
                query_images=records["images"][start:stop], memory_images=memory["images"],
                query_events=records["events"][start:stop], memory_events=memory["events"],
                exclude_same_image=args.exclude_same_image,
                exclude_same_event=args.exclude_same_event,
                topk=args.topk, max_per_event=args.max_per_event,
            )
            direct_rows.append((logits.argmax(1) + 1).cpu())
            memory_rows.append(result["prediction"].cpu())
            margins.append(result["margin"].cpu())
            agreements.append(result["agreement"].cpu())
            max_similarity.append(result["max_similarity"].cpu())
            gaps.append(retrieval_relative_gap(result, labels).cpu())
            density.append(result["scores"].mean(1).cpu())
            if stop % 5000 < args.batch_size or stop == total:
                print(f"audit {stop}/{total}", flush=True)
    target = records["labels"]
    direct = torch.cat(direct_rows)
    memory_prediction = torch.cat(memory_rows)
    payload = {
        "protocol": PROTOCOL + "_RETRIEVAL_AUDIT",
        "variant": checkpoint["variant"],
        "direct": report(target, direct, records["events"], include_event_class_metrics=True),
        "memory": report(target, memory_prediction, records["events"], include_event_class_metrics=True),
        "complementarity": partition(
            target, records["p3_region_labels"], direct, memory_prediction
        ),
        "retrieval_diagnostics": {
            "mean_margin": float(torch.cat(margins).mean()),
            "mean_agreement": float(torch.cat(agreements).mean()),
            "mean_max_similarity": float(torch.cat(max_similarity).mean()),
            "mean_relative_class_gap": float(torch.cat(gaps).mean()),
            "mean_neighbour_density": float(torch.cat(density).mean()),
        },
        "exclude_same_image": args.exclude_same_image,
        "exclude_same_event": args.exclude_same_event,
        "test_accessed": False,
    }
    payload["memory_minus_direct_macro_f1_pp"] = 100.0 * (
        payload["memory"]["macro_f1"] - payload["direct"]["macro_f1"]
    )
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

