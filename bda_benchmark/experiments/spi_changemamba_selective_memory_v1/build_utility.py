#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import sha256_file

from . import PROTOCOL
from .core import (
    ChangeMambaTokenEncoder, groups_at, retrieval_diverse_topk,
    retrieval_relative_gap, valid_miou,
)


def top_margin(probability: torch.Tensor) -> torch.Tensor:
    values = probability.topk(2, dim=1).values
    return values[:, 0] - values[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser("Build exact mIoU utility labels for selective actions")
    parser.add_argument("--records", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--max_per_event", type=int, default=2)
    parser.add_argument("--min_p3_entropy", type=float, default=0.30)
    parser.add_argument("--max_abs_p3_occurrence_margin", type=float, default=0.60)
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
    memory_keys = memory["keys"].to(device).float()
    memory_labels = memory["labels"].to(device)
    rows = []
    baseline = valid_miou(records["base_hist"])
    with torch.inference_mode():
        total = records["labels"].numel()
        for start in range(0, total, args.batch_size):
            stop = min(start + args.batch_size, total)
            index = torch.arange(start, stop)
            key, direct_logits = model(groups_at(records, index, device))
            direct_probability = direct_logits.softmax(1)
            retrieval = retrieval_diverse_topk(
                key, memory_keys, memory_labels,
                query_images=records["images"][start:stop], memory_images=memory["images"],
                query_events=records["events"][start:stop], memory_events=memory["events"],
                exclude_same_image=True, exclude_same_event=True,
                topk=args.topk, max_per_event=args.max_per_event,
            )
            memory_probability = retrieval["posterior"]
            relative_gap = retrieval_relative_gap(retrieval, memory_labels)
            density = retrieval["scores"].mean(1)
            direct_prediction = direct_probability.argmax(1) + 1
            memory_prediction = retrieval["prediction"]
            p3_state = records["groups"]["p3_state"][index].float()
            geometry = records["groups"]["geometry"][index].float()
            for local, global_index in enumerate(range(start, stop)):
                p3_label = int(records["p3_region_labels"][global_index])
                p3_affected = p3_label >= 2
                direct = direct_probability[local].cpu()
                mem = memory_probability[local].cpu()
                # Hard safety gate: when P3 is simultaneously low-entropy and
                # far from the Intact/Affected boundary, neither auxiliary
                # expert is allowed to propose a writeback.
                p3_entropy = float(p3_state[local, 8])
                p3_occurrence_margin = float(p3_state[local, 11])
                p3_uncertain = (
                    p3_entropy >= float(args.min_p3_entropy)
                    or abs(p3_occurrence_margin)
                    <= float(args.max_abs_p3_occurrence_margin)
                )
                if not p3_uncertain:
                    continue
                common = {
                    "p3_p0": float(p3_state[local, 0]), "p3_p1": float(p3_state[local, 1]),
                    "p3_p2": float(p3_state[local, 2]), "p3_p3": float(p3_state[local, 3]),
                    "p3_entropy": p3_entropy,
                    "p3_occurrence_margin": p3_occurrence_margin,
                    "p3_fragmentation": float(p3_state[local, 18]),
                    "p3_uncertainty": float(records["p3_uncertainties"][global_index]),
                    "region_area": float(geometry[local, 0]),
                    "building_probability": float(geometry[local, 5]),
                    "direct_p0": float(direct[0]), "direct_p1": float(direct[1]),
                    "direct_p2": float(direct[2]),
                    "direct_margin": float(top_margin(direct[None])[0]),
                    "direct_affected": float(direct[1:].sum()),
                    "memory_p0": float(mem[0]), "memory_p1": float(mem[1]),
                    "memory_p2": float(mem[2]),
                    "memory_margin": float(retrieval["margin"][local]),
                    "memory_agreement": float(retrieval["agreement"][local]),
                    "memory_relative_gap": float(relative_gap[local]),
                    "memory_density": float(density[local]),
                    "direct_memory_l1": float((direct - mem).abs().sum()),
                    "direct_memory_affected_delta": float(direct[1:].sum() - mem[1:].sum()),
                    "same_prediction": float(int(direct_prediction[local]) == int(memory_prediction[local])),
                }
                for source, prediction in (
                    ("direct", int(direct_prediction[local])),
                    ("memory", int(memory_prediction[local])),
                ):
                    proposal_affected = prediction >= 2
                    if proposal_affected == p3_affected:
                        continue
                    delta = (
                        records["delta_occurrence_affected"][global_index]
                        if proposal_affected else records["delta_occurrence_intact"][global_index]
                    ).double()
                    utility = 100.0 * (valid_miou(records["base_hist"] + delta) - baseline)
                    features = dict(common)
                    features.update({
                        "source_direct": float(source == "direct"),
                        "source_memory": float(source == "memory"),
                        "proposal_affected": float(proposal_affected),
                    })
                    rows.append({
                        "region_key": f"{records['images'][global_index]}:{int(records['region_ids'][global_index])}",
                        "record_index": global_index,
                        "image": records["images"][global_index],
                        "event": records["events"][global_index],
                        "source": source, "target_affected": proposal_affected,
                        "features": features, "delta": delta,
                        "utility_miou_pp": utility, "accept_label": utility > 0.0,
                    })
            if stop % 5000 < args.batch_size or stop == total:
                print(f"utility {stop}/{total} proposals={len(rows)}", flush=True)
    payload = {
        "protocol": PROTOCOL + "_UTILITY", "split": records["split"],
        "allowed_for_verifier_training": records["split"] == "train",
        "rows": rows, "base_hist": records["base_hist"], "loc_hist": records["loc_hist"],
        "event_base_hist": records["event_base_hist"],
        "metadata": {
            "records": str(Path(args.records).resolve()),
            "records_sha256": sha256_file(args.records),
            "metric": str(Path(args.metric).resolve()),
            "metric_sha256": sha256_file(args.metric),
            "memory": str(Path(args.memory).resolve()),
            "memory_sha256": sha256_file(args.memory),
            "proposal_policy": "occurrence-only; preserve P3 Damaged-vs-Destroyed ordering",
            "p3_hard_safety_gate": {
                "min_entropy": args.min_p3_entropy,
                "max_abs_occurrence_margin": args.max_abs_p3_occurrence_margin,
                "rule": "eligible when entropy>=min OR abs(margin)<=max",
            },
            "train_direct_predictions": "in-sample screening evidence; final unseen-event claim requires OOF Key training",
            "retrieval_excludes_same_image": True,
            "retrieval_excludes_same_event": True,
            "test_accessed": False,
        },
        "test_accessed": False,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(f"Saved: {destination} proposals={len(rows)} split={records['split']}")


if __name__ == "__main__":
    main()
