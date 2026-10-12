#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bda_benchmark.experiments.p3_damageformer_ablation_v1.core import sha256_file
from bda_benchmark.experiments.spi_changemamba_selective_memory_v1.build_utility import top_margin
from bda_benchmark.experiments.spi_selective_memory_oracle_v1.oracle_core import valid_miou

from . import PROTOCOL
from .core import RetrievalAugmentedClassifier


@torch.inference_mode()
def rag_probabilities(cache, checkpoint, device, batch_size):
    model = RetrievalAugmentedClassifier(
        checkpoint["key_dim"], checkpoint["variant"], checkpoint["geometry_dim"],
        checkpoint["event_count"], checkpoint["hidden_dim"], checkpoint["heads"],
    ).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    chunks = []
    for start in range(0, cache["labels"].numel(), batch_size):
        stop = min(start + batch_size, cache["labels"].numel())
        logits = model(
            cache["query_keys"][start:stop].to(device).float(),
            cache["neighbour_keys"][start:stop].to(device).float(),
            cache["neighbour_labels"][start:stop].to(device),
            cache["neighbour_geometry"][start:stop].to(device).float(),
            cache["neighbour_events"][start:stop].to(device),
        )
        chunks.append(logits.softmax(1).cpu())
    return torch.cat(chunks)


def main() -> None:
    parser = argparse.ArgumentParser("Build exact pixel-mIoU utility for Direct/RAG actions")
    parser.add_argument("--records", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--rag", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_p3_entropy", type=float, default=0.30)
    parser.add_argument("--max_abs_p3_occurrence_margin", type=float, default=0.60)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    records = torch.load(args.records, map_location="cpu", weights_only=False)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.rag, map_location="cpu", weights_only=False)
    if cache["metadata"]["records_sha256"] != sha256_file(args.records):
        raise RuntimeError("neighbour cache and records do not match")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    direct_probability = cache["direct_logits"].float().softmax(1)
    rag_probability = rag_probabilities(cache, checkpoint, device, args.batch_size)
    direct_prediction = direct_probability.argmax(1) + 1
    rag_prediction = rag_probability.argmax(1) + 1
    baseline = valid_miou(records["base_hist"])
    rows = []
    for index in range(records["labels"].numel()):
        p3_state = records["p3_state"][index].float()
        geometry = records["geometry"][index].float()
        p3_label = int(records["p3_region_labels"][index])
        p3_affected = p3_label >= 2
        p3_entropy = float(p3_state[8])
        p3_occurrence_margin = float(p3_state[11])
        if not (
            p3_entropy >= args.min_p3_entropy
            or abs(p3_occurrence_margin) <= args.max_abs_p3_occurrence_margin
        ):
            continue
        direct = direct_probability[index]
        rag = rag_probability[index]
        common = {
            "p3_p0": float(p3_state[0]), "p3_p1": float(p3_state[1]),
            "p3_p2": float(p3_state[2]), "p3_p3": float(p3_state[3]),
            "p3_entropy": p3_entropy,
            "p3_occurrence_margin": p3_occurrence_margin,
            "p3_fragmentation": float(p3_state[18]),
            "p3_uncertainty": float(records["p3_uncertainties"][index]),
            "region_area": float(geometry[0]),
            "building_probability": float(geometry[5]),
            "direct_p0": float(direct[0]), "direct_p1": float(direct[1]),
            "direct_p2": float(direct[2]),
            "direct_margin": float(top_margin(direct[None])[0]),
            "direct_affected": float(direct[1:].sum()),
            # Existing verifier names are retained for compatibility; here
            # "memory" explicitly means the retrieval-conditioned classifier.
            "memory_p0": float(rag[0]), "memory_p1": float(rag[1]),
            "memory_p2": float(rag[2]),
            "memory_margin": float(top_margin(rag[None])[0]),
            "memory_agreement": float(cache["agreement"][index]),
            "memory_relative_gap": float(cache["relative_gap"][index]),
            "memory_density": float(cache["density"][index]),
            "direct_memory_l1": float((direct - rag).abs().sum()),
            "direct_memory_affected_delta": float(direct[1:].sum() - rag[1:].sum()),
            "same_prediction": float(int(direct_prediction[index]) == int(rag_prediction[index])),
            "source_event_count": float(cache["source_event_count"][index]),
            "source_event_entropy": float(cache["source_event_entropy"][index]),
        }
        for source, prediction in (("direct", int(direct_prediction[index])), ("memory", int(rag_prediction[index]))):
            proposal_affected = prediction >= 2
            if proposal_affected == p3_affected:
                continue
            delta = (
                records["delta_occurrence_affected"][index]
                if proposal_affected else records["delta_occurrence_intact"][index]
            ).double()
            utility = 100.0 * (valid_miou(records["base_hist"] + delta) - baseline)
            features = dict(common)
            features.update({
                "source_direct": float(source == "direct"),
                "source_memory": float(source == "memory"),
                "proposal_affected": float(proposal_affected),
            })
            rows.append({
                "region_key": f"{records['images'][index]}:{int(records['region_ids'][index])}",
                "record_index": index,
                "image": records["images"][index],
                "event": records["events"][index],
                "source": source,
                "target_affected": proposal_affected,
                "features": features,
                "delta": delta,
                "utility_miou_pp": utility,
                "accept_label": utility > 0.0,
            })
    payload = {
        "protocol": PROTOCOL + "_UTILITY",
        "split": records["split"],
        "allowed_for_verifier_training": records["split"] == "train",
        "rows": rows,
        "base_hist": records["base_hist"],
        "loc_hist": records["loc_hist"],
        "event_base_hist": records["event_base_hist"],
        "metadata": {
            "records": str(Path(args.records).resolve()),
            "records_sha256": sha256_file(args.records),
            "cache": str(Path(args.cache).resolve()),
            "cache_sha256": sha256_file(args.cache),
            "rag": str(Path(args.rag).resolve()),
            "rag_sha256": sha256_file(args.rag),
            "policy": "occurrence-only; preserve P3 Damaged-vs-Destroyed ordering",
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

