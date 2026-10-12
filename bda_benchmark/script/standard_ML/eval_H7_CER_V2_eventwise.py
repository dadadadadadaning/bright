# -*- coding: utf-8 -*-
"""Validation-only metric, route, event and disaster audit for H7-CER-V2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, MutableMapping

import torch
from torch.utils.data import DataLoader

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (  # noqa: E402
    HeterogeneousRGBSARDataset,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H7_CER_V2 import (  # noqa: E402
    build_h7_cer_v2_damageformer,
)
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (  # noqa: E402
    checkpoint_state,
    event_name,
    fast_histogram,
    metrics_from_histogram,
)
from bda_benchmark.script.standard_ML.summarize_H7_event_complementarity import (  # noqa: E402
    disaster_type,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H4 import (  # noqa: E402
    set_seed,
)


PAIR_NAMES = {8: "damaged_to_background", 12: "destroyed_to_background"}
VALID_PROTOCOLS = ("H7_CER_V2_best_weights_v1", "H7_CER_V2_resume_v1")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def empty_route() -> Dict[str, int]:
    return {
        "candidate": 0,
        "open": 0,
        "a2_correct": 0,
        "p3_harmed": 0,
        "reject": 0,
    }


def finalize_route(value: Mapping[str, int]) -> Dict[str, float | int]:
    opened = int(value["open"])
    candidates = int(value["candidate"])
    a2_correct = int(value["a2_correct"])
    p3_harmed = int(value["p3_harmed"])
    return {
        **{key: int(count) for key, count in value.items()},
        "open_rate": opened / max(candidates, 1),
        "route_precision": a2_correct / max(opened, 1),
        "net_correction": a2_correct - p3_harmed,
        "reject_rate": int(value["reject"]) / max(candidates, 1),
    }


def add_route(
    total: Dict[str, int],
    candidate: torch.Tensor,
    opened: torch.Tensor,
    rejected: torch.Tensor,
    p3: torch.Tensor,
    a2: torch.Tensor,
    target: torch.Tensor,
) -> None:
    total["candidate"] += int(candidate.sum())
    total["open"] += int(opened.sum())
    total["reject"] += int((candidate & rejected).sum())
    total["a2_correct"] += int((opened & (a2 == target) & (p3 != target)).sum())
    total["p3_harmed"] += int((opened & (p3 == target) & (a2 != target)).sum())


def oracle_references(path: str) -> Dict[str, Mapping[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("test_accessed") is not False:
        raise RuntimeError("H7-CER-V2 development must remain validation-only")
    return {str(row["event"]): row for row in payload.get("events", [])}


def reference_metric(row: Mapping[str, object] | None, key: str) -> float | None:
    if not row or not isinstance(row.get("p3"), Mapping):
        return None
    return float(row["p3"][key])


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("Evaluate H7-CER-V2 on validation only")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--oracle_json", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_event_tsv", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop_size", type=int, default=640)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    set_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") not in VALID_PROTOCOLS:
        raise RuntimeError(f"unexpected H7-CER-V2 protocol: {checkpoint.get('protocol')}")
    if checkpoint.get("variant") != "cer_v2a":
        raise RuntimeError("checkpoint is not H7-CER-V2-A")
    model = build_h7_cer_v2_damageformer(pretrained_rgb=False)
    incompatible = model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict load failed: {incompatible}")
    device = torch.device(args.device)
    model.to(device).eval()

    dataset = HeterogeneousRGBSARDataset(
        args.dataset_path,
        args.val_data_list_path,
        args.crop_size,
        split="val",
        variant="h4",
        sar_stats_path=args.sar_stats_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    overall_hist = torch.zeros(4, 4, dtype=torch.float64)
    event_hist: MutableMapping[str, torch.Tensor] = defaultdict(
        lambda: torch.zeros(4, 4, dtype=torch.float64)
    )
    overall_route = empty_route()
    pair_routes: MutableMapping[int, Dict[str, int]] = defaultdict(empty_route)
    event_routes: MutableMapping[str, Dict[str, int]] = defaultdict(empty_route)
    processed = 0
    for rgb, sar, _location, damage, names in loader:
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        outputs = model(rgb, sar, return_dict=True)
        prediction = outputs["output_dam"].argmax(1).cpu()
        target = damage.long().cpu()
        candidate = outputs["cer_v2_candidate_mask"][:, 0].bool().cpu()
        opened = outputs["cer_v2_route_open_mask"][:, 0].bool().cpu()
        rejected = outputs["cer_v2_selector_probability"].argmax(1).cpu() == 2
        p3 = outputs["cer_v2_p3_prediction"][:, 0].cpu()
        a2 = outputs["cer_v2_a2_prediction"][:, 0].cpu()
        pair_id = outputs["cer_v2_pair_id"][:, 0].long().cpu()
        valid = target != args.ignore_index
        candidate &= valid
        opened &= valid
        add_route(overall_route, candidate, opened, rejected, p3, a2, target)
        overall_hist += fast_histogram(
            prediction,
            target,
            classes=4,
            ignore_index=args.ignore_index,
        )
        for index, raw_name in enumerate(names):
            event = event_name(str(raw_name))
            event_hist[event] += fast_histogram(
                prediction[index],
                target[index],
                classes=4,
                ignore_index=args.ignore_index,
            )
            add_route(
                event_routes[event], candidate[index], opened[index], rejected[index],
                p3[index], a2[index], target[index]
            )
            for pair in PAIR_NAMES:
                pair_candidate = candidate[index] & (pair_id[index] == pair)
                pair_open = opened[index] & (pair_id[index] == pair)
                add_route(
                    pair_routes[pair], pair_candidate, pair_open, rejected[index],
                    p3[index], a2[index], target[index]
                )
        processed += int(rgb.shape[0])
        if processed % 50 == 0:
            print(f"H7-CER-V2 audit: {processed}/{len(dataset)}", flush=True)

    references = oracle_references(args.oracle_json)
    event_rows = []
    for event in sorted(event_hist):
        metrics = metrics_from_histogram(event_hist[event])
        p3_valid = reference_metric(references.get(event), "valid_miou")
        event_rows.append(
            {
                "event": event,
                "disaster_type": disaster_type(event),
                "metrics": metrics,
                "valid_miou": metrics["valid_miou"],
                "p3_valid_miou": p3_valid,
                "delta_p3_pp": 100.0 * (metrics["valid_miou"] - p3_valid)
                if p3_valid is not None else None,
                "route": finalize_route(event_routes[event]),
            }
        )
    grouped: MutableMapping[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in event_rows:
        grouped[str(row["disaster_type"])].append(row)
    disaster_rows = []
    for kind, rows in sorted(grouped.items()):
        deltas = [float(row["delta_p3_pp"]) for row in rows if row["delta_p3_pp"] is not None]
        disaster_rows.append(
            {
                "disaster_type": kind,
                "event_count": len(rows),
                "delta_p3_pp_macro": sum(deltas) / len(deltas) if deltas else None,
                "non_degraded_events": sum(float(row["delta_p3_pp"] or 0) >= 0 for row in rows),
                "route_net_correction": sum(int(row["route"]["net_correction"]) for row in rows),
            }
        )

    payload = {
        "protocol": "H7_CER_V2_eventwise_validation_v1",
        "split": "val",
        "test_accessed": False,
        "processed_images": processed,
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": file_sha256(args.checkpoint),
            "protocol": checkpoint.get("protocol"),
            "variant": checkpoint.get("variant"),
            "step": int(checkpoint.get("step", 0)),
            "strict_load": True,
        },
        "overall": metrics_from_histogram(overall_hist),
        "overall_route": finalize_route(overall_route),
        "pair_routes": {
            PAIR_NAMES[pair]: finalize_route(value)
            for pair, value in sorted(pair_routes.items())
        },
        "events": event_rows,
        "disaster_types": disaster_rows,
        "thresholds_fixed_before_training": {
            "advantage": 0.25,
            "reject": 0.30,
            "local_consistency": 0.60,
            "route_budget": 0.10,
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tsv = Path(args.output_event_tsv)
    with tsv.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "event", "disaster_type", "valid_miou", "p3_valid_miou",
            "delta_p3_pp", "route_open", "route_precision", "route_net_correction",
        )
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        for row in event_rows:
            writer.writerow(
                {
                    "event": row["event"],
                    "disaster_type": row["disaster_type"],
                    "valid_miou": row["valid_miou"],
                    "p3_valid_miou": row["p3_valid_miou"],
                    "delta_p3_pp": row["delta_p3_pp"],
                    "route_open": row["route"]["open"],
                    "route_precision": row["route"]["route_precision"],
                    "route_net_correction": row["route"]["net_correction"],
                }
            )

    metrics = payload["overall"]
    route = payload["overall_route"]
    print("=" * 78)
    print(
        f"H7-CER-V2-A | Valid={100*metrics['valid_miou']:.4f} | "
        f"mIoU4={100*metrics['miou4']:.4f} | Dmg={100*metrics['damaged_iou']:.4f} | "
        f"Des={100*metrics['destroyed_iou']:.4f}"
    )
    print(
        f"Route candidate/open={route['candidate']}/{route['open']} | "
        f"precision={100*route['route_precision']:.2f}% | "
        f"net correction={route['net_correction']} | reject={100*route['reject_rate']:.2f}%"
    )
    print(f"Strict load: True\nTest accessed: False\nSaved: {output}\nSaved: {tsv}")


if __name__ == "__main__":
    main()
