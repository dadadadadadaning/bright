# -*- coding: utf-8 -*-
"""Validation-only event/disaster audit for a trained H7-CER router."""
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
from bda_benchmark.model.DamageFormer_RGBSAR_H7_CER import (  # noqa: E402
    build_h7_cer_damageformer,
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


VALID_PROTOCOLS = ("H7_CER_best_weights_v1", "H7_CER_resume_v1")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _oracle_events(path: str) -> Dict[str, Mapping[str, object]]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol") != "H7_P3_A2_validation_oracle_v1":
        raise RuntimeError("oracle_json has an unexpected protocol")
    if payload.get("test_accessed") is not False:
        raise RuntimeError("H7 development comparison must remain validation-only")
    return {str(row["event"]): row for row in payload.get("events", [])}


def _reference_metric(
    row: Mapping[str, object] | None, expert: str, key: str
) -> float | None:
    if row is None:
        return None
    values = row.get(expert)
    if not isinstance(values, Mapping) or key not in values:
        return None
    return float(values[key])


def _mean(values) -> float | None:
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def write_event_tsv(path: Path, rows) -> None:
    fields = (
        "event",
        "disaster_type",
        "candidate_valid_miou",
        "p3_valid_miou",
        "a2_valid_miou",
        "candidate_minus_p3_pp",
        "candidate_minus_a2_pp",
        "candidate_damaged_iou",
        "candidate_destroyed_iou",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser("Evaluate H7-CER by event and disaster type")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--oracle_json", default="")
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
        raise RuntimeError(f"unexpected H7-CER protocol: {checkpoint.get('protocol')!r}")
    if str(checkpoint.get("variant", "")).lower() != "cer_damaged":
        raise RuntimeError(f"unexpected H7-CER variant: {checkpoint.get('variant')!r}")
    model = build_h7_cer_damageformer(pretrained_rgb=False)
    incompatible = model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict H7-CER load failed: {incompatible}")
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
    overall = torch.zeros(4, 4, dtype=torch.float64)
    event_histograms: MutableMapping[str, torch.Tensor] = defaultdict(
        lambda: torch.zeros(4, 4, dtype=torch.float64)
    )
    processed = 0
    for rgb, sar, _location, damage, names in loader:
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        damage = damage.to(device, non_blocking=True).long()
        prediction = model(rgb, sar, return_dict=True)["output_dam"].argmax(1).cpu()
        damage_cpu = damage.cpu()
        overall += fast_histogram(prediction, damage_cpu, ignore_index=args.ignore_index)
        for index, name in enumerate(names):
            event_histograms[event_name(str(name))] += fast_histogram(
                prediction[index], damage_cpu[index], ignore_index=args.ignore_index
            )
        processed += int(rgb.shape[0])
        if processed % 50 == 0:
            print(f"H7-CER event audit: {processed}/{len(dataset)}", flush=True)

    references = _oracle_events(args.oracle_json)
    rows = []
    for event, histogram in sorted(event_histograms.items()):
        candidate = metrics_from_histogram(histogram)
        reference = references.get(event)
        p3_valid = _reference_metric(reference, "p3", "valid_miou")
        a2_valid = _reference_metric(reference, "a2", "valid_miou")
        rows.append(
            {
                "event": event,
                "disaster_type": disaster_type(event),
                "candidate": candidate,
                "confusion": histogram.tolist(),
                "candidate_valid_miou": candidate["valid_miou"],
                "p3_valid_miou": p3_valid,
                "a2_valid_miou": a2_valid,
                "candidate_minus_p3_pp": (
                    100.0 * (candidate["valid_miou"] - p3_valid)
                    if p3_valid is not None
                    else None
                ),
                "candidate_minus_a2_pp": (
                    100.0 * (candidate["valid_miou"] - a2_valid)
                    if a2_valid is not None
                    else None
                ),
                "candidate_damaged_iou": candidate["damaged_iou"],
                "candidate_destroyed_iou": candidate["destroyed_iou"],
            }
        )

    grouped: MutableMapping[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["disaster_type"])].append(row)
    disaster_rows = []
    for name, values in sorted(grouped.items()):
        disaster_rows.append(
            {
                "disaster_type": name,
                "event_count": len(values),
                "events": [str(value["event"]) for value in values],
                "candidate_valid_macro": _mean(
                    value["candidate_valid_miou"] for value in values
                ),
                "p3_valid_macro": _mean(value["p3_valid_miou"] for value in values),
                "a2_valid_macro": _mean(value["a2_valid_miou"] for value in values),
                "candidate_minus_p3_pp_macro": _mean(
                    value["candidate_minus_p3_pp"] for value in values
                ),
                "candidate_minus_a2_pp_macro": _mean(
                    value["candidate_minus_a2_pp"] for value in values
                ),
            }
        )

    payload = {
        "protocol": "H7_CER_eventwise_validation_v1",
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
        "overall": metrics_from_histogram(overall),
        "overall_confusion": overall.tolist(),
        "events": rows,
        "disaster_types": disaster_rows,
        "reference_oracle_json": str(Path(args.oracle_json).resolve())
        if args.oracle_json
        else None,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    event_tsv = Path(args.output_event_tsv)
    event_tsv.parent.mkdir(parents=True, exist_ok=True)
    write_event_tsv(event_tsv, rows)

    print(
        f"{'Event':<28} {'Type':<12} {'P3':>8} {'A2':>8} "
        f"{'CER':>8} {'CER-P3':>9} {'Dmg':>8} {'Des':>8}"
    )
    print("-" * 104)
    for row in sorted(rows, key=lambda value: float(value.get("candidate_minus_p3_pp") or -999), reverse=True):
        print(
            f"{row['event']:<28} {row['disaster_type']:<12} "
            f"{100*float(row['p3_valid_miou'] or 0):8.3f} "
            f"{100*float(row['a2_valid_miou'] or 0):8.3f} "
            f"{100*float(row['candidate_valid_miou']):8.3f} "
            f"{float(row['candidate_minus_p3_pp'] or 0):+9.3f} "
            f"{100*float(row['candidate_damaged_iou']):8.3f} "
            f"{100*float(row['candidate_destroyed_iou']):8.3f}"
        )
    print("\nDisaster-type macro")
    for row in disaster_rows:
        print(
            f"{row['disaster_type']:<12} N={row['event_count']:<2} "
            f"P3={100*float(row['p3_valid_macro'] or 0):.3f} "
            f"A2={100*float(row['a2_valid_macro'] or 0):.3f} "
            f"CER={100*float(row['candidate_valid_macro'] or 0):.3f} "
            f"CER-P3={float(row['candidate_minus_p3_pp_macro'] or 0):+.3f} pp"
        )
    print(f"Strict checkpoint load: True\nSaved: {output}\nSaved: {event_tsv}")


if __name__ == "__main__":
    main()
