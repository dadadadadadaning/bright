"""Audit whether BRIGHT post-event SAR is compatible with public SAR encoders.

This script is deliberately read-only.  It measures the actual on-disk input
contract per split/event and reports *input* compatibility separately from
task/feature compatibility.  It never opens labels or the test set for model
selection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def event_name(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[0]


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def channel_contract(array: np.ndarray) -> str:
    if array.ndim == 2:
        return "scalar"
    if array.ndim == 3 and array.shape[-1] == 1:
        return "scalar"
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        rgb = array[..., :3]
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]):
            return "repeated_scalar"
        return f"multiband_{array.shape[-1]}"
    return f"unsupported_{tuple(array.shape)}"


def scalar_view(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array
    return array[..., 0]


def summarize_split(
    dataset_root: Path,
    split_file: Path,
    max_images_per_event: int,
    pixel_stride: int,
) -> dict:
    grouped: dict[str, list[str]] = defaultdict(list)
    for sample_id in read_ids(split_file):
        grouped[event_name(sample_id)].append(sample_id)

    event_reports = {}
    split_pixels = []
    contracts = Counter()
    dtypes = Counter()
    missing = []
    for event, sample_ids in sorted(grouped.items()):
        sampled = []
        audit_count = min(len(sample_ids), max_images_per_event)
        # Cover the whole split list instead of taking one spatially adjacent prefix.
        selected_indices = np.linspace(
            0, len(sample_ids) - 1, num=audit_count, dtype=np.int64
        )
        for sample_index in selected_indices:
            sample_id = sample_ids[int(sample_index)]
            path = dataset_root / "post-event" / f"{sample_id}_post_disaster.tif"
            if not path.is_file():
                missing.append(str(path))
                continue
            array = np.asarray(Image.open(path))
            contracts[channel_contract(array)] += 1
            dtypes[str(array.dtype)] += 1
            values = scalar_view(array).astype(np.float32, copy=False).reshape(-1)[::pixel_stride]
            sampled.append(values)
            split_pixels.append(values)
        if sampled:
            values = np.concatenate(sampled)
            event_reports[event] = {
                "listed_images": len(sample_ids),
                "audited_images": audit_count,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "p01": float(np.percentile(values, 1)),
                "p50": float(np.percentile(values, 50)),
                "p99": float(np.percentile(values, 99)),
            }

    values = np.concatenate(split_pixels) if split_pixels else np.empty(0, dtype=np.float32)
    histogram, _ = np.histogram(values, bins=256, range=(0, 256)) if values.size else (np.zeros(256), None)
    histogram = histogram.astype(np.float64)
    histogram /= max(histogram.sum(), 1.0)
    return {
        "split_file": str(split_file),
        "listed_images": sum(map(len, grouped.values())),
        "events": event_reports,
        "channel_contracts": dict(contracts),
        "dtypes": dict(dtypes),
        "sampled_mean": float(values.mean()) if values.size else None,
        "sampled_std": float(values.std()) if values.size else None,
        "histogram_256": histogram.tolist(),
        "missing_count": len(missing),
        "missing_examples": missing[:10],
    }


def js_divergence(left: list[float], right: list[float]) -> float:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    middle = 0.5 * (p + q)
    keep_p = p > 0
    keep_q = q > 0
    return float(0.5 * np.sum(p[keep_p] * np.log2(p[keep_p] / middle[keep_p])) +
                 0.5 * np.sum(q[keep_q] * np.log2(q[keep_q] / middle[keep_q])))


def candidate_contracts(observed: set[str], dtypes: set[str]) -> dict:
    scalar_8bit = observed <= {"scalar", "repeated_scalar"} and dtypes == {"uint8"}
    return {
        "CROMA_radar": {
            "direct_weight_replacement": False,
            "reason": "requires two Sentinel-1 backscatter channels; BRIGHT provides one VV or HH 8-bit amplitude view",
            "allowed_next_check": "frozen encoder plus learned 1-to-2 input adapter and event-held-out linear probe",
        },
        "SARMAE": {
            "direct_weight_replacement": False,
            "photometric_adapter_possible": bool(scalar_8bit),
            "reason": "channel replication can satisfy tensor shape, but sensor, pretraining task and feature scale still differ",
            "allowed_next_check": "frozen multilevel feature probe before any residual injection",
        },
        "SARDet_100K_MSFA": {
            "direct_weight_replacement": False,
            "reason": "object-detection backbone/framework weights span different bands, polarizations, sensors and targets",
            "allowed_next_check": "migrate the staged adaptation recipe or probe a matching backbone; do not load the detector head",
        },
        "HeteCD": {
            "direct_weight_replacement": False,
            "reason": "checkpoint is a paired binary XiongAn change detector, not an interchangeable SAR encoder for three-level BRIGHT damage",
            "allowed_next_check": "retrain the full non-shared paired front-end as a separate controlled baseline",
        },
        "DSTCD": {
            "direct_weight_replacement": False,
            "reason": "the transferable object is the registration-pretrained cross-modal representation, not a standalone SAR branch",
            "allowed_next_check": "only after a displacement audit shows registration is a material error source",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("bda_benchmark/dataset"))
    parser.add_argument("--split-root", type=Path, default=Path("bda_benchmark/dataset/splitname/standard_ML"))
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--max-images-per-event", type=int, default=32)
    parser.add_argument("--pixel-stride", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("bda_benchmark/experiments/sar_transfer_contract_audit.json"))
    args = parser.parse_args()

    reports = {}
    for split in args.splits:
        reports[split] = summarize_split(
            args.dataset_root,
            args.split_root / f"{split}_set.txt",
            args.max_images_per_event,
            args.pixel_stride,
        )

    divergences = {}
    if "train" in reports:
        for split, report in reports.items():
            if split != "train":
                divergences[f"train_to_{split}_js_bits"] = js_divergence(
                    reports["train"]["histogram_256"], report["histogram_256"]
                )

    observed = set().union(*(report["channel_contracts"] for report in reports.values()))
    dtypes = set().union(*(report["dtypes"] for report in reports.values()))
    result = {
        "protocol": "BRIGHT_SAR_transfer_contract_audit_v1",
        "read_only": True,
        "test_used_for_selection": False,
        "splits": reports,
        "split_histogram_divergence": divergences,
        "candidate_transfer_contracts": candidate_contracts(observed, dtypes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("BRIGHT SAR transfer-contract audit")
    for split, report in reports.items():
        print(
            f"{split:>5}: images={report['listed_images']} "
            f"contract={report['channel_contracts']} dtype={report['dtypes']} "
            f"mean/std={report['sampled_mean']:.2f}/{report['sampled_std']:.2f}"
        )
    for name, value in divergences.items():
        print(f"{name}: {value:.6f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
