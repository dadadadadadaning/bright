from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from . import TARGET_PROTOCOL
from .core import exact_marginal_utilities, metric_labels, metric_utilities, miou_gradient


EXPECTED_CACHE_PROTOCOL = "SPI_CLUSTER_DETAIL_D1_CACHE_v1"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(cache: dict, split: str) -> None:
    if cache.get("protocol") != EXPECTED_CACHE_PROTOCOL or cache.get("split") != split:
        raise RuntimeError(f"expected frozen D1 {split} cache")
    if cache.get("meta_confirm_accessed", True) or cache.get("val450_accessed", True) or cache.get("test_accessed", True):
        raise RuntimeError("metric targets received a forbidden cache")
    if int(cache["configuration"]["minimum_cluster_area"]) != 33:
        raise RuntimeError("metric utility is preregistered for area > 32 only")


def atomic_save(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Build train-frozen metric-aligned cluster targets")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--calibration-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epsilon-pp", type=float, default=0.0)
    args = parser.parse_args(argv)

    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    calibration = torch.load(args.calibration_cache, map_location="cpu", weights_only=False)
    validate(train, "meta_train"); validate(calibration, "meta_calibration")
    if train["p3_sha256"] != calibration["p3_sha256"] or set(train["image_names"]) & set(calibration["image_names"]):
        raise RuntimeError("D1 cache provenance mismatch or split overlap")
    gradient = miou_gradient(train["fixed_s2_histograms"].sum(0))
    output_dir = Path(args.output_dir)
    summaries = {}
    for cache, split, source in (
        (train, "meta_train", args.train_cache),
        (calibration, "meta_calibration", args.calibration_cache),
    ):
        utility = metric_utilities(cache["delta_hist"], gradient)
        exact = exact_marginal_utilities(cache)
        labels = metric_labels(utility, args.epsilon_pp)
        payload = {
            "protocol": TARGET_PROTOCOL, "split": split,
            "source_cache_sha256": sha256(source), "p3_sha256": cache["p3_sha256"],
            "minimum_cluster_area": 33, "epsilon_pp": args.epsilon_pp,
            "train_frozen_miou_gradient": gradient,
            "metric_utility_pp": utility, "exact_marginal_utility_pp": exact,
            "labels": labels, "changed_pixels": cache["utility"]["changed"].long(),
            "meta_confirm_accessed": False, "val450_accessed": False,
            "official_instances_accessed": False, "test_accessed": False,
        }
        atomic_save(payload, output_dir / f"{split}_targets.pt")
        summaries[split] = {
            "clusters": len(labels), "positive": int((labels == 1).sum()),
            "negative": int((labels == 0).sum()), "ambiguous": int((labels < 0).sum()),
            "linear_exact_sign_agreement": float(
                ((utility > 0) == (exact > 0)).float().mean()
            ),
            "pearson_linear_exact": float(torch.corrcoef(torch.stack((utility, exact)))[0, 1]),
        }
    manifest = {
        "protocol": TARGET_PROTOCOL, "configuration": vars(args), "summaries": summaries,
        "train_cache_sha256": sha256(args.train_cache),
        "calibration_cache_sha256": sha256(args.calibration_cache),
        "meta_confirm_accessed": False, "val450_accessed": False,
        "official_instances_accessed": False, "test_accessed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Saved metric targets: {output_dir}")
    print("Meta-confirm/Val450/official instances/test accessed: NEVER / NEVER / NEVER / NEVER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
