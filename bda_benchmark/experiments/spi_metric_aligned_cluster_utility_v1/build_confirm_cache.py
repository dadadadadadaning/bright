from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_cluster_detail_d1_v1 import PROTOCOL
from bda_benchmark.experiments.spi_cluster_detail_d1_v1.build_cache import main as build_d1_cache


FROZEN_CONFIGURATION_KEYS = (
    "minimum_building_area",
    "minimum_cluster_area",
    "feature_channels",
    "matching_window",
    "halo_radius",
    "seed",
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        "Build one-time MACU meta_confirm cache with train-frozen D1 settings"
    )
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--confirm-list", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args(argv)

    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    if train.get("protocol") != PROTOCOL or train.get("split") != "meta_train":
        raise RuntimeError("expected the frozen D1 meta_train cache")
    if train.get("meta_confirm_accessed", True):
        raise RuntimeError("training cache provenance is contaminated")
    configuration = train.get("configuration", {})
    missing = [key for key in FROZEN_CONFIGURATION_KEYS if key not in configuration]
    if missing:
        raise RuntimeError(f"training cache lacks frozen D1 settings: {missing}")
    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing meta_confirm cache: {output}")

    forwarded = [
        "--checkpoint", args.checkpoint,
        "--dataset-path", args.dataset_path,
        "--list-path", args.confirm_list,
        "--sar-stats-json", args.sar_stats_json,
        "--output", args.output,
        "--split", "meta_confirm",
        "--expected-images", str(args.expected_images),
        "--max-images", "0",
        "--device", args.device,
        "--num-workers", str(args.num_workers),
        "--minimum-building-area", str(configuration["minimum_building_area"]),
        "--minimum-cluster-area", str(configuration["minimum_cluster_area"]),
        "--feature-channels", str(configuration["feature_channels"]),
        "--matching-window", str(configuration["matching_window"]),
        "--halo-radius", str(configuration["halo_radius"]),
        "--progress-every", str(args.progress_every),
        "--checkpoint-every", str(args.checkpoint_every),
        "--seed", str(configuration["seed"]),
    ]
    print(
        "MACU confirm cache uses train-frozen D1 settings: "
        + ", ".join(f"{key}={configuration[key]}" for key in FROZEN_CONFIGURATION_KEYS),
        flush=True,
    )
    return build_d1_cache(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
