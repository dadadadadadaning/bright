from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bda_benchmark.model.DamageFormer_ModelCentric_V1 import DamageFormerModelCentricV1
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import (
    build_dataset,
    fast_hist,
    metrics_from_hist,
    parse_batch,
)


@torch.no_grad()
def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Internal-only model-centric plugin evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--list-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--expected-images", type=int, default=0)
    args = parser.parse_args(argv)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("protocol") != "SPI_MODEL_CENTRIC_TRAIN_ONLY_v1":
        raise RuntimeError("checkpoint protocol mismatch")
    model = DamageFormerModelCentricV1(saved["variant"])
    model.load_state_dict(saved["model_state"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    dataset = build_dataset(args.dataset_path, args.list_path, args.crop_size, "test")
    if args.expected_images and len(dataset) != args.expected_images:
        raise RuntimeError(f"expected {args.expected_images} images, found {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    histogram = torch.zeros(4, 4, dtype=torch.float64, device=device)
    parent_histogram = torch.zeros_like(histogram)
    for index, batch in enumerate(loader, start=1):
        pre, post, _, damage, _ = parse_batch(batch, device)
        _, output, aux = model(pre, post, return_aux=True)
        parent = torch.nn.functional.interpolate(
            aux["parent_logits_feature"], damage.shape[-2:], mode="bilinear", align_corners=False
        )
        histogram += fast_hist(output.argmax(1), damage, 4, 255)
        parent_histogram += fast_hist(parent.argmax(1), damage, 4, 255)
        if index % 10 == 0:
            print(f"Model-centric evaluate: {index}/{len(loader)}", flush=True)
    metrics = metrics_from_hist(torch.zeros(2, 2, dtype=torch.float64, device=device), histogram)
    parent_metrics = metrics_from_hist(
        torch.zeros(2, 2, dtype=torch.float64, device=device), parent_histogram
    )
    payload = {
        "protocol": "SPI_MODEL_CENTRIC_INTERNAL_EVAL_v1",
        "variant": saved["variant"],
        "parent": parent_metrics,
        "variant_metrics": metrics,
        "gain_valid_miou_pp": 100.0 * (metrics["valid_miou"] - parent_metrics["valid_miou"]),
        "diagnostics": model.diagnostics(),
        "processed_images": len(dataset),
        "meta_confirm_accessed": False,
        "val450_accessed": False,
        "test_accessed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
