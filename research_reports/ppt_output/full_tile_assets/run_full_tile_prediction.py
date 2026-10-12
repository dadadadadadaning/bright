"""Run the archived MyModel01 checkpoint on one full BRIGHT test tile.

This utility intentionally performs only inference and figure export.  The
candidate region-memory architecture shown in the paper figure is not claimed
to be implemented by this checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.models as tv_models
from PIL import Image


MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
COLORS = np.asarray(
    [
        [255, 255, 255],
        [70, 181, 121],
        [228, 189, 139],
        [182, 70, 69],
    ],
    dtype=np.uint8,
)


def load_archived_model(source_path: Path, checkpoint_path: Path) -> torch.nn.Module:
    loader = importlib.machinery.SourceFileLoader("archived_mymodel01", str(source_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError(f"Cannot create import spec for {source_path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)

    # The archived constructor requested torchvision weights even though a full
    # checkpoint is loaded immediately afterwards.  Disable that network access.
    original_swin_t = tv_models.swin_t
    module.swin_t = lambda weights=None, **kwargs: original_swin_t(weights=None, **kwargs)
    model = module.MyModel01()

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    clean_state = {key.removeprefix("module."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def image_tensor(path: Path, sar: bool) -> tuple[torch.Tensor, np.ndarray]:
    array = np.asarray(Image.open(path), dtype=np.float32)
    if sar:
        array = np.repeat(array[..., None], 3, axis=2)
    else:
        array = array[..., :3]
    normalized = (array - MEAN) / STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, array.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pre_path = args.data_root / "pre-event" / f"{args.sample}_pre_disaster.tif"
    post_path = args.data_root / "post-event" / f"{args.sample}_post_disaster.tif"
    pre_tensor, pre_rgb = image_tensor(pre_path, sar=False)
    post_tensor, post_rgb = image_tensor(post_path, sar=True)

    started = time.time()
    model = load_archived_model(args.model_source, args.checkpoint)
    with torch.inference_mode():
        loc_logits, damage_logits = model(pre_tensor, post_tensor)
        prediction = damage_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        location = loc_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    elapsed = time.time() - started

    pred_rgb = COLORS[prediction]
    Image.fromarray(pre_rgb).save(args.output_dir / f"{args.sample}_rgb.png")
    Image.fromarray(post_rgb).save(args.output_dir / f"{args.sample}_sar.png")
    Image.fromarray(pred_rgb).save(args.output_dir / f"{args.sample}_prediction.png")
    Image.fromarray(prediction).save(args.output_dir / f"{args.sample}_prediction_index.png")
    Image.fromarray(location * 255).save(args.output_dir / f"{args.sample}_location.png")

    summary = {
        "sample": args.sample,
        "checkpoint": str(args.checkpoint),
        "model_source": str(args.model_source),
        "device": "cpu",
        "elapsed_seconds_including_model_load": elapsed,
        "class_pixel_counts": {
            str(label): int((prediction == label).sum()) for label in range(4)
        },
        "input_size": list(pre_rgb.shape[:2]),
    }
    (args.output_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
