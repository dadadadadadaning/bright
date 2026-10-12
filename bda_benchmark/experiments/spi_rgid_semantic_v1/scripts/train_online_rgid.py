from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

os.environ.setdefault("MPLBACKEND", "Agg")

BDA_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = BDA_ROOT.parent
LEGACY_RGID_ROOT = BDA_ROOT / "experiments" / "rgid_s9ic_clean"
for root in (PROJECT_ROOT, BDA_ROOT, LEGACY_RGID_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.rgid_s9ic_clean.scripts.export_bright_rgid_cache import (
    build_base,
)
from bda_benchmark.model.region_ops import build_instance_map_from_binary
from bda_benchmark.script.standard_ML.s9_instance_common import normalize_label_tensor
from rgid.calibration import BoundaryThresholdCalibrator
from rgid.config import RGIDConfig
from rgid.evaluation import load_head_checkpoint
from rgid.model import RGIDHead
from rgid.targets_losses import RGIDLoss
from bda_benchmark.experiments.spi_rgid_semantic_v1.fast_targets import build_rgid_targets_fast
from bda_benchmark.experiments.spi_rgid_semantic_v1.training import train_rgid_head_fast


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _names(path: Path) -> list[str]:
    rows = [row.strip().rstrip("\r") for row in path.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("training list must be non-empty and duplicate-free")
    return rows


def _seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _worker_seed(_: int) -> None:
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)


def _collate_with_pseudo_instances(rows):
    pre, post, localization, damage, names = default_collate(rows)
    localization = normalize_label_tensor(torch.as_tensor(localization).long())
    pseudo_instances = build_instance_map_from_binary(
        (localization > 0) & (localization != 255), min_area=32
    )
    return pre, post, localization, damage, names, pseudo_instances


def _source_loader(
    dataset_root: Path,
    names: list[str],
    *,
    batch_size: int,
    num_workers: int,
    generator: torch.Generator,
    shuffle: bool,
) -> DataLoader:
    dataset = MultimodalDamageAssessmentDatset(
        str(dataset_root), names, 1024, None, "train" if shuffle else "test"
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_seed,
        generator=generator,
        collate_fn=_collate_with_pseudo_instances,
    )


class OnlineFrozenP3Features:
    """Recompute frozen P3 features per batch; writes no feature cache."""

    def __init__(self, source: DataLoader, base: torch.nn.Module, device: torch.device):
        self.source = source
        self.base = base
        self.device = device

    def __iter__(self):
        for pre, post, localization, _, batch_names, pseudo_instances in self.source:
            pre = pre.to(self.device, non_blocking=True).float()
            post = post.to(self.device, non_blocking=True).float()
            localization = normalize_label_tensor(torch.as_tensor(localization).long())
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                output_loc, _, auxiliary = self.base(
                    pre, post, bag_specs=None, return_aux=True
                )
            fused = auxiliary.get("fused_roi")
            if fused is None or fused.ndim != 4 or fused.shape[1] != 256:
                raise ValueError("P3 must return fused_roi [B,256,Hh,Wh]")
            yield {
                "fused_roi": fused.detach(),
                "instance_ids": pseudo_instances.cpu(),
                "parent_mask": (output_loc.detach().argmax(dim=1) == 1).cpu(),
                "name": [str(value) for value in batch_names],
            }


def _calibrate(
    *,
    checkpoint: Path,
    base: torch.nn.Module,
    source: DataLoader,
    device: torch.device,
    maximum_samples: int,
) -> tuple[Path, dict]:
    head, config, payload = load_head_checkpoint(checkpoint, device=device)
    calibrator = BoundaryThresholdCalibrator()
    seen = 0
    for batch in OnlineFrozenP3Features(source, base, device):
        fused = batch["fused_roi"].to(device).float()
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = head(fused)
        targets, _ = build_rgid_targets_fast(
            batch["instance_ids"], prediction["boundary_logits"].shape[-2:], config, device
        )
        calibrator.update(
            prediction["boundary_logits"],
            targets["boundary_target"],
            building_mask=batch["parent_mask"].to(device),
        )
        seen += fused.shape[0]
        if seen >= maximum_samples:
            break
    calibration = calibrator.compute()
    row = {
        "boundary_positive_median": calibration.boundary_positive_median,
        "boundary_interior_median": calibration.boundary_interior_median,
        "boundary_gate_threshold": calibration.boundary_gate_threshold,
        "num_boundary_pixels": calibration.num_boundary_pixels,
        "num_interior_pixels": calibration.num_interior_pixels,
        "calibration_samples": seen,
        "calibration_split": "train_only",
    }
    payload["boundary_calibration"] = row
    output = checkpoint.with_name(checkpoint.stem + "_calibrated.pth")
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    return output, row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Cache-free pseudo-only RGID training")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.steps, args.batch_size, args.accumulation, args.calibration_samples) <= 0:
        raise ValueError("steps, batch size, accumulation and calibration samples must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("online RGID training requires CUDA")
    checkpoint = Path(args.checkpoint).resolve()
    dataset_root = Path(args.dataset_path).resolve()
    train_list = Path(args.train_list).resolve()
    output_dir = Path(args.output_dir).resolve()
    for path in (checkpoint, train_list):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = _names(train_list)
    generator = _seed(args.seed)
    base = build_base(checkpoint, device).eval().requires_grad_(False)
    training_source = _source_loader(
        dataset_root,
        names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        generator=generator,
        shuffle=True,
    )
    config = RGIDConfig()
    train_rgid_head_fast(
        head=RGIDHead(config.in_channels, config.hidden_channels, config.num_shared_blocks),
        train_loader=OnlineFrozenP3Features(training_source, base, device),
        config=config,
        output_dir=output_dir,
        max_optimizer_steps=args.steps,
        accumulation_steps=args.accumulation,
        learning_rate=1e-4,
        weight_decay=1e-4,
        checkpoint_steps=tuple(step for step in (100, 300, 500, args.steps) if step <= args.steps),
        device=device,
        criterion=RGIDLoss(config),
        log_every=20,
        checkpoint_extra={
            "training_mode": "online_frozen_p3_no_cache",
            "train_list": str(train_list),
            "seed": args.seed,
        },
    )
    raw_checkpoint = output_dir / f"rgid_step_{args.steps}.pth"
    calibration_source = _source_loader(
        dataset_root,
        names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        generator=_seed(args.seed + 1),
        shuffle=False,
    )
    calibrated, calibration = _calibrate(
        checkpoint=raw_checkpoint,
        base=base,
        source=calibration_source,
        device=device,
        maximum_samples=args.calibration_samples,
    )
    provenance = {
        "protocol": "RGID_PSEUDO_ONLY_TRAIN_v1",
        "official_instances_accessed": False,
        "standard_ml_val_accessed": False,
        "original_standard_test_accessed": False,
        "calibration_split": "train_only",
        "training_mode": "online_frozen_p3_no_cache",
        "p3_sha256": _sha256(checkpoint),
        "p3_checkpoint": str(checkpoint),
        "train_list": str(train_list),
        "train_list_sha256": _sha256(train_list),
        "train_samples": len(names),
        "seed": args.seed,
        "optimizer_steps": args.steps,
        "calibration": calibration,
    }
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(calibrated), "provenance": str(provenance_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
