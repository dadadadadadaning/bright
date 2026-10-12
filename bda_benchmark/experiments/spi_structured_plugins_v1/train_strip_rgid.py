from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from bda_benchmark.experiments.rgid_s9ic_clean.rgid.calibration import (
    BoundaryThresholdCalibrator,
)
from bda_benchmark.experiments.rgid_s9ic_clean.rgid.config import RGIDConfig
from bda_benchmark.experiments.rgid_s9ic_clean.rgid.targets_losses import RGIDLoss
from bda_benchmark.experiments.spi_rgid_semantic_v1.fast_targets import (
    build_rgid_targets_fast,
)
from bda_benchmark.experiments.spi_rgid_semantic_v1.scripts.train_online_rgid import (
    OnlineFrozenP3Features,
    _names,
    _seed,
    _source_loader,
    build_base,
)
from bda_benchmark.experiments.spi_rgid_semantic_v1.training import (
    train_rgid_head_fast,
)
from bda_benchmark.model.structured_plugins.rgid_head import (
    build_rgid_head,
    load_rgid_head,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def _calibrate(
    checkpoint: Path,
    base: torch.nn.Module,
    source,
    device: torch.device,
    maximum_samples: int,
) -> tuple[Path, dict]:
    head, config, payload = load_rgid_head(checkpoint, device)
    calibrator = BoundaryThresholdCalibrator()
    seen = 0
    for batch in OnlineFrozenP3Features(source, base, device):
        fused = batch["fused_roi"].to(device).float()
        with torch.autocast(
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
    result = calibrator.compute()
    row = {
        "boundary_positive_median": result.boundary_positive_median,
        "boundary_interior_median": result.boundary_interior_median,
        "boundary_gate_threshold": result.boundary_gate_threshold,
        "num_boundary_pixels": result.num_boundary_pixels,
        "num_interior_pixels": result.num_interior_pixels,
        "calibration_samples": seen,
        "calibration_split": "meta_train_only",
    }
    payload["boundary_calibration"] = row
    output = checkpoint.with_name(checkpoint.stem + "_calibrated.pth")
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    return output, row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train Strip-RGID on frozen P3 features")
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
    parser.add_argument("--strip-kernel", type=int, default=19)
    parser.add_argument("--seed", type=int, default=7341)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0 or args.accumulation <= 0 or args.batch_size <= 0:
        raise ValueError("steps, accumulation and batch size must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Strip-RGID online training requires CUDA")
    p3_path = Path(args.checkpoint).resolve()
    dataset_root = Path(args.dataset_path).resolve()
    train_list = Path(args.train_list).resolve()
    output_dir = Path(args.output_dir).resolve()
    for path in (p3_path, train_list):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = _names(train_list)
    base = build_base(p3_path, device).eval().requires_grad_(False)
    config = RGIDConfig()
    training_source = _source_loader(
        dataset_root,
        names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        generator=_seed(args.seed),
        shuffle=True,
    )
    head = build_rgid_head("strip", config, strip_kernel=args.strip_kernel)
    train_rgid_head_fast(
        head=head,
        train_loader=OnlineFrozenP3Features(training_source, base, device),
        config=config,
        output_dir=output_dir,
        max_optimizer_steps=args.steps,
        accumulation_steps=args.accumulation,
        learning_rate=1e-4,
        weight_decay=1e-4,
        checkpoint_steps=tuple(
            step for step in (100, 300, 500, args.steps) if step <= args.steps
        ),
        device=device,
        criterion=RGIDLoss(config),
        log_every=20,
        checkpoint_extra={
            "head_type": "strip",
            "strip_kernel": args.strip_kernel,
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
        raw_checkpoint,
        base,
        calibration_source,
        device,
        args.calibration_samples,
    )
    provenance = {
        "protocol": "RGID_STRUCTURED_PLUGIN_TRAIN_v1",
        "head_type": "strip",
        "strip_kernel": args.strip_kernel,
        "official_instances_accessed": False,
        "standard_ml_val_accessed": False,
        "original_standard_test_accessed": False,
        "calibration_split": "train_only",
        "training_mode": "online_frozen_p3_no_cache",
        "p3_sha256": _sha256(p3_path),
        "p3_checkpoint": str(p3_path),
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
