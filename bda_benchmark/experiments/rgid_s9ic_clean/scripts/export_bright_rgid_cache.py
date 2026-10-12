#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader

# The legacy BRIGHT loader imports matplotlib although cache export never
# plots.  Keep that import headless and make local Windows smoke checks robust.
os.environ.setdefault("MPLBACKEND", "Agg")
if os.name == "nt":
    os.environ.setdefault("WINDIR", os.environ.get("SystemRoot", r"C:\Windows"))

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
BDA_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = BDA_ROOT.parent
for root in (PROJECT_ROOT, BDA_ROOT, EXPERIMENT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from bda_benchmark.dataset.make_data_loader import (  # noqa: E402
    MultimodalDamageAssessmentDatset,
)
from bda_benchmark.model.instance_geometry.s8_backbone import (  # noqa: E402
    FrozenS8Backbone,
)
from bda_benchmark.model.region_ops import (  # noqa: E402
    build_instance_map_from_binary,
)
from bda_benchmark.script.standard_ML.s9_instance_common import (  # noqa: E402
    normalize_label_tensor,
)

try:  # Present on the target server branch, absent in some local snapshots.
    from bda_benchmark.script.standard_ML.eval_S9I_center_boundary import (  # type: ignore # noqa: E402
        build_base as _project_build_base,
    )
except ModuleNotFoundError:
    _project_build_base = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_base(checkpoint: str | Path, device: torch.device):
    """Build the accepted P3 base through the project's strict S8 loader."""

    if _project_build_base is not None:
        base = _project_build_base(str(checkpoint), device)
    else:
        frozen = FrozenS8Backbone.from_checkpoint(
            checkpoint,
            feature_key="fused_roi",
            expected_channels=256,
            allow_unsafe=True,
        )
        base = frozen.model
    base = base.to(device)
    base.eval()
    base.requires_grad_(False)
    return base


def _atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_names(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as file:
        names = [line.strip() for line in file if line.strip()]
    if not names:
        raise ValueError(f"data list is empty: {path}")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Export frozen BRIGHT fused_roi RGID cache")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-list", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    data_list = Path(args.data_list).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not data_list.is_file():
        raise FileNotFoundError(data_list)
    if not dataset_path.is_dir():
        raise NotADirectoryError(dataset_path)
    if output_dir == Path(output_dir.anchor):
        raise ValueError("output-dir cannot be a filesystem root")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.pt"))
    manifest_path = output_dir / "manifest.json"
    if (existing or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"cache output is not empty: {output_dir}; pass --overwrite"
        )
    if args.overwrite:
        for path in existing:
            path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    names = _load_names(data_list)
    dataset = MultimodalDamageAssessmentDatset(
        str(dataset_path),
        names,
        1024,
        None,
        "test",
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    base = build_base(checkpoint, device)
    base.eval()
    base.requires_grad_(False)

    samples: List[Dict[str, object]] = []
    empty_gt_count = 0
    empty_parent_count = 0
    saved_index = 0
    with torch.inference_mode():
        for batch in loader:
            pre, post, loc, _, batch_names = batch
            pre = pre.to(device, non_blocking=True).float()
            post = post.to(device, non_blocking=True).float()
            loc = normalize_label_tensor(torch.as_tensor(loc).long())
            output_loc, _, aux = base(
                pre,
                post,
                bag_specs=None,
                return_aux=True,
            )
            if "fused_roi" not in aux:
                raise KeyError("base auxiliary output has no fused_roi")
            fused_batch = aux["fused_roi"]
            if fused_batch.ndim != 4 or int(fused_batch.shape[1]) != 256:
                raise ValueError(
                    f"base fused_roi must be [B,256,Hh,Wh], "
                    f"got {tuple(fused_batch.shape)}"
                )
            gt_building = (loc > 0) & (loc != 255)
            instance_batch = build_instance_map_from_binary(
                gt_building,
                min_area=32,
            )
            parent_batch = output_loc.detach().argmax(dim=1) == 1

            for batch_index in range(int(pre.shape[0])):
                fused_roi = (
                    fused_batch[batch_index].detach().cpu().float().contiguous()
                )
                instance_ids = (
                    instance_batch[batch_index].detach().cpu().long().contiguous()
                )
                parent_mask = (
                    parent_batch[batch_index].detach().cpu().bool().contiguous()
                )
                original_name = str(batch_names[batch_index])
                if int(instance_ids.max().item()) == 0:
                    empty_gt_count += 1
                if not bool(parent_mask.any().item()):
                    empty_parent_count += 1
                filename = f"{saved_index:06d}.pt"
                payload = {
                    "fused_roi": fused_roi,
                    "instance_ids": instance_ids,
                    "parent_mask": parent_mask,
                    "name": original_name,
                    "metadata": {
                        "base_checkpoint": os.path.abspath(checkpoint),
                        "feature_key": "fused_roi",
                        "feature_channels": 256,
                    },
                }
                _atomic_torch_save(payload, output_dir / filename)
                samples.append(
                    {
                        "file": filename,
                        "name": original_name,
                        "fused_roi_shape": list(fused_roi.shape),
                        "instance_ids_shape": list(instance_ids.shape),
                        "parent_mask_shape": list(parent_mask.shape),
                    }
                )
                saved_index += 1

    manifest = {
        "sample_count": saved_index,
        "names": [sample["name"] for sample in samples],
        "samples": samples,
        "empty_gt_count": empty_gt_count,
        "empty_parent_count": empty_parent_count,
        "base_checkpoint": os.path.abspath(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "data_list": os.path.abspath(data_list),
        "dataset_path": os.path.abspath(dataset_path),
        "deterministic_dataset_mode": "test",
        "crop_size": 1024,
    }
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
