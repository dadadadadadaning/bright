#!/usr/bin/env python3
"""Build exact per-image base confusion matrices for event/bootstrap auditing."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.core import (
    load_a1_parent, load_v2_checkpoint, parse_batch, seed_worker,
)
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.model import (
    DamageFormerBuildingInternalChangePartitionV2,
)
from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2.train import read_list
from bda_benchmark.experiments.damageformer_late_decision_correction_v1.late_core import (
    confusion4, event_name, sha256_file,
)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--val_cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--a1_ckpt", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--val_list", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    if "test" in Path(args.val_list).name.lower() or "test" in Path(args.output).name.lower():
        raise RuntimeError("TEST ACCESS IS FORBIDDEN")
    cache = torch.load(args.val_cache, map_location="cpu", weights_only=False, mmap=True)
    if bool(cache.get("test_accessed", False)):
        raise RuntimeError("test-derived cache is forbidden")
    if cache["source_checkpoint_sha256"] != sha256_file(args.checkpoint):
        raise RuntimeError("checkpoint/cache identity mismatch")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    variant = str(payload.get("variant", "p_sc_residual"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DamageFormerBuildingInternalChangePartitionV2(variant).to(device)
    parent = load_a1_parent(model, args.a1_ckpt)
    region = load_v2_checkpoint(model, args.checkpoint)
    model.configure_stage("change"); model.eval()
    dataset = MultimodalDamageAssessmentDatset(
        args.dataset_path, read_list(args.val_list), 640, type="val"
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.num_workers > 0, worker_init_fn=seed_worker)
    names, events, cms = [], [], []
    for index, batch in enumerate(loader):
        pre, post, _, damage, batch_names = parse_batch(batch, device)
        _, output_damage = model(pre, post)
        pred = output_damage.argmax(1)
        if damage.ndim == 4 and damage.shape[1] == 1:
            damage = damage[:, 0]
        for bi, name in enumerate(batch_names):
            names.append(str(name)); events.append(event_name(str(name)))
            cms.append(confusion4(pred[bi], damage[bi]))
        if (index + 1) % 25 == 0 or len(names) == len(dataset):
            print(f"image cm {len(names)}/{len(dataset)}", flush=True)
    image_cms = torch.stack(cms)
    observed = image_cms.sum(0)
    expected = cache["base_pixel_cm"].long()
    if not torch.equal(observed, expected):
        raise RuntimeError(f"base reproduction failed; max_abs={int((observed-expected).abs().max())}")
    output = {
        "protocol": "DAMAGEFORMER_PROTOTYPE_MEMORY_VALIDATION_V1_IMAGE_CMS",
        "names": names, "events": events, "image_pixel_cm": image_cms,
        "val_cache": str(Path(args.val_cache).resolve()),
        "val_cache_sha256": sha256_file(args.val_cache),
        "checkpoint": region, "parent": parent,
        "exact_base_identity": True, "test_accessed": False,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"Saved {args.output}; exact base identity=True")


if __name__ == "__main__":
    main()

