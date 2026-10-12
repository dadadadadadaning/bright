"""Build Train-only Memory from an immutable, explicitly unfinished snapshot.

The original parent's completion flag is recorded, never promoted to True.
Only extraction scope changes in --smoke (four distinct-event Train images).
"""
from __future__ import annotations

import argparse
import json

import torch

from bda_benchmark.experiments.damageformer_srac_fourstage_v2.metadata import disaster_id
from bda_benchmark.experiments.damageformer_srac_joint123_v1.joint_runtime import module_fingerprint

from .common import (
    MEMORY_FORMAT, digest, event, lists, load_front, loader, new_output,
    parse_batch, region_targets, save_json, save_pt, seed_everything, source_hash,
)


def parser():
    p = argparse.ArgumentParser(__doc__)
    for name in ("checkpoint", "a1_checkpoint", "dataset", "train_list", "val_list", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--per_event_class", type=int, default=64)
    p.add_argument("--min_purity", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    return p


def run(args):
    if args.per_event_class < 1 or not 0 <= args.min_purity <= 1:
        raise ValueError("Invalid reservoir settings")
    train_names, _val_names = lists(args.train_list, args.val_list)
    device = torch.device(args.device)
    seed_everything(args.seed)
    model, front = load_front(args.checkpoint, device, args.a1_checkpoint)
    model.eval().requires_grad_(False)
    if front.get("smoke") is not False:
        raise ValueError("Snapshot protocol requires a formal parent, even for a bank smoke")
    if front["train_sha256"] != digest(args.train_list) or front["val_sha256"] != digest(args.val_list):
        raise ValueError("Split hash mismatch")
    if args.smoke:
        chosen, seen = [], set()
        for name in train_names:
            if event(name) not in seen:
                chosen.append(name)
                seen.add(event(name))
            if len(chosen) == 4:
                break
        if len(chosen) != 4:
            raise ValueError("Bank smoke requires four distinct Train events")
        train_names = chosen

    before = module_fingerprint(model)
    output = new_output(args.output)
    (output / "shards").mkdir()
    event_names = sorted({event(name) for name in train_names})
    manifest = dict(
        complete=False,
        format=MEMORY_FORMAT,
        snapshot_protocol=True,
        parent_run_complete=bool(front.get("run_complete", False)),
        parent_checkpoint_step=front["step"],
        declared_parent_steps=front["arguments"]["steps"],
        split="train",
        smoke=args.smoke,
        test_accessed=False,
        checkpoint_sha256=digest(args.checkpoint),
        source_hash=source_hash(),
        train_sha256=digest(args.train_list),
        val_sha256=digest(args.val_list),
        config=front["config"],
        event_names=event_names,
        image_count=len(train_names),
        per_event_class=args.per_event_class,
        min_purity=args.min_purity,
        seed=args.seed,
        shards=[],
        source_fingerprint_before=before,
        source_weights_and_buffers_unchanged=None,
        gt_role="Train labels create support distributions only; GT never changes Q or retrieval keys",
        extraction="deterministic full Train images at 1024; token shards stored FP16",
        reservoir="independent event x dominant-class random-priority reservoirs; unchanged from parent implementation",
        retrieval_exclusion="configured later; neighbor IDs never use query GT",
    )
    save_json(output / "manifest.json", manifest)
    reservoir = {}
    generator = torch.Generator().manual_seed(args.seed)
    with torch.inference_mode():
        for image_index, batch in enumerate(loader(args.dataset, train_names, workers=args.workers)):
            pre, post, _loc, target, names = parse_batch(batch, device)
            result = model(pre, post)
            distribution, weights = region_targets(target, result)
            tokens = result["tokens"][0].float().cpu()
            distribution = distribution[0].float().cpu()
            weights = weights[0].float().cpu()
            geometry = result["geometry"][0].float().cpu()
            mass = result["mass"][0].float().cpu()
            event_index = event_names.index(event(names[0]))
            shard = dict(
                tokens=tokens.half(), distribution=distribution, weights=weights,
                geometry=geometry.half(), mass=mass, event=event_index,
                image=image_index, name=names[0], region_ids=torch.arange(len(tokens)),
            )
            filename = f"shards/{image_index:05d}.pt"
            save_pt(output / filename, shard)
            manifest["shards"].append(dict(path=filename, sha256=digest(output / filename)))

            purity, dominant = distribution.max(dim=-1)
            eligible = (weights > 0) & (mass > 1e-6) & (purity >= args.min_purity)
            for class_index in range(4):
                ids = (eligible & (dominant == class_index)).nonzero().flatten()
                if not len(ids):
                    continue
                candidate = dict(
                    tokens=tokens[ids].half(), distribution=distribution[ids],
                    events=torch.full((len(ids),), event_index, dtype=torch.long),
                    images=torch.full((len(ids),), image_index, dtype=torch.long),
                    regions=ids.long(), geometry=geometry[ids].half(),
                    disasters=torch.full((len(ids),), disaster_id(event_names[event_index]), dtype=torch.long),
                    priority=torch.rand(len(ids), generator=generator),
                )
                key = (event_index, class_index)
                if key in reservoir:
                    candidate = {name: torch.cat((reservoir[key][name], value)) for name, value in candidate.items()}
                keep = candidate["priority"].topk(min(args.per_event_class, len(candidate["tokens"]))).indices
                reservoir[key] = {name: value[keep].clone() for name, value in candidate.items()}
            if (image_index + 1) % 25 == 0 or image_index + 1 == len(train_names):
                print(f"cache {image_index + 1}/{len(train_names)} regions={len(tokens)}", flush=True)

    if not reservoir:
        raise RuntimeError("No eligible Train regions entered Memory")
    fields = [name for name in next(iter(reservoir.values())) if name != "priority"]
    bank = {name: torch.cat([bucket[name] for _key, bucket in sorted(reservoir.items())]) for name in fields}
    bank.update(event_names=event_names, checkpoint_sha256=manifest["checkpoint_sha256"])
    save_pt(output / "bank.pt", bank)
    after = module_fingerprint(model)
    unchanged = before == after
    manifest.update(source_fingerprint_after=after, source_weights_and_buffers_unchanged=unchanged)
    if not unchanged:
        save_json(output / "manifest.json", manifest)
        raise RuntimeError("Snapshot weights/buffers changed during bank extraction; bank remains incomplete")
    class_counts = torch.bincount(bank["distribution"].argmax(-1), minlength=4).tolist()
    manifest.update(
        complete=True, bank_sha256=digest(output / "bank.pt"),
        bank_regions=len(bank["tokens"]), bank_dominant_class_counts=class_counts,
    )
    save_json(output / "manifest.json", manifest)
    print(json.dumps({"bank_regions": len(bank["tokens"]), "class_counts": class_counts}), flush=True)
    print("SNAPSHOT TRAIN MEMORY COMPLETE. Parent completion unchanged. No Val/Test entered the bank.", flush=True)


if __name__ == "__main__":
    run(parser().parse_args())
