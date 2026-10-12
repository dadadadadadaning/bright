"""Precompute immutable Train-query neighbor IDs without using query labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .common import digest, new_output, read_cache, save_json, source_hash


def _allowed(bank, event, image, exclusion):
    if exclusion == "same_image":
        return bank["images"] != image
    if exclusion == "cross_event":
        return (bank["images"] != image) & (bank["events"] != event)
    raise ValueError(exclusion)


def select_ids(query, bank, event, image, topk, policy="global", exclusion="cross_event"):
    scores = F.normalize(query.float(), dim=-1) @ F.normalize(bank["tokens"].float(), dim=-1).T
    allowed = _allowed(bank, event, image, exclusion)
    scores = scores.masked_fill(~allowed[None], -torch.inf)
    if policy == "global":
        values, indices = scores.topk(min(topk, len(bank["tokens"])), dim=1)
        indices = indices.masked_fill(~torch.isfinite(values), -1)
        if indices.shape[1] < topk:
            indices = torch.cat((indices, indices.new_full((len(indices), topk - indices.shape[1]), -1)), dim=1)
        return indices
    if policy != "per_class":
        raise ValueError(policy)
    if topk % 4:
        raise ValueError("per_class policy requires topk divisible by four")
    each = topk // 4
    dominant = bank["distribution"].argmax(-1)
    result = []
    for class_index in range(4):
        class_scores = scores.masked_fill(dominant[None] != class_index, -torch.inf)
        values, indices = class_scores.topk(min(each, len(bank["tokens"])), dim=1)
        indices = indices.masked_fill(~torch.isfinite(values), -1)
        if indices.shape[1] < each:
            indices = torch.cat((indices, indices.new_full((len(indices), each - indices.shape[1]), -1)), dim=1)
        result.append(indices)
    return torch.cat(result, dim=1)


def gather_support(indices, bank):
    valid = indices >= 0
    safe = indices.clamp_min(0).long()
    return (
        bank["tokens"][safe].float(),
        bank["distribution"][safe].float(),
        valid,
        {
            "geometry": bank["geometry"][safe].float(),
            "events": bank["events"][safe],
            "disasters": bank["disasters"][safe],
        },
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("checkpoint", "cache", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--policy", choices=("global", "per_class"), default="global")
    parser.add_argument("--exclusion", choices=("same_image", "cross_event"), default="cross_event")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.topk < 1 or (args.policy == "per_class" and args.topk % 4):
        parser.error("Invalid topk for selected policy")
    cache_root, cache, bank = read_cache(args.cache, args.checkpoint)
    device = torch.device(args.device)
    search_bank = {
        name: value.to(device)
        for name, value in bank.items()
        if name in ("tokens", "distribution", "events", "images")
    }
    output = new_output(args.output)
    offsets, total = [], 0
    for record in cache["shards"]:
        if digest(cache_root / record["path"]) != record["sha256"]:
            raise ValueError("Shard hash mismatch: " + record["path"])
        shard = torch.load(str(cache_root / record["path"]), map_location="cpu", weights_only=False, mmap=True)
        offsets.append(dict(start=total, count=len(shard["tokens"])))
        total += len(shard["tokens"])
    table = np.lib.format.open_memmap(
        output / "neighbors.npy", mode="w+", dtype=np.int32, shape=(total, args.topk)
    )
    with torch.inference_mode():
        for shard_index, (record, offset) in enumerate(zip(cache["shards"], offsets)):
            shard = torch.load(str(cache_root / record["path"]), map_location="cpu", weights_only=False, mmap=True)
            for start in range(0, offset["count"], 256):
                indices = select_ids(
                    shard["tokens"][start : start + 256].float().to(device),
                    search_bank,
                    shard["event"],
                    shard["image"],
                    args.topk,
                    args.policy,
                    args.exclusion,
                )
                begin = offset["start"] + start
                table[begin : begin + len(indices)] = indices.cpu().numpy().astype(np.int32)
            if (shard_index + 1) % 25 == 0 or shard_index + 1 == len(offsets):
                print(f"neighbors {shard_index + 1}/{len(offsets)}", flush=True)
    table.flush()
    del table
    manifest = dict(
        complete=True,
        source_hash=source_hash(),
        cache_sha256=digest(cache_root / "manifest.json"),
        bank_sha256=cache["bank_sha256"],
        table_sha256=digest(output / "neighbors.npy"),
        topk=args.topk,
        policy=args.policy,
        exclusion=args.exclusion,
        device=str(device),
        offsets=offsets,
        query_gt_used=False,
        test_accessed=False,
    )
    save_json(output / "manifest.json", manifest)


def open_table(path, cache_root, bank_hash, topk, policy, exclusion):
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("complete") or manifest["source_hash"] != source_hash():
        raise ValueError("Incomplete or source-incompatible neighbor table")
    if (manifest["topk"], manifest["policy"], manifest["exclusion"]) != (topk, policy, exclusion):
        raise ValueError("Neighbor policy mismatch")
    if manifest["cache_sha256"] != digest(Path(cache_root) / "manifest.json") or manifest["bank_sha256"] != bank_hash:
        raise ValueError("Neighbor table points to another cache/bank")
    if manifest["table_sha256"] != digest(root / "neighbors.npy"):
        raise ValueError("Neighbor table is corrupted")
    return np.load(root / "neighbors.npy", mmap_mode="r"), manifest


if __name__ == "__main__":
    main()

