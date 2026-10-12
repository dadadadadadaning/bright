"""Artifact validation, frozen-feature capture and shared evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_srac_transformer_v1.build_neighbors import (
    gather_support,
    open_table,
)
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import (
    confusion_update,
    device_bank,
    digest,
    event,
    lists,
    load_front,
    parse_batch,
    read_cache,
    save_json,
    save_pt,
    seed_everything,
)
from bda_benchmark.experiments.damageformer_srac_transformer_v1.retrieval import retrieve
from bda_benchmark.experiments.damageformer_srac_predsplit_v1.audit import metrics

from .model import VARIANTS


FORMAT = "SRAC_JOINT_CONTEXT_V1"


def source_hash():
    checksum = hashlib.sha256()
    root = Path(__file__).resolve().parents[3]
    own = sorted(Path(__file__).parent.glob("*.py"))
    inherited = [
        root / "bda_benchmark/experiments/damageformer_srac_transformer_v1/model.py",
        root / "bda_benchmark/experiments/damageformer_srac_transformer_v1/common.py",
        root / "bda_benchmark/experiments/damageformer_srac_transformer_v1/build_neighbors.py",
        root / "bda_benchmark/experiments/damageformer_srac_transformer_v1/retrieval.py",
    ]
    for path in own + inherited:
        checksum.update(path.relative_to(root).as_posix().encode())
        checksum.update(path.read_bytes())
    return checksum.hexdigest()


class FrozenFrontCapture:
    """Capture the exact tensor consumed by frozen A1 ``clf_2`` during front()."""

    def __init__(self, front):
        self.front = front.eval().requires_grad_(False)
        self.pixel_feature = None
        self._handle = self.front.base.clf_2.register_forward_pre_hook(self._capture)

    def _capture(self, _module, inputs):
        if len(inputs) != 1:
            raise RuntimeError("Unexpected A1 classifier input")
        self.pixel_feature = inputs[0].detach()

    def __call__(self, pre, post):
        self.pixel_feature = None
        with torch.no_grad():
            result = self.front(pre, post)
        if self.pixel_feature is None:
            raise RuntimeError("Failed to capture frozen A1 fused_roi")
        return result, self.pixel_feature

    def close(self):
        self._handle.remove()


def load_artifacts(args, device):
    front, front_payload = load_front(args.checkpoint, device, args.a1_checkpoint)
    cache_root, cache, bank_cpu = read_cache(args.cache, args.checkpoint)
    table, table_manifest = open_table(
        args.neighbors,
        cache_root,
        cache["bank_sha256"],
        args.topk,
        args.retrieval_policy,
        args.exclusion,
    )
    if bool(front_payload["smoke"]) != bool(args.smoke) or bool(cache["smoke"]) != bool(args.smoke):
        raise ValueError("Do not mix smoke and formal artifacts")
    if front_payload["train_sha256"] != digest(args.train_list) or front_payload["val_sha256"] != digest(args.val_list):
        raise ValueError("Front split hash mismatch")
    for record in cache["shards"]:
        if digest(cache_root / record["path"]) != record["sha256"]:
            raise ValueError("Cache shard hash mismatch: " + record["path"])
    bank = device_bank(bank_cpu, device)
    return front, front_payload, cache_root, cache, bank, table, table_manifest


def smoke_train_names(train_names):
    chosen, seen = [], set()
    for name in train_names:
        key = event(name)
        if key not in seen:
            chosen.append(name)
            seen.add(key)
        if len(chosen) == 4:
            break
    return chosen


def read_cached_name(cache_root, cache, index):
    shard = torch.load(cache_root / cache["shards"][index]["path"], map_location="cpu", weights_only=False)
    return str(shard["name"])


def supports_from_table(table, table_manifest, image_index, region_count, bank, device):
    offset = table_manifest["offsets"][image_index]
    if offset["count"] != region_count:
        raise ValueError("Recomputed region count differs from immutable neighbor table")
    selected = torch.from_numpy(
        table[offset["start"] : offset["start"] + region_count].copy()
    ).long().to(device)
    return gather_support(selected, bank)


def upsample_probability(logits, size):
    logits = F.interpolate(logits.float(), size=size, mode="bilinear", align_corners=False)
    return logits.softmax(1)


@torch.no_grad()
def evaluate_all(
    capture,
    decoders,
    data,
    device,
    bank,
    topk,
    retrieval_policy,
    exclusion,
    intervention="normal",
):
    for decoder in decoders.values():
        decoder.eval()
    names = ("a1_pixel", "direct_region", *VARIANTS)
    confusions = {name: torch.zeros(4, 4, dtype=torch.long) for name in names}
    per_event = {name: {} for name in names}
    max_abs_to_a1 = {name: 0.0 for name in names if name != "a1_pixel"}
    image_count = supported = queries = 0
    generator = torch.Generator(device=device).manual_seed(9473)
    for batch in data:
        pre, post, _loc, target, image_names = parse_batch(batch, device)
        result, pixel_feature = capture(pre, post)
        support = distribution = valid = metadata = None
        query = result["tokens"][0]
        event_name = event(image_names[0])
        event_index = bank["event_names"].index(event_name) if event_name in bank["event_names"] else -1
        support, distribution, valid, metadata = retrieve(
            query,
            bank,
            event_index,
            -1,
            topk,
            intervention,
            generator,
            return_metadata=True,
            policy=retrieval_policy,
            exclusion=exclusion,
        )
        supported += int(valid.any(1).sum())
        queries += len(valid)
        probabilities = {
            "a1_pixel": result["old_pixel"].softmax(1),
            "direct_region": result["prob"],
        }
        for variant, decoder in decoders.items():
            kwargs = {}
            if variant == "j2":
                kwargs = dict(support=support, distribution=distribution, valid=valid, metadata=metadata)
            output = decoder(
                pixel_feature,
                result["tokens"],
                result["q"],
                result["ids"],
                result["shape"],
                variant,
                **kwargs,
            )
            probabilities[variant] = upsample_probability(output["logits"], target.shape[-2:])
        for name, probability in probabilities.items():
            if name != "a1_pixel":
                difference = float((probability - probabilities["a1_pixel"]).abs().max())
                max_abs_to_a1[name] = max(max_abs_to_a1[name], difference)
        event_key = event_name
        for name, probability in probabilities.items():
            prediction = probability.argmax(1)
            confusion_update(confusions[name], target, prediction)
            per_event[name].setdefault(event_key, torch.zeros(4, 4, dtype=torch.long))
            confusion_update(per_event[name][event_key], target, prediction)
        image_count += len(pre)
        if image_count % 25 == 0:
            print(f"evaluate joint context {image_count}", flush=True)
    return {
        "metrics": {name: metrics(value) for name, value in confusions.items()},
        "per_event": {
            name: {key: metrics(value) for key, value in events.items()}
            for name, events in per_event.items()
        },
        "image_count": image_count,
        "memory_support_coverage": supported / max(1, queries),
        "max_abs_probability_to_a1": max_abs_to_a1,
        "test_accessed": False,
    }

