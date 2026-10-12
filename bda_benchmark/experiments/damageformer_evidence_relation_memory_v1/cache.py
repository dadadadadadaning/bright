from __future__ import annotations
import hashlib
from typing import Dict, Iterable, Tuple
import torch

from .config import EXPECTED_STAGE1_SHA

REQUIRED = [
    "features", "labels", "label_purity", "names", "events",
    "pred_building_ids", "base_probs", "base_pred", "uncertainty",
    "peak_mean_delta", "region_count", "region_offsets",
    "region_tokens", "region_critical", "region_logits",
    "base_pixel_cm", "building_pixel_cm", "test_accessed",
]

def _forbid_test_path(path: str) -> None:
    if "test" in path.lower():
        raise RuntimeError(f"Development protocol forbids Test access: {path}")

def load_stage1_cache(path: str, expected_split: str | None = None, require_sha: bool = True) -> dict:
    _forbid_test_path(path)
    x = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    missing = [k for k in REQUIRED if k not in x]
    if missing:
        raise KeyError(f"Stage1 cache missing keys: {missing}")
    if bool(x.get("test_accessed", False)):
        raise RuntimeError("Refusing cache with test_accessed=True")
    if expected_split is not None and x.get("split") not in {expected_split, "validation" if expected_split=="val" else expected_split}:
        raise RuntimeError(f"Expected split={expected_split}, got {x.get('split')}")
    if require_sha:
        sha = x.get("source_checkpoint_sha256")
        if sha != EXPECTED_STAGE1_SHA:
            raise RuntimeError(f"Stage1 identity mismatch: expected {EXPECTED_STAGE1_SHA}, got {sha}")

    n = len(x["labels"])
    off = x["region_offsets"]
    if len(off) != n + 1:
        raise RuntimeError("region_offsets length != buildings+1")
    if int(off[-1]) != len(x["region_tokens"]):
        raise RuntimeError("region_offsets tail != total regions")
    if not torch.equal(off[1:] - off[:-1], x["region_count"]):
        raise RuntimeError("region_offsets/count integrity failed")
    if x["features"].shape != (n, 512):
        raise RuntimeError(f"Expected features [N,512], got {tuple(x['features'].shape)}")
    if x["region_tokens"].shape[1] != 128:
        raise RuntimeError("Expected 128D Region tokens")
    if x["region_logits"].shape[1] != 3:
        raise RuntimeError("Expected 3-way region_logits [I,D,X]")
    return x

def event_mapping(train_events, other_events=()):
    names = sorted(set(train_events) | set(other_events))
    mapping = {e:i for i,e in enumerate(names)}
    return mapping

def encode_events(events, mapping):
    return torch.tensor([mapping[e] for e in events], dtype=torch.long)

def stable_image_split(cache: dict, fit_ratio: float = 0.80, seed: int = 3407) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic event-stratified split by image/name, never by individual building."""
    import random
    by_event: Dict[str, list[str]] = {}
    for name, event in zip(cache["names"], cache["events"]):
        by_event.setdefault(event, [])
        if name not in by_event[event]:
            by_event[event].append(name)

    fit_names, cal_names = set(), set()
    for event in sorted(by_event):
        names = sorted(by_event[event])
        rng = random.Random(seed + int(hashlib.sha256(event.encode()).hexdigest()[:8], 16))
        rng.shuffle(names)
        if len(names) <= 1:
            cut = len(names)
        else:
            cut = max(1, min(len(names)-1, int(round(len(names)*fit_ratio))))
        fit_names.update(names[:cut])
        cal_names.update(names[cut:])

    fit = torch.tensor([n in fit_names for n in cache["names"]], dtype=torch.bool)
    cal = torch.tensor([n in cal_names for n in cache["names"]], dtype=torch.bool)
    if torch.any(fit & cal):
        raise RuntimeError("Fit/Cal overlap detected")
    if not torch.all(fit | cal):
        raise RuntimeError("Some buildings were not assigned to Fit or Cal")
    return fit, cal

def global_region_mean(cache: dict) -> torch.Tensor:
    # feature_definition = concat(region_mean, region_std, top1, top2)
    return cache["features"][:, :128].float()

def c1_features(cache: dict) -> torch.Tensor:
    u = cache["uncertainty"].float().unsqueeze(1)
    return torch.cat([
        cache["base_probs"].float(),
        u,
        global_region_mean(cache),
    ], dim=1)

def candidate_mask(cache: dict) -> torch.Tensor:
    # Building-level cache label order: base_pred 0/1/2 = I/D/X
    return cache["base_pred"].long() == 0

def clean_label_mask(cache: dict, min_purity: float = 0.70) -> torch.Tensor:
    return (cache["labels"].long() >= 0) & (cache["label_purity"].float() >= min_purity)

def building_area(cache: dict) -> torch.Tensor:
    return cache["building_pixel_cm"].long().sum(dim=(1,2))
