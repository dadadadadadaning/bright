"""Explicit snapshot protocol, separate from the completed-parent protocol."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from bda_benchmark.experiments.damageformer_srac_cause_audit_v1.support import (
    load_snapshot, validate_snapshot_payload,
)
from bda_benchmark.experiments.damageformer_srac_joint123_v1 import common as original_joint
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import (
    device_bank, digest, event, lists, loader, new_output, parse_batch,
    region_targets, save_json, save_pt, seed_everything,
)

MEMORY_FORMAT = "SRAC_SNAPSHOT_MEMORY_V1"
METRIC_FORMAT = "SRAC_SNAPSHOT_METRIC_V1"
JOINT_FORMAT = "SRAC_SNAPSHOT_JOINT123_V1"


def source_hash():
    checksum = hashlib.sha256(original_joint.source_hash().encode())
    root = Path(__file__).parent
    files = sorted(root.glob("*.py")) + sorted(root.glob("*.sh"))
    files += [root.parent / "damageformer_srac_cause_audit_v1" / name
              for name in ("support.py", "snapshot.py")]
    for path in files:
        checksum.update(path.relative_to(root.parent).as_posix().encode())
        checksum.update(path.read_bytes())
    return checksum.hexdigest()


def validate_front_payload(payload, train_sha256, val_sha256):
    validate_snapshot_payload(payload)
    if payload.get("smoke") is not False:
        raise ValueError("Snapshot comparison requires a non-smoke parent checkpoint")
    if payload.get("train_sha256") != train_sha256 or payload.get("val_sha256") != val_sha256:
        raise ValueError("Snapshot split hashes differ")
    # Deliberately do NOT mutate run_complete, declared budget, or checkpoint step.
    return payload


def snapshot_metadata(path):
    path = Path(path)
    manifest_path = path.parent / "snapshot.json"
    if not manifest_path.is_file():
        raise ValueError("Use an immutable front_snapshot.pth with adjacent snapshot.json, not live best.pth")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("format") != "SRAC_CAUSE_DIAGNOSTIC_SNAPSHOT_V1"
            or manifest.get("complete") is not True
            or manifest.get("diagnostic_only") is not True
            or manifest.get("snapshot_file") != path.name
            or manifest.get("snapshot_sha256") != digest(path)):
        raise ValueError("Invalid snapshot manifest / SHA256")
    if manifest.get("smoke") is not False or manifest.get("test_accessed") is not False:
        raise ValueError("Use a formal Train/Val-only snapshot")
    return manifest


def load_front(path, device, a1_override=None):
    metadata = snapshot_metadata(path)
    if a1_override is None:
        raise ValueError("Explicit original A1 checkpoint is required")
    model, payload = load_snapshot(path, a1_override, device)
    validate_front_payload(payload, payload.get("train_sha256"), payload.get("val_sha256"))
    for key, expected in (("step", payload["step"]), ("declared_steps", payload["arguments"]["steps"]),
                          ("run_complete", bool(payload.get("run_complete", False))),
                          ("a1_sha256", payload["a1_sha256"]), ("source_hash", payload["source_hash"])):
        if metadata.get(key) != expected:
            raise ValueError("Snapshot metadata conflicts with checkpoint: " + key)
    return model, payload


def read_cache(path, checkpoint):
    root = Path(path)
    snapshot_metadata(checkpoint)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != MEMORY_FORMAT or manifest.get("snapshot_protocol") is not True
            or manifest.get("complete") is not True or manifest.get("split") != "train"
            or manifest.get("test_accessed") is not False
            or manifest.get("source_weights_and_buffers_unchanged") is not True):
        raise ValueError("Require completed, immutable Train-only snapshot bank")
    if manifest.get("checkpoint_sha256") != digest(checkpoint) or manifest.get("source_hash") != source_hash():
        raise ValueError("Stale or incompatible snapshot bank")
    if type(manifest.get("smoke")) is not bool:
        raise ValueError("Missing explicit bank smoke flag")
    expected_count = 4 if manifest["smoke"] else 2918
    if manifest.get("image_count") != expected_count or len(manifest.get("shards", [])) != expected_count:
        raise ValueError("Incomplete snapshot bank shards")
    metadata = snapshot_metadata(checkpoint)
    if manifest.get("parent_checkpoint_step") != metadata["step"]:
        raise ValueError("Bank belongs to a different snapshot step")
    if digest(root / "bank.pt") != manifest.get("bank_sha256"):
        raise ValueError("Bank checksum mismatch")
    bank = torch.load(root / "bank.pt", map_location="cpu", weights_only=False)
    if bank.get("checkpoint_sha256") != manifest["checkpoint_sha256"]:
        raise ValueError("Bank payload / manifest checkpoint mismatch")
    tokens, distribution = bank["tokens"], bank["distribution"]
    if tokens.ndim != 2 or len(tokens) == 0 or distribution.shape != (len(tokens), 4):
        raise ValueError("Invalid Memory dimensions")
    if not torch.isfinite(tokens).all() or not torch.isfinite(distribution).all() or (distribution < 0).any():
        raise ValueError("Invalid Memory values")
    if not torch.allclose(distribution.sum(-1), torch.ones(len(tokens)), atol=1e-5, rtol=1e-5):
        raise ValueError("Memory labels are not normalized")
    return root, manifest, bank


def load_metric(path, cache_manifest, front_ckpt, device):
    from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import RetrievalProjector
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != METRIC_FORMAT or payload.get("complete") is not True:
        raise ValueError("Require a completed snapshot retrieval metric")
    steps = payload.get("arguments", {}).get("steps", 0)
    if (steps < 1 or payload.get("step") != steps or payload.get("additional_optimizer_updates") != steps
            or (not payload.get("smoke") and steps != 2000)):
        raise ValueError("Metric fitting budget incomplete")
    if payload.get("train_only") is not True or payload.get("test_accessed") is not False:
        raise ValueError("Metric must be Train-only")
    if (payload.get("source_hash") != source_hash() or payload.get("front_sha256") != digest(front_ckpt)
            or payload.get("bank_sha256") != cache_manifest["bank_sha256"]
            or bool(payload.get("smoke")) != bool(cache_manifest["smoke"])):
        raise ValueError("Metric provenance mismatch")
    with torch.random.fork_rng(devices=[]):
        projector = RetrievalProjector(payload["dim"]).to(device)
    projector.load_state_dict(payload["model"], strict=True)
    if any(not torch.isfinite(value).all() for value in projector.state_dict().values()):
        raise ValueError("Metric contains nonfinite values")
    return projector.eval().requires_grad_(False), payload


def reproduction_gate(result, payload, probe=False):
    if probe:
        return dict(passed=None, reason="Two-image probe is not full-Val reproduction")
    checks = {}
    expected = {"a1_pixel": payload["load_report"]["expected_valid_miou"],
                "direct_region": payload["valid_miou"]}
    for name, value in expected.items():
        actual = result["metrics"][name]["valid_miou"]
        delta = abs(actual - value)
        checks[name] = dict(expected=value, actual=actual, absolute_error=delta,
                            passed=math.isfinite(delta) and delta <= 1e-5)
    return dict(passed=result["image_count"] == 450 and all(row["passed"] for row in checks.values()),
                tolerance=1e-5, checks=checks)
