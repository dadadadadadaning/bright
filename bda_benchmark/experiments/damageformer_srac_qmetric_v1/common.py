"""Isolated artifact adapters; parent sources and checkpoints are never rewritten."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from bda_benchmark.experiments.damageformer_srac_transformer_v1 import common as parent
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import (
    confusion_update, device_bank, digest, event, front_payload, lists,
    load_a1_strict, loader, new_output, parse_batch, read_cache,
    region_targets, save_json, save_pt, seed_everything,
)

METRIC_FORMAT = "SRAC_QMETRIC_RETRIEVAL_V1"


def source_hash():
    checksum = hashlib.sha256(parent.source_hash().encode())
    root = Path(__file__).resolve().parents[3]
    files = sorted(Path(__file__).parent.glob("*.py"))
    files += [root / "bda_benchmark/experiments/damageformer_srac_joint_context_v1" / name
              for name in ("model.py", "common.py", "train.py")]
    for path in files:
        checksum.update(path.relative_to(root).as_posix().encode())
        checksum.update(path.read_bytes())
    return checksum.hexdigest()


def load_front(path, device, a1_override=None):
    from .model import AnnealedFront, FRONT_FORMAT

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") == parent.FRONT_FORMAT:
        # Retain ALL parent provenance/completion checks; do not bypass a gate.
        return parent.load_front(path, device, a1_override)
    if payload.get("format") != FRONT_FORMAT:
        raise ValueError("Unsupported front checkpoint format")
    if not payload.get("run_complete") or payload.get("completed_steps") != payload["arguments"]["steps"]:
        raise ValueError("Q experiment has not completed its declared budget")
    if payload.get("source_hash") != source_hash():
        raise ValueError("Source mismatch; use the archived matching experiment code")
    a1_path = Path(a1_override or payload["a1_checkpoint"])
    if not a1_path.is_file() or digest(a1_path) != payload["a1_sha256"]:
        raise ValueError("Exact frozen A1 checkpoint is required")
    model = AnnealedFront(**payload["config"]).to(device)
    load_a1_strict(model, a1_path)
    model.load_adapter_state_dict(payload["adapter"])
    # A best checkpoint can predate the end of the ramp. Never replace its beta
    # with the final training beta or a constructor default.
    if abs(model.current_beta - float(payload["q_beta_current"])) > 1e-6:
        raise ValueError("Checkpoint Q beta was not restored")
    model.base.eval().requires_grad_(False)
    return model, payload


def load_metric(path, cache_manifest, front_ckpt, device):
    from .metric import RetrievalProjector

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != METRIC_FORMAT or not payload.get("complete"):
        raise ValueError("A completed retrieval metric is required")
    declared_steps = payload.get("arguments", {}).get("steps", 0)
    if (declared_steps < 1 or payload.get("step") != declared_steps
            or payload.get("additional_optimizer_updates") != declared_steps
            or (not payload.get("smoke") and declared_steps != 2000)):
        raise ValueError("Metric did not complete its declared fitting budget")
    if payload.get("train_only") is not True or payload.get("test_accessed") is not False:
        raise ValueError("Retrieval metric must be fitted only on Train")
    if payload.get("source_hash") != source_hash():
        raise ValueError("Metric source hash mismatch")
    if payload["front_sha256"] != digest(front_ckpt):
        raise ValueError("Metric belongs to a different frozen front")
    if payload["bank_sha256"] != cache_manifest["bank_sha256"]:
        raise ValueError("Metric belongs to a different Memory bank")
    if bool(payload["smoke"]) != bool(cache_manifest["smoke"]):
        raise ValueError("Cannot mix smoke and formal artifacts")
    # Linear's constructor consumes CPU RNG before identity initialization.
    # Loading a frozen metric must not change the baseline decoder seed stream.
    with torch.random.fork_rng(devices=[]):
        projector = RetrievalProjector(payload["dim"]).to(device)
    projector.load_state_dict(payload["model"], strict=True)
    return projector.eval().requires_grad_(False), payload


def evaluate_front(model, data, device, **kwargs):
    """Parent predictions/metrics plus descriptive Q diagnostics (no GT in Q)."""
    names = ("q_top1_mean", "q_entropy_valid_mean", "q_valid_candidates_mean",
             "q_soft_occupied_mean", "q_hard_occupied_mean", "q_region_slots",
             "q_logit_scale_exp", "q_similarity_coefficient",
             "adapter_delta_norm", "adapter_base_norm", "adapter_delta_base_ratio")
    sums, counts = {}, {}

    def collect(_module, _inputs, output):
        for name in names:
            if name in output:
                sums[name] = sums.get(name, 0.0) + float(output[name])
                counts[name] = counts.get(name, 0) + 1

    handle = model.register_forward_hook(collect)
    try:
        result = parent.evaluate_front(model, data, device, **kwargs)
    finally:
        handle.remove()
    result["q_diagnostics_image_mean"] = {name: sums[name] / counts[name] for name in sums}
    return result
