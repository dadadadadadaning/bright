"""Diagnostic-only loading; the parent's completed-training gate stays intact."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch

from bda_benchmark.experiments.damageformer_srac_fourstage_v2.a1_init import KNOWN_A1
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import (
    FRONT_FORMAT,
    digest,
    load_a1_strict,
    source_hash,
)

CONFIG_KEYS = {
    "dim", "decoder_width", "stride", "iterations", "temperature",
    "spatial_weight", "building_weight", "region_context",
}


def validate_snapshot_payload(payload, *, expected_source_hash=None):
    """Validate provenance without declaring an intermediate front complete.

    This intentionally does not call or modify ``parent.common.load_front``.
    Incomplete checkpoints remain diagnostic inputs and cannot satisfy that
    parent's formal downstream training/evaluation gate.
    """
    if not isinstance(payload, Mapping) or payload.get("format") != FRONT_FORMAT:
        raise ValueError("Wrong front checkpoint format")
    expected = source_hash() if expected_source_hash is None else expected_source_hash
    if payload.get("source_hash") != expected:
        raise ValueError("Source changed since front training; use its archived code")
    if payload.get("a1_sha256") != KNOWN_A1:
        raise ValueError("Snapshot must reference the exact audited A1 SHA256")
    if payload.get("test_accessed") is not False:
        raise ValueError("Snapshot must explicitly declare test_accessed=False")
    score = payload.get("valid_miou")
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Missing or invalid checkpoint valid_miou")
    arguments = payload.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("Missing front training arguments")
    steps, step = arguments.get("steps"), payload.get("step")
    if type(steps) is not int or type(step) is not int or not 1 <= step <= steps:
        raise ValueError("Invalid front step / declared training budget")
    if type(payload.get("smoke")) is not bool:
        raise ValueError("Missing explicit front smoke status")
    if arguments.get("smoke") != payload["smoke"]:
        raise ValueError("Conflicting front smoke status")
    if not payload["smoke"] and steps != 50000:
        raise ValueError("Formal parent front requires the declared 50000-step budget")
    if "run_complete" in payload and type(payload["run_complete"]) is not bool:
        raise ValueError("Invalid run_complete flag")
    if payload.get("run_complete") and payload.get("completed_steps") != steps:
        raise ValueError("Completed front has inconsistent completed_steps")
    config = payload.get("config")
    if not isinstance(config, Mapping) or set(config) != CONFIG_KEYS:
        raise ValueError("Front config keys do not exactly match the parent model")
    for key in ("dim", "decoder_width", "stride", "iterations"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid front config {key}")
    for key in ("temperature", "spatial_weight", "building_weight"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Invalid front config {key}")
        if value < 0 or (key == "temperature" and value == 0):
            raise ValueError(f"Invalid front config {key}")
    if type(config["region_context"]) is not bool:
        raise ValueError("region_context must be a boolean")
    if config["region_context"] and config["dim"] % 4:
        raise ValueError("Region context dimension must be divisible by four")
    for key, value in config.items():
        if key in arguments and arguments[key] != value:
            raise ValueError(f"Front config and training arguments disagree: {key}")
    adapter = payload.get("adapter")
    if not isinstance(adapter, Mapping) or not adapter:
        raise ValueError("Missing adapter-only state")
    for name, value in adapter.items():
        if not isinstance(name, str) or name.startswith("base.") or not torch.is_tensor(value):
            raise ValueError("Invalid adapter-only state entry")
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite adapter tensor: {name}")
    return payload


def verify_splits(payload, train_list, val_list):
    """Require the exact recorded split files, not only their cardinalities."""
    for key, path in (("train_sha256", train_list), ("val_sha256", val_list)):
        expected = payload.get(key)
        if not isinstance(expected, str) or digest(path) != expected:
            raise ValueError("Split differs from parent checkpoint: " + key)


def diagnostic_source_hash():
    checksum = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.glob("*.py")) + [root / "run.sh"]:
        checksum.update(path.name.encode("utf-8"))
        checksum.update(path.read_bytes())
    return checksum.hexdigest()


def load_snapshot(checkpoint, a1_checkpoint, device):
    """Load a trusted local diagnostic snapshot and freeze every module.

    PyTorch checkpoints are pickle files: only use checkpoints from the trusted
    parent run. No dataset, validation image, or Test split is opened here.
    """
    checkpoint = Path(checkpoint)
    manifest_path = checkpoint.parent / "snapshot.json"
    # Hash and unpickle the same open file, protecting against atomic replacement.
    with checkpoint.open("rb") as handle:
        hasher = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
        checksum = hasher.hexdigest()
        handle.seek(0)
        payload = torch.load(handle, map_location="cpu", weights_only=False)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("snapshot_file") != checkpoint.name or manifest.get("snapshot_sha256") != checksum:
            raise ValueError("Diagnostic snapshot manifest / SHA256 mismatch")
        if manifest.get("complete") is not True or manifest.get("diagnostic_only") is not True:
            raise ValueError("Invalid diagnostic snapshot manifest")
    validate_snapshot_payload(payload)
    a1_path = Path(a1_checkpoint)
    if not a1_path.is_file() or digest(a1_path) != payload["a1_sha256"]:
        raise ValueError("The exact audited A1 file required by this snapshot is unavailable")
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import SRACTransformerFront

    model = SRACTransformerFront(**payload["config"])
    if model.config != payload["config"]:
        raise ValueError("Instantiated model configuration differs from checkpoint")
    expected = model.adapter_state_dict()
    if set(expected) != set(payload["adapter"]):
        raise ValueError("Adapter checkpoint keys do not exactly match the model")
    for name, value in payload["adapter"].items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f"Adapter tensor shape / dtype mismatch: {name}")
    load_a1_strict(model, a1_path)
    model.load_adapter_state_dict(payload["adapter"])
    model.to(device).requires_grad_(False).eval()
    if any(parameter.requires_grad for parameter in model.parameters()) or any(module.training for module in model.modules()):
        raise RuntimeError("Diagnostic model freeze / evaluation contract failed")
    return model, payload


def module_fingerprint(module):
    """Hash every parameter and buffer, including nonpersistent buffers, on CPU."""
    checksum = hashlib.sha256()
    for kind, tensors in (("parameter", module.named_parameters()), ("buffer", module.named_buffers())):
        for name, value in sorted(tensors):
            tensor = value.detach().to(device="cpu").contiguous()
            metadata = json.dumps([kind, name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":"))
            checksum.update(metadata.encode("utf-8"))
            checksum.update(b"\0")
            # Byte view supports bfloat16 and scalar buffers without numpy casts.
            checksum.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return checksum.hexdigest()
