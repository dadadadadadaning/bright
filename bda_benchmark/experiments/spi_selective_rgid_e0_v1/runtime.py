from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from bda_benchmark.experiments.spi_rgid_semantic_v1.method import (
    SplitRiskConfig,
    score_split_risk,
)
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import file_sha256


def load_rgid(checkpoint_path: str, provenance_path: str, p3_path: str, train_list: str, device):
    root = Path(__file__).resolve().parents[1] / "rgid_s9ic_clean"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from rgid.geometry import RGIDDecoder

    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    allowed_protocols = {
        "RGID_PSEUDO_ONLY_TRAIN_v1",
        "RGID_STRUCTURED_PLUGIN_TRAIN_v1",
    }
    if provenance.get("protocol") not in allowed_protocols:
        raise RuntimeError("RGID provenance violation: protocol")
    required = {
        "official_instances_accessed": False,
        "standard_ml_val_accessed": False,
        "original_standard_test_accessed": False,
        "calibration_split": "train_only",
    }
    for key, expected in required.items():
        if provenance.get(key) != expected:
            raise RuntimeError(f"RGID provenance violation: {key}")
    if provenance.get("p3_sha256") != file_sha256(p3_path):
        raise RuntimeError("RGID/P3 fingerprint mismatch")
    if provenance.get("train_list_sha256") != file_sha256(train_list):
        raise RuntimeError("RGID was not trained exclusively on the declared meta_train list")
    if provenance["protocol"] == "RGID_STRUCTURED_PLUGIN_TRAIN_v1":
        from bda_benchmark.model.structured_plugins.rgid_head import load_rgid_head

        head, config, payload = load_rgid_head(checkpoint_path, device=device)
        if payload.get("head_type") != provenance.get("head_type"):
            raise RuntimeError("RGID head type/provenance mismatch")
    else:
        from rgid.evaluation import load_head_checkpoint

        head, config, payload = load_head_checkpoint(checkpoint_path, device=device)
    calibration = payload.get("boundary_calibration")
    if not isinstance(calibration, Mapping) or "boundary_gate_threshold" not in calibration:
        raise RuntimeError("RGID checkpoint has no train-only boundary calibration")
    return (
        head.eval().requires_grad_(False),
        RGIDDecoder(config),
        float(calibration["boundary_gate_threshold"]),
        provenance,
    )


def decode_and_score(
    *, head, decoder, boundary_threshold: float, fused_roi: torch.Tensor,
    probabilities: torch.Tensor, parent_labels: torch.Tensor, device: torch.device,
):
    support = parent_labels > 0
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        geometry = head(fused_roi.float())
    center_logits = geometry["center_logits"][0, 0]
    boundary_logits = geometry["boundary_logits"][0, 0]
    decoded = decoder.decode(
        center_logits, boundary_logits, support, mode="g3", inputs_are_logits=True,
        boundary_gate_threshold=boundary_threshold,
    )
    children = np.asarray(decoded.instance_labels, dtype=np.int32)
    parents = parent_labels.detach().cpu().numpy().astype(np.int32, copy=False)
    if not np.array_equal(children > 0, parents > 0):
        raise RuntimeError("RGID changed P3/LocCC building support")
    center = center_logits.detach().float().sigmoid().cpu()
    boundary = boundary_logits.detach().float().sigmoid().cpu()
    config = SplitRiskConfig(score_threshold=0.0)
    evidence = {}
    for parent_id in (int(x) for x in np.unique(parents) if x > 0):
        mask = parents == parent_id
        child_count = len(np.unique(children[mask]))
        evidence[parent_id] = score_split_risk(
            parent_id=parent_id, parent_mask=mask, probabilities=probabilities,
            center_probability=center, boundary_probability=boundary,
            child_count=child_count, config=config,
        )
    return children, evidence, decoded.stats
