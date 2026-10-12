from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from bda_benchmark.experiments.spi_core_ring_adapter_v1 import VARIANTS as CORE_VARIANTS
from bda_benchmark.experiments.spi_core_ring_adapter_v1.model import ParameterMatchedAdapter, adapter_inputs
from bda_benchmark.experiments.spi_core_ring_adapter_v1.regions import build_region_maps
from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import checkpoint_state, file_sha256


def load_frozen_models(base_checkpoint, m4_checkpoint, device, expected_m4_steps):
    base_sha = file_sha256(base_checkpoint)
    source = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    p3 = DamageFormer_S8_MICB_DSMIL().to(device)
    incompatible = p3.load_state_dict(checkpoint_state(source), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")
    p3.eval().requires_grad_(False)

    saved = torch.load(m4_checkpoint, map_location="cpu", weights_only=False)
    if saved.get("base_checkpoint_sha256") != base_sha:
        raise RuntimeError("M4 was not trained from the supplied P3 checkpoint")
    if int(saved.get("optimizer_step", 0)) != int(expected_m4_steps):
        raise RuntimeError(
            f"M4 step mismatch: expected={expected_m4_steps}, observed={saved.get('optimizer_step')}"
        )
    if "trainable_states" in saved:
        if tuple(saved.get("variants", ())) != CORE_VARIANTS:
            raise RuntimeError("unexpected paired M4 checkpoint variants")
        prefix = "m4_core_ring_region."
        state = {key[len(prefix):]: value for key, value in saved["trainable_states"].items() if key.startswith(prefix)}
    elif saved.get("variant") == "m4_core_ring_region":
        state = saved["trainable_state"]
    else:
        raise RuntimeError("checkpoint does not contain m4_core_ring_region")
    hidden = int(saved.get("args", {}).get("hidden_channels", 64))
    m4 = ParameterMatchedAdapter(256, hidden, 4).to(device)
    m4.load_state_dict(state, strict=True)
    m4.eval().requires_grad_(False)
    manifest = {
        "p3_sha256": base_sha,
        "m4_checkpoint": str(Path(m4_checkpoint).resolve()),
        "m4_step": int(saved["optimizer_step"]),
        "m4_seed": int(saved["seed"]),
    }
    return p3, m4, manifest


def frozen_candidate_forward(p3, m4, pre, post, region_args):
    with torch.no_grad():
        loc_output, base_output, auxiliary = p3(pre, post, return_aux=True)
        feature = auxiliary["fused_roi"].detach()
        base_low = p3.clf_2(feature)
        regions = build_region_maps(auxiliary["p_building"], **region_args)
        context, contrast, support = adapter_inputs("m4_core_ring_region", feature, regions)
        candidate_low = base_low + m4(feature, context, contrast, support)
    return loc_output, base_output, base_low, candidate_low, feature, regions


def upsample_logits(logits, output_size):
    return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
