from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import checkpoint_state, file_sha256


def load_frozen_p3(checkpoint_path: str | Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("P3 checkpoint must be a mapping")
    model = DamageFormer_S8_MICB_DSMIL()
    incompatible = model.load_state_dict(checkpoint_state(payload), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict P3 load failed: {incompatible}")
    model.to(device).eval().requires_grad_(False)
    return model, {"p3_checkpoint": str(Path(checkpoint_path).resolve()), "p3_sha256": file_sha256(checkpoint_path)}

