from __future__ import annotations
from dataclasses import dataclass

EXPECTED_STAGE1_SHA = "0b76a887a16d4c1c5371cba1570a3a6c641435ef2c97494363323b73aa4239f7"

# Building-level cache labels
BLD_UNKNOWN = -1
BLD_INTACT = 0
BLD_DAMAGED = 1
BLD_DESTROYED = 2

# Pixel-level 4x4 confusion labels
PIX_BG = 0
PIX_INTACT = 1
PIX_DAMAGED = 2
PIX_DESTROYED = 3

@dataclass(frozen=True)
class Protocol:
    seed: int = 3407
    coverage: float = 0.02
    fit_ratio: float = 0.80
    min_purity: float = 0.70
    coarse_k_per_class: int = 50
    fine_k_per_class: int = 20
    local_global_weight: float = 0.35
    region_temperature: float = 0.25
    region_beta_di: float = 0.50
    region_gamma_entropy: float = 0.25
    relation_high_quantile: float = 0.90
    utility_steps: int = 2000
    utility_lr: float = 1e-3

DEFAULT_PROTOCOL = Protocol()
