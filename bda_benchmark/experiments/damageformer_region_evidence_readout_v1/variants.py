from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class ReadoutSpec:
    name: str
    title: str
    uses_region: bool
    uses_gate: bool
    preserves_bg_destroyed_probability: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


VARIANTS: Dict[str, ReadoutSpec] = {
    "w0_full_residual": ReadoutSpec(
        "w0_full_residual", "Unconstrained four-class region residual negative control", True, False, False
    ),
    "w1_id_calibration": ReadoutSpec(
        "w1_id_calibration", "Global affine calibration of Intact/Damaged logits", False, False, False
    ),
    "w2_id_logit": ReadoutSpec(
        "w2_id_logit", "Region-conditioned Intact/Damaged antisymmetric logit residual", True, True, False
    ),
    "w3_id_mass": ReadoutSpec(
        "w3_id_mass", "Region-conditioned mass-preserving Intact/Damaged mixer", True, True, True
    ),
    "w5_logit4_kd": ReadoutSpec(
        "w5_logit4_kd", "Region-conditioned four-class logit adapter with base preservation", True, False, False
    ),
}


def get_variant(name: str) -> ReadoutSpec:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown readout {name!r}; choose from {sorted(VARIANTS)}") from exc
