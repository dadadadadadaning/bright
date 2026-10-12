"""Torch-free experiment manifest, also usable by local contract tests."""
from dataclasses import asdict, dataclass
from typing import Dict, Mapping


@dataclass(frozen=True)
class VariantSpec:
    key: str
    title: str
    roi_alpha: float
    use_bags: bool
    use_hierarchy: bool
    claim: str


VARIANTS: Dict[str, VariantSpec] = {
    "a0": VariantSpec("a0", "DamageFormer-compatible pixel path", 1.0, False, False,
                      "Common-parent pixel path without soft ROI or building-bag supervision."),
    "a1": VariantSpec("a1", "Soft building ROI", 0.1, False, False,
                      "A1-A0 isolates the soft building ROI contribution."),
    "a2": VariantSpec("a2", "ROI + bag classification", 0.1, True, False,
                      "A2-A1 tests building-bag MICB/DSMIL supervision as one block."),
    "a3": VariantSpec("a3", "Full P3 hierarchical supervision", 0.1, True, True,
                      "A3-A2 isolates occurrence/severity/guard auxiliary supervision."),
}


def get_variant(name: str) -> VariantSpec:
    key = str(name).lower()
    if key not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {name!r}")
    return VARIANTS[key]


def variant_manifest() -> Dict[str, Mapping[str, object]]:
    return {key: asdict(value) for key, value in VARIANTS.items()}

