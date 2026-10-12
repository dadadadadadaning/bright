"""Frozen Stage1 region evidence with controlled decision readouts."""

from .model import DamageFormerRegionEvidenceReadout
from .variants import VARIANTS, ReadoutSpec, get_variant

__all__ = ["DamageFormerRegionEvidenceReadout", "VARIANTS", "ReadoutSpec", "get_variant"]

