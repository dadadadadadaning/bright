"""RGB-anchored, change-refined soft superpixels for DamageFormer."""

from .model import DamageFormerRGBAnchoredChangeSuperpixelV3
from .variants import VARIANTS, VariantSpec, get_variant

__all__ = ["DamageFormerRGBAnchoredChangeSuperpixelV3", "VariantSpec", "VARIANTS", "get_variant"]
