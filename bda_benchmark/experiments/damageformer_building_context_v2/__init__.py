"""DamageFormer Building Context V2.

Reuses the existing A1/C1-C9 experiment without modifying it.
"""

from .variants import VARIANTS, get_variant
from .model import DamageFormerBuildingContextV2

__all__ = ["VARIANTS", "get_variant", "DamageFormerBuildingContextV2"]
