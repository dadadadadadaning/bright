"""Shape-preserving plug-ins for controlled BRIGHT ablations.

These blocks are intentionally not registered as a complete segmentation model.
They must be enabled one at a time by an experiment protocol.
"""

from .blocks import (
    AffectedOccurrenceResidual,
    CrossScaleBoundaryFusion,
    LocalCrossModalMatcher,
    RegionRelationBlock,
    StripSpatialBlock,
)
from .rgid_head import StripRGIDHead, build_rgid_head, load_rgid_head

__all__ = [
    "AffectedOccurrenceResidual",
    "CrossScaleBoundaryFusion",
    "LocalCrossModalMatcher",
    "RegionRelationBlock",
    "StripRGIDHead",
    "StripSpatialBlock",
    "build_rgid_head",
    "load_rgid_head",
]
