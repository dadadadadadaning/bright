"""Reliable cross-modal building blocks for the S8-R ablation family."""

from .blocks import (
    BoundedSpatialReliabilityGate,
    ChangeRelationPyramid,
    ConservativeStage3Rectifier,
    LocalStructuralRelationDifference,
    PixelDamageHierarchyHead,
    ResidualSARAdapter,
    build_hierarchical_targets,
    reliable_auxiliary_losses,
)

__all__ = [
    "BoundedSpatialReliabilityGate",
    "ChangeRelationPyramid",
    "ConservativeStage3Rectifier",
    "LocalStructuralRelationDifference",
    "PixelDamageHierarchyHead",
    "ResidualSARAdapter",
    "build_hierarchical_targets",
    "reliable_auxiliary_losses",
]
