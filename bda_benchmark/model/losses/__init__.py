"""Losses used by the controlled multimodal damage experiments."""

from .severity_distance_loss import (
    expected_severity_distance,
    masked_severity_cross_entropy,
)

__all__ = ["expected_severity_distance", "masked_severity_cross_entropy"]
