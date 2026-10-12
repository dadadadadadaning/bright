# -*- coding: utf-8 -*-
"""S9-Id Center+Offset head for explicit building-pixel ownership."""

from __future__ import annotations

from .DamageFormer_S9Ic_ReliableGeometry import (
    FEATURE_PRIORITY,
    ReliableGeometryHead,
    select_geometry_feature,
)


class CenterOffsetGeometryHead(ReliableGeometryHead):
    """Unique Gaussian center + boundary + (dx,dy) offset prediction."""

    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            predict_offset=True,
        )


__all__ = [
    "CenterOffsetGeometryHead",
    "FEATURE_PRIORITY",
    "select_geometry_feature",
]

