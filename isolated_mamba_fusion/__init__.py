"""Standalone RGB/SAR fusion prototype package.

This package intentionally does not depend on the original BRIGHT/DamageFormer
codebase.  Public modules are re-exported from ``isolated_mamba_fusion.modules``.
"""

from .modules import BaseFusion, DiffMambaFusion, ExistGuidedMambaFusion

__all__ = ["BaseFusion", "DiffMambaFusion", "ExistGuidedMambaFusion"]

