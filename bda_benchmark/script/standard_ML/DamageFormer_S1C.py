"""Compatibility import for the former duplicated S1C model file.

The canonical implementation is ``bda_benchmark/model/DamageFormer_S1C.py``.
New code should import it through ``bda_benchmark.model`` or the model registry.
"""

from __future__ import annotations

import sys
from pathlib import Path


BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT))

from model.DamageFormer_S1C import DamageFormer_S1C

__all__ = ["DamageFormer_S1C"]

