"""Small CPU-only evidence gates for the same-pass audit."""

from __future__ import annotations


def final_evidence_gate(image_count, expected_count, reproduction, immutable, pool_ok):
    """Never label a partial or failed reproduction as formal evidence."""
    if expected_count != 450 or image_count != 450:
        return False
    if reproduction.get("passed") is not True:
        return False
    if not immutable or not all(value is True for value in immutable.values()):
        return False
    return pool_ok is True
