"""Small pure gates so failed/partial runs cannot be labelled formal evidence."""

from __future__ import annotations


def formal_gate(count, expected, reproduction, immutable, pool_ok, source_ok,
                evidence_ok):
    return bool(count == expected == 450
                and reproduction.get("passed") is True
                and immutable and all(value is True for value in immutable.values())
                and pool_ok is True and source_ok is True and evidence_ok is True)
