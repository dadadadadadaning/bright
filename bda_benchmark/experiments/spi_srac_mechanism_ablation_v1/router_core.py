from __future__ import annotations

from bda_benchmark.experiments.spi_changemamba_selective_memory_v1.verifier_core import *  # noqa: F401,F403
from bda_benchmark.experiments.spi_changemamba_selective_memory_v1.verifier_core import (
    DIRECT_FEATURES,
    MEMORY_FEATURES as _OLD_MEMORY_FEATURES,
)

# Add the event-risk features that distinguish the proposed safe router from
# a plain retrieval classifier.  V1 remains the no-Memory control.
VARIANT_FEATURES = {
    "v1_direct": DIRECT_FEATURES,
    "v2_rag": _OLD_MEMORY_FEATURES + ("source_event_count", "source_event_entropy"),
}


def filter_rows(rows, variant):
    if variant not in VARIANT_FEATURES:
        raise ValueError(f"unknown router variant {variant}")
    if variant == "v1_direct":
        return [row for row in rows if row["source"] == "direct"]
    return list(rows)
