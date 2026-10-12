# -*- coding: utf-8 -*-
"""Run external optical--SAR baselines under the common BRIGHT trainer."""
from __future__ import annotations

from bda_benchmark.model.DamageFormer_RGBSAR_ExternalCD import (
    VALID_VARIANTS,
    VARIANTS,
    build_external_cd_bright,
)
from bda_benchmark.script.standard_ML import (
    train_DamageFormer_RGBSAR_Heterogeneous as trainer,
)


def main() -> None:
    # Reuse one audited data/evaluation/checkpoint implementation.  Only the
    # model factory and declared variants change, preventing protocol drift.
    trainer.VALID_VARIANTS = VALID_VARIANTS
    trainer.VARIANTS = VARIANTS
    trainer.build_heterogeneous_damageformer = build_external_cd_bright
    trainer.main()


if __name__ == "__main__":
    main()
