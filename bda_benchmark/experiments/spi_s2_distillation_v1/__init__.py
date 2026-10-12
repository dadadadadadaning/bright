"""Isolated S2-guided structural distillation experiment."""

TRAINABLE_VARIANTS = (
    "k2_semantic_local_control",
    "k3_s2_distilled_local",
    "k4_s2_distilled_gbc",
    "k5_s2_distilled_gbc_variance",
)

ALL_VARIANTS = (
    "k0_frozen_p3",
    "k1_fixed_s2",
) + TRAINABLE_VARIANTS

PROTOCOL = "SPI_S2_DISTILLATION_NO_ORACLE_STANDARD_ML_v1"

