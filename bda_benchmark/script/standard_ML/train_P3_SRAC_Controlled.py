# -*- coding: utf-8 -*-
"""Validated training entry for controlled P3/SRAC-inspired experiments."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.model.DamageFormer_P3_SRAC_Controlled import (  # noqa: E402
    P3_SRAC_VARIANTS,
    VALID_P3_SRAC_VARIANTS,
    build_p3_srac_controlled,
)
from bda_benchmark.script.standard_ML import (  # noqa: E402
    train_DamageFormer_RGBSAR_Heterogeneous as trainer,
)
from bda_benchmark.script.standard_ML.p3_srac_controlled_runtime import (  # noqa: E402
    structure_region_supervision,
)


MODEL_HOLDER = {}


def build_from_environment(variant: str, pretrained_rgb: bool = True):
    del pretrained_rgb
    checkpoint = os.environ.get("P3_CKPT", "")
    if not checkpoint:
        raise RuntimeError("P3_CKPT must point to the locked P3 best weights")
    model = build_p3_srac_controlled(
        variant,
        checkpoint,
        native_sar_checkpoint=(
            os.environ.get("H6_A2_CKPT") if variant == "sr1paa" else None
        ),
        cell_size=int(os.environ.get("SRAC_CELL_SIZE", "4")),
        relation_channels=int(os.environ.get("SRAC_RELATION_CHANNELS", "128")),
        memory_capacity=int(os.environ.get("SRAC_MEMORY_PER_CLASS", "512")),
        retrieval_topk=int(os.environ.get("SRAC_TOPK_PER_CLASS", "2")),
    )
    if variant in ("sr1pb", "sr1pib"):
        iterative = variant == "sr1pib"
        source_variable = "SR1PAI_CKPT" if iterative else "SR1PA_CKPT"
        expected_variant = "sr1pai" if iterative else "sr1pa"
        source = os.environ.get(source_variable, "")
        if not source or not Path(source).is_file():
            raise RuntimeError(
                f"{variant.upper()} requires {source_variable} pointing to "
                "the completed partition-pretraining latest_resume.pth"
            )
        checkpoint_value = torch.load(
            source, map_location="cpu", weights_only=False
        )
        if checkpoint_value.get("variant") != expected_variant:
            raise RuntimeError(
                f"{source_variable} variant must be {expected_variant!r}, got "
                f"{checkpoint_value.get('variant')!r}"
            )
        source_state = checkpoint_value.get("model", {})
        partition_state = {
            key[len("partition."):]: value
            for key, value in source_state.items()
            if key.startswith("partition.")
        }
        if not partition_state:
            raise RuntimeError(
                f"{source_variable} contains no partition parameters"
            )
        model.partition.load_state_dict(partition_state, strict=True)
        model.partition_source = {
            "path": str(Path(source).resolve()),
            "step": int(checkpoint_value.get("step", -1)),
        }
    MODEL_HOLDER["model"] = model
    return model


def region_objective(outputs, labels, unchanged_weight=1.0, orthogonality_weight=0.0):
    del unchanged_weight, orthogonality_weight
    return structure_region_supervision(
        outputs,
        labels,
        ignore_index=255,
        model=MODEL_HOLDER.get("model"),
        minimum_building_fraction=float(
            os.environ.get("SRAC_MIN_BUILDING_FRACTION", "0.50")
        ),
        minimum_purity=float(os.environ.get("SRAC_MIN_PURITY", "0.70")),
        reconstruction_weight=float(
            os.environ.get("SRAC_RECONSTRUCTION_WEIGHT", "0.25")
        ),
        building_weight=float(os.environ.get("SRAC_BUILDING_WEIGHT", "1.0")),
        boundary_weight=float(os.environ.get("SRAC_BOUNDARY_WEIGHT", "0.5")),
        entropy_weight=float(os.environ.get("SRAC_ENTROPY_WEIGHT", "0.02")),
        entropy_target=float(os.environ.get("SRAC_ENTROPY_TARGET", "0.60")),
    )


_original_checkpoint_payload = trainer.checkpoint_payload
_original_best_payload = trainer.best_weights_payload


def checkpoint_payload(*args, **kwargs):
    payload = _original_checkpoint_payload(*args, **kwargs)
    payload.update(
        {
            "test_accessed": False,
            "selection_uses_test": False,
            "p3_source": getattr(args[0], "p3_provenance", {}),
            "partition_source": getattr(args[0], "partition_source", {}),
            "native_sar_source": getattr(
                args[0], "native_sar_provenance", {}
            ),
            "paper_claim": "controlled_sr1_sr2_not_official_srac_reproduction",
        }
    )
    return payload


def best_weights_payload(*args, **kwargs):
    payload = _original_best_payload(*args, **kwargs)
    payload.update(
        {
            "test_accessed": False,
            "selection_uses_test": False,
            "p3_source": getattr(args[0], "p3_provenance", {}),
            "partition_source": getattr(args[0], "partition_source", {}),
            "native_sar_source": getattr(
                args[0], "native_sar_provenance", {}
            ),
            "paper_claim": "controlled_sr1_sr2_not_official_srac_reproduction",
        }
    )
    return payload


def main() -> None:
    trainer.VALID_VARIANTS = VALID_P3_SRAC_VARIANTS
    trainer.VARIANTS = P3_SRAC_VARIANTS
    trainer.build_heterogeneous_damageformer = build_from_environment
    trainer.common_unique_regularization = region_objective
    trainer.checkpoint_payload = checkpoint_payload
    trainer.best_weights_payload = best_weights_payload
    print(
        "P3-SRAC controlled contract: SR0 exactly reproduces frozen P3; "
        "SR1 replaces only the final damage decision with a differentiable "
        "RGB-guided region-primary classifier but also uses RGB localisation "
        "for occurrence; SR1F preserves P3 Background/Building probability "
        "and changes only conditional I/D/X; SR1P keeps P3's classifier and "
        "inserts an exactly-safe superpixel projection in feature space; "
        "SR1PA explicitly pretrains a building-boundary partition while its "
        "output remains exact P3; SR1PB loads that partition and only then "
        "opens the safe feature projection; SR1PAI/SR1PIB repeat this "
        "controlled pair with three differentiable assignment/seed-update "
        "iterations; "
        "SR1PAA keeps exact P3 predictions and trains only an eight-neighbour "
        "damage-affinity head from frozen P3 RGB geometry plus frozen H6 "
        "native-SAR Stage2/3 features; "
        "SR2 adds a class-balanced "
        "training-memory retrieval block. P3 localisation remains frozen. "
        "No validation/test feature can update memory.",
        flush=True,
    )
    trainer.main()


if __name__ == "__main__":
    main()
