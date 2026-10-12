#!/usr/bin/env python3
"""Fail fast when the server contains a mixed pseudo-instance code stack."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path


def _project_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def verify(project_root: Path) -> dict[str, object]:
    project_root = project_root.resolve()
    manifest_path = Path(__file__).with_name("component_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = [
        row["path"]
        for row in manifest["components"]
        if not (project_root / row["path"]).is_file()
    ]
    if missing:
        raise RuntimeError(f"missing main-line components: {missing}")

    from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import (
        DamageFormer_S8_MICB_DSMIL,
        connected_components_4,
    )
    from bda_benchmark.experiments.spi_selective_context_v1.core import VARIANTS
    from bda_benchmark.script.standard_ML.spi_region_feedback.core import (
        predicted_building_regions,
    )

    cc_signature = inspect.signature(connected_components_4)
    if "minimum_area" not in cc_signature.parameters:
        raise RuntimeError(
            "mixed server versions: connected_components_4 must accept "
            "minimum_area; synchronize model/DamageFormer_S8_MICB_DSMIL.py"
        )
    region_signature = inspect.signature(predicted_building_regions)
    if "minimum_area" not in region_signature.parameters:
        raise RuntimeError("predicted_building_regions lost the minimum_area contract")
    model_signature = inspect.signature(DamageFormer_S8_MICB_DSMIL.__init__)
    residual_default = model_signature.parameters["mil_residual_scale"].default
    if float(residual_default) != 0.0:
        raise RuntimeError("locked P3 must default to mil_residual_scale=0")

    expected_variants = (
        "s0_raw_p3",
        "s1_loccc_hard_a6",
        "s2_loccc_soft_context",
        "s3_loccc_uncertain_interior",
        "s4_local_uncertain_control",
    )
    if tuple(VARIANTS) != expected_variants:
        raise RuntimeError(f"selective-context variants changed: {VARIANTS}")

    source = Path(inspect.getsourcefile(connected_components_4) or "").resolve()
    expected_source = (project_root / "model/DamageFormer_S8_MICB_DSMIL.py").resolve()
    if source != expected_source:
        raise RuntimeError(
            f"wrong module imported: expected={expected_source}, observed={source}"
        )
    return {
        "project_root": str(project_root),
        "connected_components_source": str(source),
        "connected_components_signature": str(cc_signature),
        "predicted_regions_signature": str(region_signature),
        "p3_mil_residual_scale_default": float(residual_default),
        "selective_variants": list(VARIANTS),
        "component_count": len(manifest["components"]),
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=_project_root_from_file())
    args = parser.parse_args()
    print(json.dumps(verify(args.project_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

