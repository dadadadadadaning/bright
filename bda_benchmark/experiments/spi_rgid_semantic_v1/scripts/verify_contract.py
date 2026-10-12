from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_VARIANTS = {
    "c0_raw_p3",
    "c1_local_window_a6",
    "c2_loccc_a6",
    "c3_rgid_g3_a6",
}
EXPECTED_FORBIDDEN = {
    "official_instance_annotations",
    "standard_ml_val450",
    "original_standard_test",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def verify_contract(contract: dict, project_root: Path) -> None:
    if contract.get("protocol") != "SPI_RGID_SEMANTIC_V1_ISOLATED":
        raise ValueError("unexpected protocol identity")
    if set(contract.get("variants", {})) != EXPECTED_VARIANTS:
        raise ValueError("C0--C3 variant set has changed")
    if set(contract.get("training_forbidden_sources", [])) != EXPECTED_FORBIDDEN:
        raise ValueError("training leakage boundary has changed")
    if abs(float(contract.get("locked_p3_valid_miou")) - 0.60502) > 1e-12:
        raise ValueError("locked P3 reference has changed")

    a6 = contract.get("locked_a6", {})
    expected_a6 = {
        "minimum_area": 32,
        "minimum_purity": 0.70,
        "minimum_probability_margin": 0.10,
        "local_kernel_size": 5,
        "area_upper_bound": None,
        "core_erosion": 0,
    }
    if a6 != expected_a6:
        raise ValueError("locked A6 contract has changed")

    for relative in contract.get("read_only_dependencies", []):
        if not (project_root / relative).exists():
            raise FileNotFoundError(f"missing read-only dependency: {relative}")

    outputs = [str(value) for value in contract.get("owned_output_namespaces", [])]
    if len(outputs) != len(set(outputs)) or not outputs:
        raise ValueError("owned output namespaces must be non-empty and unique")


def verify_provenance(path: Path) -> None:
    provenance = load_json(path)
    required_false = {
        "official_instances_accessed": False,
        "standard_ml_val_accessed": False,
        "original_standard_test_accessed": False,
    }
    for key, expected in required_false.items():
        if provenance.get(key) is not expected:
            raise ValueError(f"unsafe RGID provenance: {key}={provenance.get(key)!r}")
    if provenance.get("calibration_split") != "train_only":
        raise ValueError("RGID calibration_split must be train_only")
    if not provenance.get("p3_sha256"):
        raise ValueError("RGID provenance must record p3_sha256")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()

    verify_contract(load_json(args.contract), args.project_root.resolve())
    if args.provenance is not None:
        verify_provenance(args.provenance)
    print("SPI-RGID semantic isolation contract verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
