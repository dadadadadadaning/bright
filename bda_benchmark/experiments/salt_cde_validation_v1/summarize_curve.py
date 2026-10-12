# -*- coding: utf-8 -*-
"""Summarise all declared perturbations without turning diagnostics into labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODES = ("shift_x4", "shift_y4", "shift_x8", "shift_y8", "shift_x16", "shift_y16", "pair_shuffle", "zero")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def gate_mean(payload):
    values = [float(value) for key, value in payload.get("diagnostics", {}).items() if key.endswith("_sar_gate")]
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser("SALT perturbation curve")
    parser.add_argument("--audit_dir", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    root = Path(args.audit_dir)
    clean = read(root / "none.json")
    clean_miou = float(clean["metrics"]["valid_miou"])
    clean_gate = gate_mean(clean)
    rows = {}
    for mode in MODES:
        payload = read(root / f"{mode}.json")
        score = float(payload["metrics"]["valid_miou"])
        gate = gate_mean(payload)
        offset_errors = [
            float(value["absolute_error"])
            for value in payload.get("offset_recovery", {}).values()
        ]
        rows[mode] = {
            "valid_miou": score,
            "drop_pp": 100.0 * (clean_miou - score),
            "mean_gate": gate,
            "gate_reduction_from_clean": (
                clean_gate - gate if clean_gate is not None and gate is not None else None
            ),
            "mean_feature_offset_error": (
                float(np.mean(offset_errors)) if offset_errors else None
            ),
        }
    result = {
        "protocol": "SALT_CDE_PERTURBATION_CURVE_v1",
        "variant": clean["variant"],
        "clean_valid_miou": clean_miou,
        "clean_gate": clean_gate,
        "perturbations": rows,
        "test_accessed": False,
    }
    destination = Path(args.output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
