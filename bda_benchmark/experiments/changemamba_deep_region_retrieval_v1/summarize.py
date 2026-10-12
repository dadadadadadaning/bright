#!/usr/bin/env python3
"""Strict mechanism admission; never converts region evidence into a Test claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import PROTOCOL


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.result_root)
    records = torch.load(args.records, map_location="cpu", weights_only=False)
    base = float(records["metadata"]["base_valid_miou"])
    grid = read(root / "grid__c4_structural__p3_attn_tail_evidence.json")
    simple = read(root / "slico__c0_base__p0_mean_evidence.json")
    structural = read(root / "slico__c4_structural__p0_mean_evidence.json")
    primary = read(root / "slico__c4_structural__p3_attn_tail_rag.json")
    controls = {name: read(root / f"slico__c4_structural__p3_attn_tail_rag_{name}.json")
                for name in ("shuffled_labels", "wrong_labels", "random_neighbours")}

    partition_pp = 100 * (primary["direct"]["macro_f1"] - grid["direct"]["macro_f1"])
    representation_pp = 100 * (structural["direct"]["macro_f1"] - simple["direct"]["macro_f1"])
    pooling_pp = 100 * (primary["direct"]["macro_f1"] - structural["direct"]["macro_f1"])
    retrieval_pp = 100 * (primary["rag"]["macro_f1"] - primary["direct"]["macro_f1"])
    control_margin = min(100 * (primary["rag"]["macro_f1"] - value["rag"]["macro_f1"])
                         for value in controls.values())
    result = {
        "protocol": PROTOCOL + "_SUMMARY",
        "scope": "region-level mechanism admission; not final pixel mIoU and not Test evidence",
        "changemamba_val_reproduction": base,
        "gates": {
            "G0_base_reproduced": base > 0,
            "G1_slico_minus_grid_direct_macro_f1_pp": partition_pp,
            "G2_structural_minus_base_representation_pp": representation_pp,
            "G3_attention_minus_mean_pooling_pp": pooling_pp,
            "G4_rag_minus_direct_pp": retrieval_pp,
            "G5_correct_rag_minus_strongest_control_pp": control_margin,
        },
        "primary": {
            "direct_macro_f1": primary["direct"]["macro_f1"],
            "knn_macro_f1": primary["knn"]["macro_f1"],
            "rag_macro_f1": primary["rag"]["macro_f1"],
            "rag_recovery_rate": primary["complementarity"]["rag_recovery_rate"],
            "rag_harm_rate": primary["complementarity"]["rag_harm_rate"],
        },
        "admission": {
            "region_representation_pass": representation_pp >= 0.20 and pooling_pp >= 0.0,
            "retrieval_pass": retrieval_pp >= 0.20 and control_margin >= 0.20,
            "learnable_superpixel_allowed_next": partition_pp >= 0.0,
            "pixel_fusion_allowed_next": retrieval_pp >= 0.20 and control_margin >= 0.20,
            "test_allowed": False,
        },
        "test_accessed": False,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

