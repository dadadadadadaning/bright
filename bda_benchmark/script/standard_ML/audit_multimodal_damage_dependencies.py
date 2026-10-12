#!/usr/bin/env python3
"""Read-only dependency audit for the staged R1--R6 experiment chain."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    third_party = ROOT / "third_party"
    sar_dino_checkpoint = os.environ.get("SAR_DINO_CKPT", "")
    sar_dino_checkpoint_ready = bool(sar_dino_checkpoint) and Path(
        sar_dino_checkpoint
    ).is_file()
    records = {
        "r1_flowmamba": {
            "source": str(third_party / "FlowMamba"),
            "source_ready": (third_party / "FlowMamba" / "changedetection").is_dir(),
            "dependencies": {
                "timm": package("timm"),
                "mmengine": package("mmengine"),
            },
        },
        "r2_m2cd": {
            "source": str(third_party / "M2CD"),
            "source_ready": (third_party / "M2CD" / "opencd").is_dir(),
            "dependencies": {
                "mmengine": package("mmengine"),
                "mmseg": package("mmseg"),
            },
        },
        "r4_sar_dino": {
            "source": str(third_party / "MM-OVSeg"),
            "source_ready": (third_party / "MM-OVSeg").is_dir(),
            "checkpoint_required": True,
            "checkpoint": sar_dino_checkpoint or None,
            "checkpoint_ready": sar_dino_checkpoint_ready,
        },
    }
    print(json.dumps(records, ensure_ascii=False, indent=2))
    print("\nStage readiness:")
    for stage, record in records.items():
        ready = bool(record["source_ready"]) and all(
            record.get("dependencies", {}).values()
        )
        if record.get("checkpoint_required"):
            ready = ready and bool(record.get("checkpoint_ready"))
        print(f"{stage:<18}: {'READY' if ready else 'BLOCKED'}")
    print("Test accessed: False")


if __name__ == "__main__":
    main()
