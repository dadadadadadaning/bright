# -*- coding: utf-8 -*-
"""Validate and report an external SARMAE checkpoint before H11 training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

BDA_ROOT = Path(__file__).resolve().parents[2]
if str(BDA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(BDA_ROOT.parent))

from bda_benchmark.model.DamageFormer_RGBSAR_H11_SATR import SARMAEEncoder  # noqa: E402
from bda_benchmark.script.standard_ML.train_H11_SATR import (  # noqa: E402
    file_sha256,
    migrate_sarmae_encoder_state,
)


def main() -> None:
    parser = argparse.ArgumentParser("Check H11 SARMAE checkpoint compatibility")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("SARMAE checkpoint root must be a mapping")
    # Audit the new default transfer contract: preserve the official
    # three-channel PatchEmbed exactly.  The legacy one-channel collapse is
    # retained only as the completed T1 ablation and is covered by unit tests.
    encoder = SARMAEEncoder(in_channels=3)
    report = migrate_sarmae_encoder_state(encoder, checkpoint)
    payload = {
        "protocol": "H11_SARMAE_checkpoint_audit_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "sha256": file_sha256(args.checkpoint),
        "source_protocol": checkpoint.get("protocol"),
        "source_epoch": checkpoint.get("epoch"),
        "test_accessed": False,
        **report,
    }
    print(json.dumps(payload, indent=2), flush=True)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {output}", flush=True)
    print("SARMAE checkpoint: COMPATIBLE", flush=True)


if __name__ == "__main__":
    main()
