#!/usr/bin/env python3
"""Fail early when a B0 checkpoint is accidentally supplied as B2 DiffFusion."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def state_dict(payload):
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint is not a mapping")
    for key in ("model", "state_dict", "model_state_dict", "net"):
        value = payload.get(key)
        if isinstance(value, dict) and value:
            return value
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise RuntimeError("cannot locate model state_dict in checkpoint")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = state_dict(payload)
    keys = [str(key).removeprefix("module.") for key in state]
    required = {
        "diff_fusion": any("diff_fusion." in key for key in keys),
        "diff_proj": any("diff_proj." in key for key in keys),
        "diff_gamma": any(key.endswith("diff_gamma") for key in keys),
    }
    print(f"Checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"Parameter tensors: {len(keys)}")
    print("B2 markers:", required)
    if not all(required.values()):
        raise RuntimeError(
            "Checkpoint is not a trained B2 DiffFusion model. It looks like a "
            "B0/base ChangeMamba checkpoint; do not use non-strict loading because "
            "the missing difference branch would remain randomly initialized."
        )
    print("B2 DiffFusion checkpoint markers passed.")


if __name__ == "__main__":
    main()

