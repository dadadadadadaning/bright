#!/usr/bin/env python3
"""V3 entrypoint reusing the mature V2 optimizer/evaluation engine."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from bda_benchmark.experiments.damageformer_building_internal_change_partition_v2 import train as engine

from .losses import stage_losses
from .model import DamageFormerRGBAnchoredChangeSuperpixelV3
from .variants import VARIANTS, get_variant


def _checkpoint_payload(model, optimizer, scheduler, step, best_metrics, parent_report, args):
    return {
        "protocol": "DAMAGEFORMER_RGB_ANCHORED_CHANGE_SUPERPIXEL_V3",
        "variant": args.variant,
        "stage": args.stage,
        "step": int(step),
        "plugin_model": model.biscp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_metrics": best_metrics,
        "parent_report": parent_report,
        "contract": model.checkpoint_contract(),
        "test_accessed": False,
    }


def _rewrite_summary() -> None:
    try:
        output = Path(sys.argv[sys.argv.index("--output_dir") + 1]) / "training_summary.json"
    except (ValueError, IndexError):
        return
    if not output.is_file():
        return
    data = json.loads(output.read_text(encoding="utf-8"))
    data["protocol"] = "DAMAGEFORMER_RGB_ANCHORED_CHANGE_SUPERPIXEL_V3"
    data["engine"] = "V2 stage-wise training engine; V3 model/loss/checkpoint contract"
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    engine.VARIANTS = VARIANTS
    engine.get_variant = get_variant
    engine.DamageFormerBuildingInternalChangePartitionV2 = (
        DamageFormerRGBAnchoredChangeSuperpixelV3
    )
    engine.stage_losses = stage_losses
    engine._checkpoint_payload = _checkpoint_payload
    engine.main()
    _rewrite_summary()


if __name__ == "__main__":
    main()
