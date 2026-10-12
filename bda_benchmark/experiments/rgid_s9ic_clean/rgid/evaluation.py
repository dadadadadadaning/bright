from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

import torch

from .config import RGIDConfig
from .geometry import RGIDDecoder
from .metrics import aggregate_metric_dicts, evaluate_instance_labels
from .model import RGIDHead


def load_head_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    config = RGIDConfig.from_dict(dict(checkpoint["config"]))
    head = RGIDHead(
        in_channels=config.in_channels,
        hidden_channels=config.hidden_channels,
        num_shared_blocks=config.num_shared_blocks,
    ).to(device)
    head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    head.eval()
    return head, config, checkpoint


def _aggregate_decode_stats(rows):
    output: Dict[str, object] = {}
    for row in rows:
        for key, value in row.items():
            if key == "markers_per_split_region":
                continue
            if isinstance(value, dict):
                target = output.setdefault(key, {})
                for sub_key, sub_value in value.items():
                    target[sub_key] = target.get(sub_key, 0) + sub_value
            elif isinstance(value, list):
                output.setdefault(key, []).extend(value)
            elif isinstance(value, (int, float)):
                output[key] = output.get(key, 0) + value
    markers = output.get("markers_in_split_candidates", [])
    output["markers_per_split_region"] = (
        sum(markers) / len(markers) if markers else 0.0
    )
    return output


@torch.no_grad()
def evaluate_g123(
    *,
    head,
    dataloader,
    config: RGIDConfig,
    device="cuda",
    boundary_gate_threshold=None,
):
    device = torch.device(device)
    head = head.to(device).eval()
    decoder = RGIDDecoder(config)
    metric_rows = {mode: [] for mode in ("g1", "g2", "g3")}
    decode_rows = {mode: [] for mode in ("g1", "g2", "g3")}
    for batch in dataloader:
        required = ("fused_roi", "parent_mask", "instance_ids")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"validation batch missing {missing}")
        fused_roi = batch["fused_roi"].to(device).float()
        outputs = head(fused_roi)
        center_logits = outputs["center_logits"].float().cpu()
        boundary_logits = outputs["boundary_logits"].float().cpu()
        parents = batch["parent_mask"].detach().cpu()
        instances = batch["instance_ids"].detach().cpu()
        for index in range(fused_roi.shape[0]):
            for mode in ("g1", "g2", "g3"):
                decoded = decoder.decode(
                    center_logits[index, 0],
                    boundary_logits[index, 0],
                    parents[index],
                    mode,
                    inputs_are_logits=True,
                    boundary_gate_threshold=boundary_gate_threshold,
                )
                metric_rows[mode].append(
                    evaluate_instance_labels(
                        instances[index].numpy(),
                        decoded.instance_labels,
                        match_iou_threshold=config.match_iou_threshold,
                        merge_split_overlap_threshold=(
                            config.merge_split_overlap_threshold
                        ),
                    )
                )
                decode_rows[mode].append(decoded.stats.to_dict())
    return {
        mode: {
            "instance_metrics": aggregate_metric_dicts(metric_rows[mode]),
            "decode_stats": _aggregate_decode_stats(decode_rows[mode]),
        }
        for mode in ("g1", "g2", "g3")
    }


def save_evaluation(results, output_json) -> None:
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
