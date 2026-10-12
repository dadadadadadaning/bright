"""Feature-only access to a frozen ChangeMamba-BRIGHT model.

The accepted ChangeMamba model is not modified.  This module mirrors the
public decoder forward pass so that classification-preceding 128-channel
features can be cached for the retrieval experiment.
"""
from __future__ import annotations

from typing import Any

import torch


def change_pyramid(decoder: Any, pre_features, post_features) -> list[torch.Tensor]:
    """Return the exact hierarchical decoder states, deep-to-shallow."""
    previous = None
    output: list[torch.Tensor] = []
    for stage_idx, (pre_feat, post_feat) in enumerate(
        zip(reversed(pre_features), reversed(post_features))
    ):
        collected = []
        for mode in decoder.fusion_modes_by_stage[stage_idx]:
            collected.extend(
                decoder._collect_fusion_features(
                    mode, decoder.stage_blocks[stage_idx][mode], pre_feat, post_feat
                )
            )
        current = decoder.fuse_layers[stage_idx](torch.cat(collected, dim=1))
        if previous is not None:
            current = decoder.smooth_layers[stage_idx - 1](
                decoder._upsample_add(previous, current)
            )
        output.append(current)
        previous = current
    return output


def forward_features(model: Any, pre: torch.Tensor, post: torch.Tensor) -> dict[str, Any]:
    """Run frozen ChangeMamba and expose pre-head feature maps.

    ``post`` follows the original BRIGHT three-channel display-SAR contract.
    Native-SAR adapters are deliberately not mixed into this first controlled
    experiment because that would change two factors at once.
    """
    pre_features = model.encoder_1(pre)
    post_features = model.encoder_2(post)
    rgb_structure = model.decoder_building(pre_features)
    pyramid = change_pyramid(model.decoder_damage, pre_features, post_features)
    change_local = pyramid[-1]
    change_semantic = pyramid[-2] if len(pyramid) > 1 else pyramid[-1]
    return {
        "rgb_structure": rgb_structure,
        "change_local": change_local,
        "change_semantic": change_semantic,
    }


__all__ = ["change_pyramid", "forward_features"]
