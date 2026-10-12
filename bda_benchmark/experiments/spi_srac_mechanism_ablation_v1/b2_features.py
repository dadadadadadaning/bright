from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _to_nchw(value: torch.Tensor, channels: int, name: str) -> torch.Tensor:
    if value.ndim != 4:
        raise RuntimeError(f"{name} must be 4-D, got {tuple(value.shape)}")
    if value.shape[1] == channels:
        return value.contiguous()
    if value.shape[-1] == channels:
        return value.permute(0, 3, 1, 2).contiguous()
    if value.shape[2] == channels:
        return value.permute(0, 2, 1, 3).contiguous()
    raise RuntimeError(f"cannot infer layout for {name}: {tuple(value.shape)}")


def load_b2_model(args: Any, device: torch.device):
    """Strictly load the BRIGHT B2 DiffFusion architecture.

    Deliberately refuses B0/encoder-only weights.  A silent non-strict load
    would make the experiment claim DiffFusion while actually ignoring it.
    """
    root = Path(args.changemamba_root).resolve()
    if not (root / "changedetection").is_dir():
        raise RuntimeError(f"invalid ChangeMamba root: {root}")
    import sys
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from changedetection.checkpoints import load_model_weights
    from changedetection.configs.config import get_config
    from changedetection.models.ChangeMambaMMBDA_B2_DiffFusion import (
        ChangeMambaMMBDA_B2_DiffFusion,
    )
    from changedetection.script.script_utils import get_vssm_kwargs

    args.opts = None
    config = get_config(args)
    model = ChangeMambaMMBDA_B2_DiffFusion(
        output_building=2,
        output_damage=4,
        pretrained=None,
        **get_vssm_kwargs(config),
    ).to(device).eval()
    report = load_model_weights(model, args.changemamba_ckpt)
    unresolved = (
        report.get("missing_keys", [])
        or report.get("unexpected_keys", [])
        or report.get("mismatched_keys", [])
    )
    if unresolved:
        raise RuntimeError(f"strict B2 DiffFusion checkpoint load failed: {report}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, report


@torch.inference_mode()
def forward_b2_features(
    model: Any,
    pre: torch.Tensor,
    post: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Expose exactly attributable B2 pre-head evidence maps.

    ``base_change`` is the original ChangeDecoder output, ``explicit_diff``
    is B2's projected DiffMambaFusion output, and ``enhanced_change`` is the
    feature consumed by the trained classifier.  The three must be audited
    separately before any Memory claim is made.
    """
    pre_features = model.encoder_1(pre)
    post_features = model.encoder_2(post)
    rgb_structure = model.decoder_building(pre_features)
    base_change = model.decoder_damage(pre_features, post_features)

    channels = int(model.enc_last_dim)
    pre_deep = _to_nchw(pre_features[-1], channels, "pre_deep")
    post_deep = _to_nchw(post_features[-1], channels, "post_deep")
    explicit_diff = model.diff_fusion(pre_deep, post_deep)
    explicit_diff = _to_nchw(explicit_diff, channels, "explicit_diff")
    explicit_diff = model.diff_proj(explicit_diff)
    explicit_diff = F.interpolate(
        explicit_diff,
        size=base_change.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    gamma = torch.tanh(model.diff_gamma).reshape(1, 1, 1, 1)
    enhanced_change = base_change + gamma * explicit_diff
    return {
        "rgb_structure": rgb_structure,
        "base_change": base_change,
        "explicit_diff": explicit_diff,
        "enhanced_change": enhanced_change,
        "diff_gamma_raw": model.diff_gamma.detach().reshape(1),
        "diff_gamma_effective": gamma.detach().reshape(1),
    }


__all__ = ["forward_b2_features", "load_b2_model"]

