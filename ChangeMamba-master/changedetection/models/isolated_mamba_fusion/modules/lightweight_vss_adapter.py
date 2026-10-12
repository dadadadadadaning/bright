"""Optional VMamba/VSS adapter with a safe fallback path.

This adapter tries to use an installed VSSBlock when available.  If VMamba,
selective_scan, or any custom CUDA dependency is missing, it falls back to the
pure PyTorch block in ``fallback_blocks.py``.

No code from the original BRIGHT/DamageFormer project is imported here.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .fallback_blocks import FallbackMambaLikeBlock


def _try_import_vssblock() -> Tuple[Optional[type], Optional[type]]:
    """Try a few common VMamba import locations.

    Returns:
        (VSSBlock, LayerNorm2d) if available, otherwise (None, None).
    """

    candidates = [
        "vmamba",
        "changedetection.models.vmamba",
    ]
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        vss_block = getattr(module, "VSSBlock", None)
        layer_norm_2d = getattr(module, "LayerNorm2d", None)
        if vss_block is not None:
            return vss_block, layer_norm_2d
    return None, None


class LightweightVSSAdapter(nn.Module):
    """Use VMamba VSSBlock if possible, otherwise use a fallback block.

    Args:
        channels: input/output channel count.
        backend: ``"auto"``, ``"fallback"``, or ``"vss"``.
            - ``auto``: try VSSBlock, fallback silently if unavailable.
            - ``fallback``: always use fallback block.
            - ``vss``: require VSSBlock; if construction fails, raise.
        vss_kwargs: optional keyword overrides for VSSBlock.
    """

    def __init__(
        self,
        channels: int,
        backend: str = "auto",
        fallback_kernel_size: int = 3,
        fallback_expansion: float = 1.0,
        fallback_norm_type: str = "bn",
        vss_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if backend not in {"auto", "fallback", "vss"}:
            raise ValueError(f"Unsupported backend: {backend}")

        self.channels = channels
        self.backend = backend
        self.active_backend = "fallback"
        self._vss_failed_at_forward = False

        self.fallback = FallbackMambaLikeBlock(
            channels=channels,
            expansion=fallback_expansion,
            kernel_size=fallback_kernel_size,
            norm_type=fallback_norm_type,
        )
        self.vss: Optional[nn.Module] = None

        if backend in {"auto", "vss"}:
            try:
                self.vss = self._build_vss(channels, vss_kwargs or {})
                self.active_backend = "vss"
            except Exception:
                if backend == "vss":
                    raise
                self.vss = None
                self.active_backend = "fallback"

    def _build_vss(self, channels: int, overrides: Dict[str, Any]) -> nn.Module:
        vss_block, layer_norm_2d = _try_import_vssblock()
        if vss_block is None:
            raise ImportError("No VSSBlock found.")

        norm_layer = layer_norm_2d if layer_norm_2d is not None else nn.BatchNorm2d
        kwargs: Dict[str, Any] = dict(
            hidden_dim=channels,
            drop_path=0.0,
            norm_layer=norm_layer,
            channel_first=True,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=0.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            use_checkpoint=False,
        )
        kwargs.update(overrides)
        return vss_block(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.vss is not None and not self._vss_failed_at_forward:
            try:
                return self.vss(x)
            except Exception:
                if self.backend == "vss":
                    raise
                # If the installed VSSBlock requires unavailable CUDA kernels,
                # permanently switch to fallback for this module instance.
                self._vss_failed_at_forward = True
                self.active_backend = "fallback"
                for param in self.vss.parameters():
                    param.requires_grad_(False)
        return self.fallback(x)
