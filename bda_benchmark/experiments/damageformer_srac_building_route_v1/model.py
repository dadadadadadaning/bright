"""Matched, trainable region readouts on one frozen SRAC front and Q.

The source front performs one complete forward and supplies predicted building
probability, the original Q, its adapted content, and the raw fused ROI.  Each
route head then forms its *own* region tokens from those pixel features.  In
particular, split modes never split already-mixed original region tokens.
"""
from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from .routing import MODES, pool_routes, read_regions, route_from_q, route_geometry


class FrozenFrontCapture:
    """Run a frozen front once and capture the exact A1 features it consumed.

    A1 passes ``fused_roi`` both into ``clf_2`` and through the front as the
    second input to ``embedding``.  The predicted building probability is the
    second input to ``adapter``.  Capturing these existing arguments avoids a
    second A1 forward and works for both the original front and completed
    Q-factorial fronts that preserve the SRAC adapter/embedding interface.
    """

    def __init__(self, front: nn.Module):
        if not hasattr(front, "adapter") or not hasattr(front, "embedding"):
            raise TypeError("Expected an SRAC front with adapter and embedding")
        self.front = front.eval().requires_grad_(False)
        self._raw: list[torch.Tensor] = []
        self._building: list[torch.Tensor] = []
        self._handles = [
            front.adapter.register_forward_pre_hook(self._capture_adapter),
            front.embedding.register_forward_pre_hook(self._capture_embedding),
        ]

    def _capture_adapter(self, _module: nn.Module, args: tuple[Any, ...]) -> None:
        if len(args) < 2 or not torch.is_tensor(args[1]):
            raise RuntimeError("SRAC adapter did not receive predicted building probability")
        self._building.append(args[1].detach())

    def _capture_embedding(self, _module: nn.Module, args: tuple[Any, ...]) -> None:
        if len(args) < 2 or not torch.is_tensor(args[1]):
            raise RuntimeError("SRAC embedding did not receive raw fused ROI")
        self._raw.append(args[1].detach())

    def __call__(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, Any]:
        self._raw.clear()
        self._building.clear()
        self.front.eval()
        with torch.no_grad():
            result = self.front(pre, post)
        if len(self._raw) != 1 or len(self._building) != 1:
            raise RuntimeError("Expected exactly one frozen front adapter/embedding forward")
        if not isinstance(result, dict) or "q" not in result or "adapted_content" not in result:
            raise RuntimeError("SRAC front did not return Q and adapted pixel content")
        return dict(result=result, fused_roi=self._raw[0], building=self._building[0],
                    output_size=tuple(pre.shape[-2:]))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class BuildingRouteHead(nn.Module):
    """A matched trainable embedding/classifier for one predicted-B route mode.

    All three modes copy the *same* front readout initialization.  The source
    front is registered for device movement but held in eval mode and excluded
    from gradients.  Optimizers should use ``trainable_parameters()`` rather
    than ``model.parameters()``; checkpoints should use ``head_state_dict()``.
    """

    def __init__(
        self,
        front: nn.Module,
        mode: str = "unsplit",
        *,
        threshold: float = 0.5,
        soft_floor: float = 0.0,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown route mode: {mode}")
        if not hasattr(front, "embedding") or not hasattr(front, "region_classifier"):
            raise TypeError("Expected an SRAC front with embedding and region_classifier")
        self.front = front.eval().requires_grad_(False)
        self.embedding = copy.deepcopy(front.embedding)
        self.region_classifier = copy.deepcopy(front.region_classifier)
        # Deepcopy preserves the source front's frozen requires_grad flags.
        # The copied readout is the *only* trainable part of this experiment.
        self.embedding.requires_grad_(True)
        self.region_classifier.requires_grad_(True)
        self.mode = mode
        self.threshold = float(threshold)
        self.soft_floor = float(soft_floor)

    def train(self, mode: bool = True):
        super().train(mode)
        self.front.eval()
        return self

    def trainable_parameters(self):
        yield from self.embedding.parameters()
        yield from self.region_classifier.parameters()

    def head_state_dict(self) -> dict[str, Any]:
        """Only the new readout; the large source front stays referenced by hash."""
        return dict(embedding=self.embedding.state_dict(),
                    region_classifier=self.region_classifier.state_dict(),
                    mode=self.mode, threshold=self.threshold, soft_floor=self.soft_floor)

    def load_head_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("mode") != self.mode or state.get("threshold") != self.threshold or state.get("soft_floor") != self.soft_floor:
            raise ValueError("Readout checkpoint route configuration differs")
        if set(state) != {"embedding", "region_classifier", "mode", "threshold", "soft_floor"}:
            raise ValueError("Readout checkpoint has missing or extra fields")
        self.embedding.load_state_dict(state["embedding"], strict=True)
        self.region_classifier.load_state_dict(state["region_classifier"], strict=True)

    def from_frozen(
        self,
        result: dict[str, Any],
        fused_roi: torch.Tensor,
        building: torch.Tensor,
        output_size: tuple[int, int],
    ) -> dict[str, Any]:
        """Train the readout on a *single already-computed* frozen front pass."""
        if not isinstance(result, dict) or any(key not in result for key in (
                "q", "ids", "count", "shape", "adapted_content")):
            raise ValueError("Frozen source lacks the required Q and pixel features")
        adapted = result["adapted_content"].detach()
        raw = fused_roi.detach()
        if adapted.ndim != 4 or raw.ndim != 4 or adapted.shape[0] != raw.shape[0]:
            raise ValueError("Adapted and raw features must be batched feature maps")
        if adapted.shape[-2:] != tuple(result["shape"]) or raw.shape[-2:] != tuple(result["shape"]):
            raise ValueError("All features and Q must share the native feature grid")
        route = route_from_q(
            result["q"], result["ids"], result["count"], result["shape"], building,
            self.mode, threshold=self.threshold, soft_floor=self.soft_floor,
        )
        e = self.embedding
        spatial_weight = e.score(adapted).float().sigmoid().flatten(2)
        main = pool_routes(adapted, route["q"] * spatial_weight, route["ids"], route["count"])
        raw_stats = torch.cat((raw.float(), raw.float().square()), dim=1)
        stats = pool_routes(raw_stats, route["q"], route["ids"], route["count"])
        mean, second = stats["tokens"].chunk(2, dim=-1)
        std = (second - mean.square()).clamp_min(0).add(1e-6).sqrt()
        geometry = route_geometry(route)
        tokens = e.main(main["tokens"])
        tokens = tokens + e.skip_scale.tanh() * e.stats(torch.cat((mean, std), dim=-1))
        tokens = e.norm(tokens + e.geometry(geometry))
        valid = stats["mass"] > 1e-6
        if e.context is not None:
            tokens = e.context(tokens, valid)
        else:
            tokens = tokens.masked_fill(~valid[..., None], 0)
        region_logits = self.region_classifier(tokens)
        probability = read_regions(region_logits, route, output_size)
        return dict(prob=probability, region_logits=region_logits, tokens=tokens,
                    mass=stats["mass"], geometry=geometry, valid_regions=valid,
                    route_q=route["q"], route_ids=route["ids"],
                    route_count=route["count"], route_mode=self.mode,
                    q=result["q"], ids=result["ids"], count=result["count"],
                    shape=result["shape"], loc=result.get("loc"),
                    old_pixel=result.get("old_pixel"),
                    building_grid=route["building_grid"])

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> dict[str, Any]:
        capture = FrozenFrontCapture(self.front)
        try:
            source = capture(pre, post)
        finally:
            capture.close()
        return self.from_frozen(**source)
