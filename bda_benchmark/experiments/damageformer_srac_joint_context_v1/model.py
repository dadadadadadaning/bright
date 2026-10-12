"""Matched pixel/region/Memory joint decoder on a frozen SRAC front.

The important contract is that the new classifier starts as an *exact copy* of
the frozen A1 pixel classifier.  Region or Memory context enters through extra
zero-initialized input columns of the same 1x1 classifier.  Consequently every
variant has the same step-zero prediction and any later difference must be
learned from its context source.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn


VARIANTS = ("j0", "j1", "j2")
VARIANT_MEANING = {
    "j0": "pixel only; region-context tensor is exactly zero",
    "j1": "pixel plus current-region context read with learned no-Memory slots",
    "j2": "pixel plus current-region context read from Train-only Memory",
}


def unpool_region_tokens(
    tokens: torch.Tensor,
    q: torch.Tensor,
    ids: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    """Use the exact same sparse association Q to return region tokens to pixels.

    Args:
        tokens: ``[B, K, D]`` region tokens.
        q: ``[B, S, H*W]`` normalized local assignments.
        ids: ``[S, H*W]`` global region IDs for each local support.
        shape: feature-grid ``(H, W)``.
    Returns:
        Context map ``[B, D, H, W]``.
    """
    if tokens.ndim != 3 or q.ndim != 3 or ids.ndim != 2:
        raise ValueError("Expected tokens[B,K,D], q[B,S,N], ids[S,N]")
    batch, regions, dim = tokens.shape
    supports, pixels = ids.shape
    height, width = shape
    if q.shape != (batch, supports, pixels) or pixels != height * width:
        raise ValueError("Sparse association shape mismatch")
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= regions):
        raise ValueError("Region ID outside token table")
    if not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError("Q must be normalized over local supports")

    pixel_tokens = tokens.new_zeros(batch, pixels, dim)
    for support_index in range(supports):
        index = ids[support_index][None, :, None].expand(batch, -1, dim)
        gathered = tokens.gather(1, index)
        pixel_tokens = pixel_tokens + gathered * q[:, support_index, :, None].to(tokens.dtype)
    return pixel_tokens.transpose(1, 2).reshape(batch, dim, height, width)


class ContextMemoryReader(nn.Module):
    """Pre-LN cross-attention that returns an enhanced token, not class logits."""

    def __init__(self, dim: int = 128, num_events: int = 1, heads: int = 4, slots: int = 8):
        super().__init__()
        if slots < 1 or dim % heads:
            raise ValueError("Invalid Memory reader dimensions")
        self.dim = int(dim)
        self.slots = int(slots)
        self.label_embedding = nn.Embedding(4, dim)
        self.disaster_embedding = nn.Embedding(8, dim)
        self.event_embedding = nn.Embedding(num_events + 1, dim)
        self.geometry_embedding = nn.Linear(5, dim)
        self.learned_slots = nn.Parameter(torch.empty(slots, dim))
        self.learned_label_logits = nn.Parameter(torch.zeros(slots, 4))
        nn.init.normal_(self.learned_slots, std=0.02)
        with torch.no_grad():
            self.learned_label_logits[torch.arange(slots), torch.arange(slots) % 4] = 1.0
        self.query_norm = nn.LayerNorm(dim)
        self.key_norm = nn.LayerNorm(dim)
        self.value_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def _learned_support(self, query: torch.Tensor):
        labels = self.learned_label_logits.softmax(-1)
        values = self.learned_slots + labels @ self.label_embedding.weight
        keys = self.learned_slots[None].expand(len(query), -1, -1)
        values = values[None].expand(len(query), -1, -1)
        valid = torch.ones(len(query), self.slots, dtype=torch.bool, device=query.device)
        return keys, values, valid

    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor | None = None,
        distribution: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        mode: str = "nomemory",
        metadata: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if query.ndim != 2 or query.shape[-1] != self.dim:
            raise ValueError("Query must be [N,D]")
        fallback_key, fallback_value, fallback_valid = self._learned_support(query)
        if mode == "nomemory":
            keys, values, attention_valid = fallback_key, fallback_value, fallback_valid
        elif mode == "memory":
            if support is None or distribution is None or valid is None:
                raise ValueError("Memory mode requires support, distribution and valid")
            if support.shape != (len(query), self.slots, self.dim):
                raise ValueError("Memory support shape mismatch")
            if distribution.shape != (len(query), self.slots, 4) or valid.shape != (len(query), self.slots):
                raise ValueError("Memory label/mask shape mismatch")
            memory_values = support + distribution @ self.label_embedding.weight
            if metadata is not None:
                memory_values = memory_values + self.disaster_embedding(metadata["disasters"])
                memory_values = memory_values + self.event_embedding(metadata["events"] + 1)
                memory_values = memory_values + self.geometry_embedding(metadata["geometry"].float())
            # Missing historical neighbors are replaced slot-for-slot by learned
            # controls, avoiding all-masked attention and keeping J1/J2 matched.
            keys = torch.where(valid[..., None], support, fallback_key)
            values = torch.where(valid[..., None], memory_values, fallback_value)
            attention_valid = fallback_valid
        else:
            raise ValueError(f"Unknown reader mode: {mode}")

        context, _ = self.cross_attention(
            self.query_norm(query)[:, None],
            self.key_norm(keys),
            self.value_norm(values),
            key_padding_mask=~attention_valid,
            need_weights=False,
        )
        hidden = query + context[:, 0]
        return hidden + self.ffn(self.ffn_norm(hidden))


class IdentityJointClassifier(nn.Module):
    """A partitioned implementation of one ``[pixel, context]`` 1x1 head.

    Splitting the two input blocks is algebraically identical to one wider
    convolution.  It also preserves the A1 convolution bit-for-bit at step zero;
    some convolution kernels change floating-point accumulation order when 128
    all-zero input columns are appended to the same weight tensor.
    """

    def __init__(self, a1_classifier: nn.Conv2d, context_dim: int):
        super().__init__()
        self.pixel = nn.Conv2d(256, 4, 1, bias=a1_classifier.bias is not None)
        self.context = nn.Conv2d(context_dim, 4, 1, bias=False)
        self.initialize_a1_identity(a1_classifier)

    @torch.no_grad()
    def initialize_a1_identity(self, a1_classifier: nn.Conv2d):
        self.pixel.weight.copy_(a1_classifier.weight)
        if self.pixel.bias is not None:
            self.pixel.bias.copy_(a1_classifier.bias)
        self.context.weight.zero_()

    def forward(self, pixel: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.pixel(pixel) + self.context(context)


class JointContextDecoder(nn.Module):
    """A1 pixel evidence and unpooled region context share one 1x1 classifier."""

    def __init__(
        self,
        a1_classifier: nn.Conv2d,
        dim: int = 128,
        num_events: int = 1,
        heads: int = 4,
        slots: int = 8,
    ):
        super().__init__()
        if not isinstance(a1_classifier, nn.Conv2d):
            raise TypeError("A1 classifier must be Conv2d")
        if a1_classifier.in_channels != 256 or a1_classifier.out_channels != 4:
            raise ValueError("Expected the A1 256-to-4 pixel classifier")
        if a1_classifier.kernel_size != (1, 1):
            raise ValueError("A1 identity initialization requires a 1x1 classifier")
        self.dim = int(dim)
        self.reader = ContextMemoryReader(dim, num_events, heads, slots)
        self.classifier = IdentityJointClassifier(a1_classifier, dim)

    def _read_chunks(
        self,
        query: torch.Tensor,
        mode: str,
        support: torch.Tensor | None,
        distribution: torch.Tensor | None,
        valid: torch.Tensor | None,
        metadata: Dict[str, torch.Tensor] | None,
        chunk_size: int,
    ) -> torch.Tensor:
        outputs = []
        for start in range(0, len(query), chunk_size):
            stop = min(len(query), start + chunk_size)
            chunk_metadata = None if metadata is None else {
                name: value[start:stop] for name, value in metadata.items()
            }
            outputs.append(
                self.reader(
                    query[start:stop],
                    None if support is None else support[start:stop],
                    None if distribution is None else distribution[start:stop],
                    None if valid is None else valid[start:stop],
                    mode=mode,
                    metadata=chunk_metadata,
                )
            )
        return torch.cat(outputs)

    def forward(
        self,
        pixel_feature: torch.Tensor,
        region_tokens: torch.Tensor,
        q: torch.Tensor,
        ids: torch.Tensor,
        shape: tuple[int, int],
        variant: str,
        support: torch.Tensor | None = None,
        distribution: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        metadata: Dict[str, torch.Tensor] | None = None,
        chunk_size: int = 256,
    ) -> Dict[str, torch.Tensor]:
        if variant not in VARIANTS:
            raise ValueError(variant)
        if pixel_feature.shape[0] != region_tokens.shape[0] or pixel_feature.shape[-2:] != shape:
            raise ValueError("Pixel/region feature-grid mismatch")
        if variant == "j0":
            enhanced = region_tokens.new_zeros(region_tokens.shape)
        else:
            if len(region_tokens) != 1:
                raise ValueError("Matched Memory experiment requires image batch size one")
            enhanced = self._read_chunks(
                region_tokens[0],
                "nomemory" if variant == "j1" else "memory",
                support,
                distribution,
                valid,
                metadata,
                chunk_size,
            )[None]
        context = unpool_region_tokens(enhanced, q, ids, shape)
        logits = self.classifier(pixel_feature, context)
        return {"logits": logits, "context": context, "enhanced_tokens": enhanced}

