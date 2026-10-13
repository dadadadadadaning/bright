"""A CPU-tested joint pixel / region / retrieved-context classifier prototype.

This file contains only the classification core, not a BRIGHT trainer or a
validated experiment. A caller supplies frozen, paired frontend features and
legal support rows selected without query ground truth. The exact same forward
    graph can read either regions in the current image or historical Train
supports. Retrieval, data splits, protocol checks and loss design belong to the
experiment runner, which is deliberately not implemented here.

All floating inputs are FP32. Q has shape [B, S, h*w]; ids has shape [S, h*w]
and stores the region ID attached to each of the S sparse assignments. The
assignment weights for each pixel sum to one. Regions need not be buildings.

The final head is zero-initialized, so its initial prediction equals the given
same-forward grid_probq probability up to the 1e-8 log clamp. It subsequently
learns a joint correction from pixel, region and support information. There is
no rejection threshold, confidence gate, learned scalar alpha or branch switch.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _float32_finite(name: str, value: Tensor) -> None:
    if value.dtype != torch.float32:
        raise ValueError(f"{name} must be float32; got {value.dtype}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def _validate_q(q: Tensor, ids: Tensor, batch: int, pixels: int, regions: int) -> None:
    _float32_finite("q", q)
    if q.ndim != 3 or q.shape[0] != batch or q.shape[2] != pixels:
        raise ValueError("q must have shape [B, S, h*w]")
    if q.shape[1] == 0 or ids.shape != q.shape[1:]:
        raise ValueError("ids must have shape [S, h*w] with S > 0")
    if ids.dtype != torch.long:
        raise ValueError("ids must be torch.long")
    if ids.device != q.device:
        raise ValueError("q and ids must share a device")
    if bool((ids < 0).any()) or bool((ids >= regions).any()):
        raise ValueError("ids contain an out-of-range region")
    if bool((q < 0).any()):
        raise ValueError("q weights must be nonnegative")
    if not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=2e-5, rtol=2e-5):
        raise ValueError("q weights must sum to one per pixel")


def sparse_unpool(region_values: Tensor, q: Tensor, ids: Tensor) -> Tensor:
    """Q-weighted region -> pixel mapping, returning [B, C, N].

    Duplicate region IDs contribute their individual assignment weights. This
    operation is not an inverse of pooling. Public model.forward validates Q;
    this small helper assumes its input dimensions and IDs are already valid.
    """
    batch, _, channels = region_values.shape
    pixels = ids.shape[1]
    result = region_values.new_zeros((batch, pixels, channels))
    for slot in range(ids.shape[0]):
        result = result + region_values[:, ids[slot], :] * q[:, slot, :, None]
    return result.transpose(1, 2)


def sparse_pool(pixel_values: Tensor, q: Tensor, ids: Tensor, regions: int) -> tuple[Tensor, Tensor]:
    """Q-weighted pixel mean [B, C, N] -> ([B, R, C], [B, R]).

    An empty region returns zero values and zero mass. The caller may use this
    to construct current-image support *predictions* without any query labels.
    """
    batch, channels, pixels = pixel_values.shape
    values = pixel_values.transpose(1, 2)
    total = pixel_values.new_zeros((batch, regions, channels))
    mass = pixel_values.new_zeros((batch, regions))
    for slot in range(ids.shape[0]):
        index = ids[slot][None, :].expand(batch, pixels)
        total = total.scatter_add(1, index[:, :, None].expand(-1, -1, channels),
                                  values * q[:, slot, :, None])
        mass = mass.scatter_add(1, index, q[:, slot, :])
    divisor = torch.where(mass > 0, mass, torch.ones_like(mass))
    return total / divisor[:, :, None], mass


class JointRegionMemoryClassifier(nn.Module):
    """Common classifier for source-matched current-image/Train-support arms.

    The caller chooses support IDs outside this model. For a current-image
    control, use distinct nearby regions and their predicted distributions;
    for Memory, use legal retrieved Train rows and their Train soft labels.
    Both use paired F256/T128 features and the identical encoder/attention/head.

    `chunk_size` bounds temporary encoded support tensors by that number of
    query regions. It does not remove the caller's allocated support tensors
    or attention activations retained for training; GPU memory must be measured
    in the actual trainer before making a fit/batch-size claim.
    """

    def __init__(self, chunk_size: int = 64, hidden: int = 128, heads: int = 4) -> None:
        super().__init__()
        if chunk_size < 1 or hidden < 1 or heads < 1 or hidden % heads:
            raise ValueError("chunk_size > 0 and hidden must be a positive multiple of heads")
        self.chunk_size = chunk_size
        self.hidden = hidden
        self.heads = heads
        self.region_encoder = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, hidden), nn.GELU())
        self.query_projection = nn.Linear(hidden, hidden)
        self.key_projection = nn.Linear(hidden, hidden)
        self.value_projection = nn.Linear(hidden, hidden)
        self.label_embedding = nn.Linear(4, hidden, bias=False)
        # No output bias: structurally absent supports give an exactly zero context.
        self.context_projection = nn.Linear(hidden, hidden, bias=False)
        self.pixel_projection = nn.Sequential(nn.Conv2d(256, 64, 1), nn.GELU())
        self.joint_head = nn.Sequential(
            nn.Conv2d(64 + 2 * hidden + 4, 128, 1), nn.GELU(), nn.Conv2d(128, 4, 1)
        )
        nn.init.zeros_(self.joint_head[-1].weight)
        nn.init.zeros_(self.joint_head[-1].bias)

    def _read(self, query: Tensor, support_f: Tensor, support_t: Tensor,
              support_labels: Tensor, support_valid: Tensor, chunk_size: int) -> Tensor:
        batch, regions, width = query.shape
        count = support_f.shape[2]
        dimension = width // self.heads
        outputs = []
        for start in range(0, regions, chunk_size):
            end = min(start + chunk_size, regions)
            valid = support_valid[:, start:end]
            # Mask before LayerNorm and learned projections, so NaN/Inf padding
            # cannot affect either outputs or parameter gradients.
            sf = torch.where(valid[..., None], support_f[:, start:end], 0.0)
            st = torch.where(valid[..., None], support_t[:, start:end], 0.0)
            labels = torch.where(valid[..., None], support_labels[:, start:end], 0.0)
            encoded = self.region_encoder(torch.cat((sf, st), dim=-1))
            q = self.query_projection(query[:, start:end]).reshape(batch, end-start, self.heads, dimension)
            k = self.key_projection(encoded).reshape(batch, end-start, count, self.heads, dimension)
            v = self.value_projection(encoded + self.label_embedding(labels)).reshape(
                batch, end-start, count, self.heads, dimension)
            scores = torch.einsum("brhd,brkhd->brhk", q, k) * (dimension ** -0.5)
            valid_heads = valid[:, :, None, :]
            scores = scores.masked_fill(~valid_heads, -torch.finfo(scores.dtype).max)
            attention = torch.softmax(scores, dim=-1) * valid_heads.to(scores.dtype)
            # All-invalid rows have a zero numerator. This is padding handling,
            # not a learned rejection decision or a confidence threshold.
            attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
            context = torch.einsum("brhk,brkhd->brhd", attention, v).reshape(batch, end-start, width)
            outputs.append(self.context_projection(context))
        return torch.cat(outputs, dim=1)

    def forward(self, pixel_features: Tensor, region_f: Tensor, region_t: Tensor,
                q: Tensor, ids: Tensor, base_probability_full: Tensor,
                support_f: Tensor, support_t: Tensor, support_labels: Tensor,
                support_valid: Tensor, *, chunk_size: int | None = None) -> Dict[str, Tensor]:
        """Produce full-resolution logits/probabilities and diagnostic tensors.

        Floating inputs: pixel_features [B,256,h,w], region_f [B,R,256],
        region_t [B,R,128], base_probability_full [B,4,H,W], support_f
        [B,R,K,256], support_t [B,R,K,128], support_labels [B,R,K,4].
        support_valid is boolean [B,R,K]. Invalid support padding may contain
        NaNs; valid rows must be finite. No query GT or event label is accepted.
        """
        if pixel_features.ndim != 4 or pixel_features.shape[1] != 256:
            raise ValueError("pixel_features must have shape [B,256,h,w]")
        batch, _, height, width = pixel_features.shape
        if min(batch, height, width) < 1:
            raise ValueError("pixel dimensions must be nonempty")
        if region_f.ndim != 3 or region_f.shape[0] != batch or region_f.shape[2] != 256:
            raise ValueError("region_f must have shape [B,R,256]")
        regions = region_f.shape[1]
        if regions < 1 or region_t.shape != (batch, regions, 128):
            raise ValueError("region_t must have shape [B,R,128] and R > 0")
        if base_probability_full.ndim != 4 or base_probability_full.shape[:2] != (batch, 4):
            raise ValueError("base_probability_full must have shape [B,4,H,W]")
        if min(base_probability_full.shape[2:]) < 1:
            raise ValueError("base_probability_full must have nonempty spatial dimensions")
        if support_valid.ndim != 3 or support_valid.shape[:2] != (batch, regions):
            raise ValueError("support_valid must have shape [B,R,K]")
        count = support_valid.shape[2]
        if support_valid.dtype != torch.bool or count < 1:
            raise ValueError("support_valid must be boolean, with K > 0; mask all slots for empty supports")
        for name, value, channels in (("support_f", support_f, 256), ("support_t", support_t, 128),
                                      ("support_labels", support_labels, 4)):
            if value.shape != (batch, regions, count, channels):
                raise ValueError(f"{name} has the wrong shape")
            if value.dtype != torch.float32:
                raise ValueError(f"{name} must be float32")
        all_inputs = (pixel_features, region_f, region_t, q, ids, base_probability_full,
                      support_f, support_t, support_labels, support_valid)
        if any(value.device != pixel_features.device for value in all_inputs):
            raise ValueError("all inputs must share a device")
        if self.query_projection.weight.device != pixel_features.device:
            raise ValueError("model and inputs must share a device")
        for name, value in (("pixel_features", pixel_features), ("region_f", region_f),
                            ("region_t", region_t), ("base_probability_full", base_probability_full)):
            _float32_finite(name, value)
        _validate_q(q, ids, batch, height * width, regions)
        for name, value in (("support_f", support_f), ("support_t", support_t),
                            ("support_labels", support_labels)):
            if not bool(torch.isfinite(value[support_valid]).all()):
                raise ValueError(f"valid {name} contains non-finite values")
        labels = support_labels[support_valid]
        if bool((labels < 0).any()) or not torch.allclose(labels.sum(-1), torch.ones_like(labels[:, 0]),
                                                        atol=2e-5, rtol=2e-5):
            raise ValueError("valid support_labels must be nonnegative probability distributions")
        if bool((base_probability_full < 0).any()) or not torch.allclose(
                base_probability_full.sum(1), torch.ones_like(base_probability_full[:, 0]),
                atol=2e-5, rtol=2e-5):
            raise ValueError("base_probability_full must contain normalized nonnegative probabilities")
        chunk = self.chunk_size if chunk_size is None else chunk_size
        if not isinstance(chunk, int) or chunk < 1:
            raise ValueError("chunk_size must be a positive integer")
        region_encoding = self.region_encoder(torch.cat((region_f, region_t), dim=-1))
        memory_context = self._read(region_encoding, support_f, support_t, support_labels, support_valid, chunk)
        region_grid = sparse_unpool(region_encoding, q, ids).reshape(batch, self.hidden, height, width)
        memory_grid = sparse_unpool(memory_context, q, ids).reshape(batch, self.hidden, height, width)
        base_grid = F.interpolate(base_probability_full, size=(height, width), mode="bilinear", align_corners=False)
        joint = torch.cat((self.pixel_projection(pixel_features), region_grid, memory_grid, base_grid), dim=1)
        delta_grid = self.joint_head(joint)
        delta_full = F.interpolate(delta_grid, size=base_probability_full.shape[-2:], mode="bilinear", align_corners=False)
        logits = base_probability_full.clamp_min(1e-8).log() + delta_full
        return {"logits": logits, "probabilities": torch.softmax(logits, dim=1),
                "delta_grid": delta_grid, "region_encoding": region_encoding,
                "memory_context": memory_context, "base_probability_grid": base_grid}
