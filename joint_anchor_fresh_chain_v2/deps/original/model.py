"""Joint pixel / parent-region / dual-slot / reference-case classifier.

Retrieval is external and frozen. Each selected parent expands into its two
role-tagged slots. The support table is encoded once, then gathered in bounded
query chunks; no [all queries, all candidates, raw 384-D] expansion is built.
Slot summaries and Reader outputs both return through their own routing
weights. Parent context returns through original Q. There is no Memory
accept/reject gate. Structural padding masks only remove absent evidence.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from .slots import finite_fp32, sparse_unpool, validate_routes
except ImportError:
    from slots import finite_fp32, sparse_unpool, validate_routes


class JointRegionMemoryClassifier(nn.Module):
    """Batch-one classifier shared identically by all four matched arms.

    T128 is explicitly a *parent* context, not a fabricated child token. Slot
    0/1 are learned role embeddings, with no constraint on their output class.
    The caller supplies current-image predicted labels or Train support labels
    through the same ``support_labels`` interface; query GT is not accepted.
    """

    def __init__(self, chunk_size: int = 32, hidden: int = 128, heads: int = 4) -> None:
        super().__init__()
        if min(chunk_size, hidden, heads) < 1 or hidden % heads:
            raise ValueError("positive chunk_size/hidden/heads and hidden divisible by heads required")
        self.chunk_size, self.hidden, self.heads = chunk_size, hidden, heads
        self.region_encoder = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, hidden), nn.GELU())
        self.slot_feature_encoder = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, hidden))
        self.parent_context_encoder = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, hidden))
        self.role_embedding = nn.Embedding(2, hidden)
        self.fraction_embedding = nn.Linear(1, hidden, bias=False)
        self.slot_normalization = nn.Sequential(nn.LayerNorm(hidden), nn.GELU())
        self.query_projection = nn.Linear(hidden, hidden)
        self.key_projection = nn.Linear(hidden, hidden)
        self.value_projection = nn.Linear(hidden, hidden)
        self.label_embedding = nn.Linear(4, hidden, bias=False)
        self.context_projection = nn.Linear(hidden, hidden, bias=False)
        self.pixel_projection = nn.Sequential(nn.Conv2d(256, 64, 1), nn.GELU())
        self.joint_head = nn.Sequential(nn.Conv2d(64 + 3 * hidden + 4, 128, 1), nn.GELU(), nn.Conv2d(128, 4, 1))
        nn.init.zeros_(self.joint_head[-1].weight)
        nn.init.zeros_(self.joint_head[-1].bias)

    def _encode_slots(self, features: Tensor, parent_t: Tensor,
                      fraction: Tensor, valid: Tensor) -> Tensor:
        clean_f = torch.where(valid[..., None], features, 0.0)
        clean_t = torch.where(valid.any(-1)[..., None], parent_t, 0.0)
        clean_fraction = torch.where(valid, fraction, 0.0)
        encoded = (self.slot_feature_encoder(clean_f)
                   + self.parent_context_encoder(clean_t)[..., None, :]
                   + self.role_embedding.weight
                   + self.fraction_embedding(clean_fraction[..., None]))
        encoded = self.slot_normalization(encoded)
        return torch.where(valid[..., None], encoded, 0.0)

    def _read(self, query: Tensor, support_ids: Tensor, table_k: Tensor,
              table_v: Tensor, table_valid: Tensor, query_valid: Tensor,
              chunk_size: int) -> Tensor:
        # query: [R,2,H]; support_ids: [R,K], tables: [M,2,H].
        regions, _, width = query.shape
        if table_k.shape[0] == 0:
            return query * 0.0
        dimension = width // self.heads
        outputs = []
        for start in range(0, regions, chunk_size):
            end = min(start + chunk_size, regions)
            fathers = support_ids[start:end]
            safe = fathers.clamp_min(0)
            valid = (fathers[..., None] >= 0) & table_valid[safe]
            valid = valid.flatten(1)  # [chunk, K*2], same candidates for both query roles.
            k = table_k[safe].reshape(end-start, -1, self.heads, dimension)
            v = table_v[safe].reshape(end-start, -1, self.heads, dimension)
            q = self.query_projection(query[start:end]).reshape(end-start, 2, self.heads, dimension)
            scores = torch.einsum("rqhd,rkhd->rqhk", q, k) * (dimension ** -0.5)
            mask = valid[:, None, None, :]
            scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)
            attention = torch.softmax(scores, -1) * mask.to(scores.dtype)
            denominator = attention.sum(-1, keepdim=True)
            attention = attention / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
            context = torch.einsum("rqhk,rkhd->rqhd", attention, v).reshape(end-start, 2, width)
            context = self.context_projection(context)
            outputs.append(torch.where(query_valid[start:end, :, None], context, 0.0))
        return torch.cat(outputs, 0)

    def forward(self, pixel_features: Tensor, region_f: Tensor, region_t: Tensor,
                q: Tensor, ids: Tensor, base_probability_full: Tensor,
                slot_f: Tensor, slot_fraction: Tensor, slot_valid: Tensor,
                route_q: Tensor, route_ids: Tensor, support_ids: Tensor,
                support_f: Tensor, support_t: Tensor, support_labels: Tensor,
                support_fraction: Tensor, support_valid: Tensor, *,
                chunk_size: int | None = None) -> dict[str, Tensor]:
        if pixel_features.ndim != 4 or pixel_features.shape[:2] != (1, 256) or min(pixel_features.shape) < 1:
            raise ValueError("pixel_features must be nonempty [1,256,h,w]; batch-one experiment")
        _, _, height, width = pixel_features.shape
        if region_f.ndim != 3 or region_f.shape[0] != 1 or region_f.shape[2] != 256 or region_f.shape[1] < 1:
            raise ValueError("region_f must be [1,R,256]")
        regions = region_f.shape[1]
        if region_t.shape != (1, regions, 128) or slot_f.shape != (1, regions, 2, 256):
            raise ValueError("region_t/slot_f shape mismatch")
        if slot_fraction.shape != (1, regions, 2) or slot_valid.shape != slot_fraction.shape or slot_valid.dtype != torch.bool:
            raise ValueError("slot fraction/boolean validity shape mismatch")
        if base_probability_full.ndim != 4 or base_probability_full.shape[:2] != (1, 4) or min(base_probability_full.shape) < 1:
            raise ValueError("base_probability_full must be nonempty [1,4,H,W]")
        if support_valid.ndim != 2 or support_valid.shape[1] != 2 or support_valid.dtype != torch.bool:
            raise ValueError("support_valid must be boolean [M,2]")
        members = support_valid.shape[0]
        if (support_f.shape != (members, 2, 256) or support_t.shape != (members, 128)
                or support_labels.shape != (members, 2, 4) or support_fraction.shape != (members, 2)):
            raise ValueError("support table shape mismatch")
        if support_ids.ndim != 2 or support_ids.shape[0] != regions or support_ids.shape[1] < 1 or support_ids.dtype != torch.long:
            raise ValueError("support_ids must be int64 [R,K], with -1 padding")
        if bool(((support_ids < -1) | (support_ids >= members)).any()):
            raise ValueError("support_ids out of range")
        inputs = (pixel_features, region_f, region_t, q, ids, base_probability_full,
                  slot_f, slot_fraction, slot_valid, route_q, route_ids, support_ids,
                  support_f, support_t, support_labels, support_fraction, support_valid)
        if any(value.device != pixel_features.device for value in inputs) or self.query_projection.weight.device != pixel_features.device:
            raise ValueError("all model inputs/parameters must share a device")
        for name, value in (("pixel_features", pixel_features), ("region_f", region_f),
                            ("region_t", region_t), ("base_probability_full", base_probability_full),
                            ("slot_f", slot_f), ("slot_fraction", slot_fraction)):
            finite_fp32(name, value)
        parent_ids = validate_routes(q, ids, regions, 1, height * width)
        child_ids = validate_routes(route_q, route_ids, 2 * regions, 1, height * width)
        candidates = q.shape[1]
        if candidates != 9 or route_q.shape[1] != 18:
            raise ValueError("expected 9 parent and 18 routed candidates")
        if (not torch.equal(child_ids[:, :candidates], 2 * parent_ids)
                or not torch.equal(child_ids[:, candidates:], 2 * parent_ids + 1)
                or not torch.allclose(route_q[:, :candidates] + route_q[:, candidates:], q, atol=2e-5, rtol=2e-5)):
            raise ValueError("child routes must preserve parent IDs and assignment mass")
        for name, value in (("support_f", support_f), ("support_labels", support_labels), ("support_fraction", support_fraction)):
            if value.dtype != torch.float32 or not bool(torch.isfinite(value[support_valid]).all()):
                raise ValueError(f"valid {name} must be finite float32")
        if support_t.dtype != torch.float32 or not bool(torch.isfinite(support_t[support_valid.any(-1)]).all()):
            raise ValueError("valid support parent T128 must be finite float32")
        if bool(((slot_fraction < 0) | (slot_fraction > 1)).any()) or bool(((support_fraction[support_valid] < 0) | (support_fraction[support_valid] > 1)).any()):
            raise ValueError("slot fractions must be in [0,1]")
        if not torch.equal(slot_valid, slot_fraction > 0):
            raise ValueError("query slot validity must match positive slot fraction")
        labels = support_labels[support_valid]
        if bool((labels < 0).any()) or not torch.allclose(labels.sum(-1), torch.ones_like(labels[:, 0]), atol=2e-5, rtol=2e-5):
            raise ValueError("valid support labels must be nonnegative normalized distributions")
        if bool((base_probability_full < 0).any()) or not torch.allclose(
            base_probability_full.sum(1), torch.ones_like(base_probability_full[:, 0]), atol=2e-5, rtol=2e-5
        ):
            raise ValueError("base probabilities must be nonnegative and sum to one")
        chunk = self.chunk_size if chunk_size is None else chunk_size
        if not isinstance(chunk, int) or chunk < 1:
            raise ValueError("chunk_size must be a positive integer")
        parent_encoding = self.region_encoder(torch.cat((region_f, region_t), -1))
        slot_encoding = self._encode_slots(slot_f[0], region_t[0], slot_fraction[0], slot_valid[0])
        # Encode each support only once, before gathering candidate rows.
        table = self._encode_slots(support_f, support_t, support_fraction, support_valid)
        clean_labels = torch.where(support_valid[..., None], support_labels, 0.0)
        table_k = self.key_projection(table)
        table_v = self.value_projection(table + self.label_embedding(clean_labels))
        memory_context = self._read(slot_encoding, support_ids, table_k, table_v, support_valid, slot_valid[0], chunk)
        parent_grid = sparse_unpool(parent_encoding, q, ids).reshape(1, self.hidden, height, width)
        slot_grid = sparse_unpool(slot_encoding.reshape(1, 2 * regions, self.hidden), route_q, route_ids).reshape(1, self.hidden, height, width)
        memory_grid = sparse_unpool(memory_context.reshape(1, 2 * regions, self.hidden), route_q, route_ids).reshape(1, self.hidden, height, width)
        base_grid = F.interpolate(base_probability_full, size=(height, width), mode="bilinear", align_corners=False)
        joint = torch.cat((self.pixel_projection(pixel_features), parent_grid, slot_grid, memory_grid, base_grid), 1)
        delta_grid = self.joint_head(joint)
        delta_full = F.interpolate(delta_grid, size=base_probability_full.shape[-2:], mode="bilinear", align_corners=False)
        logits = base_probability_full.clamp_min(1e-8).log() + delta_full
        return dict(logits=logits, probabilities=logits.softmax(1), delta_grid=delta_grid,
                    region_encoding=parent_encoding, slot_encoding=slot_encoding[None],
                    memory_context=memory_context[None], base_probability_grid=base_grid)
