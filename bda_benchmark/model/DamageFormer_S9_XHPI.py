# -*- coding: utf-8 -*-
"""S9-XHPI core: cross-modal hierarchical prototype-instance memory.

The module is deliberately split from the BRIGHT pixel backbone.  S8 supplies
RGB, SAR and fused feature maps; S9 learns a retrieval metric, builds a
train-only memory, routes queries through class-conditioned change prototypes,
and retrieves real building instances.  Prototypes are an index, never a
second classifier added directly to the pixel logits.

The raw building record contains:
    rgb_structure, sar_global, sar_ring, change_global, local_tokens,
    label (0/1/2), event, base_probs and sample_id.

Only invariant/shared RGB--SAR features receive an alignment loss.  Damage is
represented in a separate change branch, preventing alignment from erasing the
post-event SAR evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_DAMAGE_CLASSES = 3


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-6)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass(frozen=True)
class MetricConfig:
    raw_dim: int = 128
    embed_dim: int = 128
    hidden_dim: int = 256
    temperature: float = 0.10

    def validate(self) -> None:
        if min(self.raw_dim, self.embed_dim, self.hidden_dim) <= 0:
            raise ValueError("metric dimensions must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


class CrossModalMetricProjector(nn.Module):
    """Shared/private-aware building metric used by S9-X1 and later stages."""

    def __init__(self, config: MetricConfig = MetricConfig()):
        super().__init__()
        config.validate()
        self.config = config
        d, e, h = config.raw_dim, config.embed_dim, config.hidden_dim

        # The two shared projections are aligned only where the target is
        # expected to be stable.  They are not themselves damage classifiers.
        self.rgb_shared = _mlp(d, h, e)
        self.sar_shared = _mlp(d, h, e)
        self.structure_proj = _mlp(d, h, e)
        self.sar_proj = _mlp(2 * d, h, e)
        self.local_proj = _mlp(d, h, e)

        # change_global remains explicit; abs/product encode cross-modal
        # relation while sar_ring preserves modality-specific damage evidence.
        self.change_proj = _mlp(d + 3 * e, h, e)
        self.exist_head = nn.Linear(e, 2)
        self.severity_head = nn.Linear(e, 2)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        required = ("rgb_structure", "sar_global", "sar_ring", "change_global")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"missing raw descriptor fields: {missing}")
        rgb = batch["rgb_structure"].float()
        sar_global = batch["sar_global"].float()
        sar_ring = batch["sar_ring"].float()
        change_global = batch["change_global"].float()
        expected = self.config.raw_dim
        for name, value in (
            ("rgb_structure", rgb),
            ("sar_global", sar_global),
            ("sar_ring", sar_ring),
            ("change_global", change_global),
        ):
            if value.ndim != 2 or value.shape[1] != expected:
                raise ValueError(f"{name} must be [N,{expected}], got {tuple(value.shape)}")

        shared_rgb = _normalize(self.rgb_shared(rgb))
        shared_sar = _normalize(self.sar_shared(sar_global))
        structure = _normalize(self.structure_proj(rgb))
        sar = _normalize(self.sar_proj(torch.cat((sar_global, sar_ring), dim=1)))
        change_input = torch.cat(
            (
                change_global,
                torch.abs(shared_sar - shared_rgb),
                shared_sar * shared_rgb,
                _normalize(self.sar_shared(sar_ring)),
            ),
            dim=1,
        )
        change = _normalize(self.change_proj(change_input))
        output = {
            "shared_rgb": shared_rgb,
            "shared_sar": shared_sar,
            "structure": structure,
            "sar": sar,
            "change": change,
            "exist_logits": self.exist_head(change),
            "severity_logits": self.severity_head(change),
        }
        if "local_summary" in batch:
            output["local"] = self.project_local(batch["local_summary"])
        return output

    def project_local(self, local_tokens: torch.Tensor) -> torch.Tensor:
        if local_tokens.ndim != 2 or local_tokens.shape[1] != self.config.raw_dim:
            raise ValueError(
                f"local_tokens must be [L,{self.config.raw_dim}], "
                f"got {tuple(local_tokens.shape)}"
            )
        return _normalize(self.local_proj(local_tokens.float()))


def hierarchical_log_probs(
    exist_logits: torch.Tensor,
    severity_logits: torch.Tensor,
) -> torch.Tensor:
    if exist_logits.ndim != 2 or exist_logits.shape[1] != 2:
        raise ValueError("exist logits must be [N,2]")
    if severity_logits.shape != exist_logits.shape:
        raise ValueError("severity logits must match exist logits")
    exist = F.log_softmax(exist_logits, dim=1)
    severity = F.log_softmax(severity_logits, dim=1)
    return torch.stack(
        (
            exist[:, 0],
            exist[:, 1] + severity[:, 0],
            exist[:, 1] + severity[:, 1],
        ),
        dim=1,
    )


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    events: Optional[torch.Tensor] = None,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Supervised contrast with optional different-event positive constraint."""

    if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings/labels have incompatible shapes")
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0
    z = _normalize(embeddings)
    logits = z @ z.t() / temperature
    eye = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None, :]) & ~eye
    if events is not None:
        if events.shape != labels.shape:
            raise ValueError("events must match labels")
        positives &= events[:, None].ne(events[None, :])
    valid_anchor = positives.any(dim=1)
    if not bool(valid_anchor.any().item()):
        return embeddings.sum() * 0.0
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (~eye)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_positive = (log_prob * positives).sum(dim=1) / positives.sum(dim=1).clamp_min(1)
    return -mean_positive[valid_anchor].mean()


def metric_training_loss(
    outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    event_ids: torch.Tensor,
    lambda_align: float = 0.10,
    lambda_exist_contrast: float = 0.20,
    lambda_severity_contrast: float = 0.20,
    lambda_local_contrast: float = 0.10,
    temperature: float = 0.10,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Hierarchical class loss plus cross-event metric and safe alignment."""

    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("labels must be a non-empty [N] tensor")
    log_probs = hierarchical_log_probs(outputs["exist_logits"], outputs["severity_logits"])
    class_weight = log_probs.new_tensor((1.0, 1.5, 1.5))
    classification = F.nll_loss(log_probs, labels, weight=class_weight)

    exist_labels = (labels > 0).long()
    exist_contrast = supervised_contrastive_loss(
        outputs["change"], exist_labels, event_ids, temperature
    )
    damaged = labels > 0
    if int(damaged.sum()) >= 2:
        severity_contrast = supervised_contrastive_loss(
            outputs["change"][damaged], labels[damaged] - 1,
            event_ids[damaged], temperature,
        )
    else:
        severity_contrast = outputs["change"].sum() * 0.0
    if "local" in outputs:
        local_contrast = supervised_contrastive_loss(
            outputs["local"], labels, event_ids, temperature
        )
    else:
        local_contrast = outputs["change"].sum() * 0.0

    # Intact is expected to be most invariant.  Damage classes get weak
    # alignment only, so post-event evidence is not erased.
    align_weight = torch.where(
        labels == 0,
        torch.ones_like(labels, dtype=torch.float32),
        torch.where(
            labels == 1,
            torch.full_like(labels, 0.25, dtype=torch.float32),
            torch.full_like(labels, 0.10, dtype=torch.float32),
        ),
    ).to(outputs["shared_rgb"].device)
    align_distance = 1.0 - (outputs["shared_rgb"] * outputs["shared_sar"]).sum(dim=1)
    alignment = (align_distance * align_weight).sum() / align_weight.sum().clamp_min(1e-6)

    total = (
        classification
        + lambda_align * alignment
        + lambda_exist_contrast * exist_contrast
        + lambda_severity_contrast * severity_contrast
        + lambda_local_contrast * local_contrast
    )
    prediction = log_probs.argmax(dim=1)
    return total, {
        "total": float(total.detach().item()),
        "classification": float(classification.detach().item()),
        "alignment": float(alignment.detach().item()),
        "exist_contrast": float(exist_contrast.detach().item()),
        "severity_contrast": float(severity_contrast.detach().item()),
        "local_contrast": float(local_contrast.detach().item()),
        "accuracy": float((prediction == labels).float().mean().item()),
    }


@torch.no_grad()
def spherical_kmeans(
    features: torch.Tensor,
    clusters: int,
    iterations: int = 30,
    seed: int = 3407,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Small dependency-free spherical k-means for offline prototype building."""

    features = _normalize(features.cpu())
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty matrix")
    clusters = min(int(clusters), int(features.shape[0]))
    if clusters <= 0:
        raise ValueError("clusters must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(features.shape[0], (1,), generator=generator).item())
    selected = [first]
    min_distance = 1.0 - features @ features[first]
    for _ in range(1, clusters):
        index = int(min_distance.argmax().item())
        selected.append(index)
        min_distance = torch.minimum(min_distance, 1.0 - features @ features[index])
    centers = features[torch.tensor(selected)].clone()
    assignment = torch.zeros(features.shape[0], dtype=torch.long)
    for _ in range(iterations):
        new_assignment = (features @ centers.t()).argmax(dim=1)
        if torch.equal(new_assignment, assignment) and _ > 0:
            break
        assignment = new_assignment
        updated = []
        for index in range(clusters):
            members = features[assignment == index]
            if members.numel() == 0:
                updated.append(centers[index])
            else:
                updated.append(_normalize(members.mean(dim=0, keepdim=True))[0])
        centers = torch.stack(updated)
    return centers, assignment


@torch.no_grad()
def greedy_kcenter(features: torch.Tensor, count: int, seed: int = 3407) -> torch.Tensor:
    """Select a diversity coreset by greedy farthest-point sampling."""

    features = _normalize(features.cpu())
    count = min(max(0, int(count)), int(features.shape[0]))
    if count == 0:
        return torch.empty((0,), dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(features.shape[0], (1,), generator=generator).item())
    selected = [first]
    minimum = 1.0 - features @ features[first]
    for _ in range(1, count):
        index = int(minimum.argmax().item())
        selected.append(index)
        minimum = torch.minimum(minimum, 1.0 - features @ features[index])
    return torch.tensor(selected, dtype=torch.long)


def prototype_diversity_loss(
    prototypes: torch.Tensor,
    prototype_labels: torch.Tensor,
    margin: float = 0.80,
) -> torch.Tensor:
    """Penalty used for diagnostics or learnable-prototype ablations."""

    prototypes = _normalize(prototypes)
    similarity = prototypes @ prototypes.t()
    same = prototype_labels[:, None].eq(prototype_labels[None, :])
    same.fill_diagonal_(False)
    if not bool(same.any().item()):
        return prototypes.sum() * 0.0
    return F.relu(similarity[same] - margin).mean()


def _gather_local_tokens(
    flat_tokens: torch.Tensor,
    offsets: torch.Tensor,
    indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    chunks: List[torch.Tensor] = []
    new_offsets = [0]
    for index in indices.tolist():
        start, end = int(offsets[index]), int(offsets[index + 1])
        chunk = flat_tokens[start:end]
        chunks.append(chunk)
        new_offsets.append(new_offsets[-1] + int(chunk.shape[0]))
    output = torch.cat(chunks, dim=0) if chunks else flat_tokens[:0]
    return output, torch.tensor(new_offsets, dtype=torch.long)


@torch.no_grad()
def build_hierarchical_memory(
    raw_payload: Mapping[str, object],
    projector: CrossModalMetricProjector,
    prototypes_per_class: int = 4,
    max_per_event_class: int = 64,
    hard_fraction: float = 0.20,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, object]:
    """Project, coreset-select and index train-only raw building records."""

    tensor_keys = ("rgb_structure", "sar_global", "sar_ring", "change_global")
    raw = {key: raw_payload[key] for key in tensor_keys}
    if not all(isinstance(value, torch.Tensor) for value in raw.values()):
        raise TypeError("raw descriptor tensors are missing")
    labels = raw_payload["labels"]
    events = raw_payload["events"]
    base_probs = raw_payload["base_probs"]
    if not isinstance(labels, torch.Tensor) or not isinstance(base_probs, torch.Tensor):
        raise TypeError("labels and base_probs must be tensors")
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise TypeError("events must be a sequence")

    projector = projector.to(device).eval()
    projected_chunks: Dict[str, List[torch.Tensor]] = {
        key: [] for key in ("structure", "sar", "change")
    }
    batch_size = 1024
    count = labels.shape[0]
    for start in range(0, count, batch_size):
        end = min(count, start + batch_size)
        batch = {key: value[start:end].to(device) for key, value in raw.items()}
        output = projector(batch)
        for key in projected_chunks:
            projected_chunks[key].append(output[key].cpu())
    projected = {key: torch.cat(value, dim=0) for key, value in projected_chunks.items()}

    # Event/class cap with explicit hard-example reservation.
    selected: List[int] = []
    unique_events = sorted({str(event) for event in events})
    for class_index in range(NUM_DAMAGE_CLASSES):
        for event_index, event in enumerate(unique_events):
            group = torch.tensor(
                [i for i, item in enumerate(events) if str(item) == event and int(labels[i]) == class_index],
                dtype=torch.long,
            )
            if group.numel() == 0:
                continue
            cap = min(max_per_event_class, int(group.numel()))
            hard_count = min(int(round(cap * hard_fraction)), cap)
            predictions = base_probs[group].argmax(dim=1)
            hard_pool = group[predictions != class_index]
            if hard_pool.numel() > 0 and hard_count > 0:
                # Least confident failures first.
                confidence = base_probs[hard_pool, class_index]
                hard = hard_pool[confidence.argsort()[:hard_count]]
            else:
                hard = group[:0]
            remaining_mask = ~torch.isin(group, hard)
            remaining = group[remaining_mask]
            cover_count = cap - int(hard.numel())
            cover_local = greedy_kcenter(
                projected["change"][remaining], cover_count,
                seed=3407 + 101 * class_index + event_index,
            )
            chosen = torch.cat((hard, remaining[cover_local]), dim=0)
            selected.extend(int(index) for index in chosen.tolist())

    selected_index = torch.tensor(sorted(set(selected)), dtype=torch.long)
    if selected_index.numel() == 0:
        raise RuntimeError("no memory instances survived coreset selection")

    memory_labels = labels[selected_index].long().cpu()
    prototypes: List[torch.Tensor] = []
    prototype_labels: List[int] = []
    assignment_global = torch.empty(selected_index.shape[0], dtype=torch.long)
    prototype_offset = 0
    for class_index in range(NUM_DAMAGE_CLASSES):
        positions = torch.nonzero(memory_labels == class_index, as_tuple=False)[:, 0]
        if positions.numel() == 0:
            raise RuntimeError(f"memory has no class {class_index} entries")
        centers, local_assignment = spherical_kmeans(
            projected["change"][selected_index[positions]],
            prototypes_per_class,
            seed=3407 + class_index,
        )
        prototypes.append(centers)
        prototype_labels.extend([class_index] * centers.shape[0])
        assignment_global[positions] = local_assignment + prototype_offset
        prototype_offset += centers.shape[0]

    local_flat = raw_payload["local_tokens"]
    local_offsets = raw_payload["local_offsets"]
    if not isinstance(local_flat, torch.Tensor) or not isinstance(local_offsets, torch.Tensor):
        raise TypeError("local_tokens/local_offsets are required")
    selected_local_raw, selected_local_offsets = _gather_local_tokens(
        local_flat, local_offsets, selected_index
    )
    local_projected_parts: List[torch.Tensor] = []
    for start in range(0, selected_local_raw.shape[0], 4096):
        local_projected_parts.append(
            projector.project_local(selected_local_raw[start : start + 4096].to(device)).cpu()
        )
    local_projected = (
        torch.cat(local_projected_parts, dim=0)
        if local_projected_parts else torch.empty((0, projector.config.embed_dim))
    )
    sample_ids = list(raw_payload.get("sample_ids", [str(i) for i in range(count)]))
    selected_events = [str(events[i]) for i in selected_index.tolist()]
    selected_samples = [str(sample_ids[i]) for i in selected_index.tolist()]
    prototype_tensor = torch.cat(prototypes, dim=0)
    prototype_label_tensor = torch.tensor(prototype_labels, dtype=torch.long)
    usage = torch.bincount(assignment_global, minlength=prototype_tensor.shape[0]).float()

    return {
        "format_version": 1,
        "experiment": "S9_XHPI",
        "projector_config": asdict(projector.config),
        "structure": projected["structure"][selected_index],
        "sar": projected["sar"][selected_index],
        "change": projected["change"][selected_index],
        "labels": memory_labels,
        "events": selected_events,
        "sample_ids": selected_samples,
        "base_probs": base_probs[selected_index].float().cpu(),
        "local_tokens": local_projected,
        "local_offsets": selected_local_offsets,
        "prototypes": prototype_tensor,
        "prototype_labels": prototype_label_tensor,
        "prototype_assignment": assignment_global,
        "prototype_usage": usage,
        "selected_raw_indices": selected_index,
        "metadata": {
            "source_split": "train_only",
            "raw_count": int(count),
            "memory_count": int(selected_index.numel()),
            "prototypes_per_class_requested": int(prototypes_per_class),
            "max_per_event_class": int(max_per_event_class),
            "hard_fraction": float(hard_fraction),
            **dict(raw_payload.get("metadata", {})),
        },
    }


@dataclass(frozen=True)
class RetrievalConfig:
    route_top_prototypes: int = 4
    rerank_candidates: int = 64
    topk: int = 8
    weight_change: float = 0.40
    weight_sar: float = 0.25
    weight_structure: float = 0.10
    weight_local: float = 0.25
    beta: float = 8.0
    similarity_threshold: float = 0.65
    purity_threshold: float = 0.70
    retrieval_margin_threshold: float = 0.10
    agreement_threshold: float = 0.45
    base_margin_max: float = 0.35
    minimum_events: int = 2
    max_neighbors_per_event: int = 3
    fusion_alpha: float = 0.15

    def validate(self) -> None:
        if min(self.route_top_prototypes, self.rerank_candidates, self.topk) <= 0:
            raise ValueError("routing and top-k values must be positive")
        weights = (
            self.weight_change,
            self.weight_sar,
            self.weight_structure,
            self.weight_local,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("retrieval weights must be non-negative with positive sum")
        if self.beta <= 0 or not 0 <= self.fusion_alpha <= 1:
            raise ValueError("invalid beta/fusion_alpha")


@dataclass
class RetrievalResult:
    probabilities: torch.Tensor
    prototype_probabilities: torch.Tensor
    neighbor_indices: torch.Tensor
    neighbor_scores: torch.Tensor
    gate: float
    active: bool
    purity: float
    margin: float
    best_similarity: float
    agreement: float
    event_count: int


class HierarchicalPrototypeInstanceMemory:
    """Prototype routing followed by multimodal real-instance reranking."""

    def __init__(self, payload: Mapping[str, object], device: torch.device):
        required = (
            "structure", "sar", "change", "labels", "local_tokens", "local_offsets",
            "prototypes", "prototype_labels", "prototype_assignment",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"memory is missing fields: {missing}")
        self.device = device
        self.structure = _normalize(payload["structure"].to(device))  # type: ignore[union-attr]
        self.sar = _normalize(payload["sar"].to(device))  # type: ignore[union-attr]
        self.change = _normalize(payload["change"].to(device))  # type: ignore[union-attr]
        self.labels = payload["labels"].long().to(device)  # type: ignore[union-attr]
        self.local_tokens = _normalize(payload["local_tokens"].to(device))  # type: ignore[union-attr]
        self.local_offsets = payload["local_offsets"].long().cpu()  # type: ignore[union-attr]
        self.prototypes = _normalize(payload["prototypes"].to(device))  # type: ignore[union-attr]
        self.prototype_labels = payload["prototype_labels"].long().to(device)  # type: ignore[union-attr]
        self.prototype_assignment = payload["prototype_assignment"].long().to(device)  # type: ignore[union-attr]
        self.events = [str(item) for item in payload.get("events", [])]
        if len(self.events) != self.labels.numel():
            raise ValueError("event metadata length does not match memory")

    def _local_score(self, query_tokens: torch.Tensor, memory_index: int) -> torch.Tensor:
        start = int(self.local_offsets[memory_index])
        end = int(self.local_offsets[memory_index + 1])
        target = self.local_tokens[start:end]
        if query_tokens.numel() == 0 or target.numel() == 0:
            return self.change.new_tensor(0.0)
        similarity = _normalize(query_tokens) @ target.t()
        # Symmetric Chamfer-style set matching avoids a single query peak
        # explaining an entire damaged building.
        return 0.5 * (similarity.max(dim=1).values.mean() + similarity.max(dim=0).values.mean())

    def search(
        self,
        query: Mapping[str, torch.Tensor],
        base_probabilities: torch.Tensor,
        config: RetrievalConfig,
        query_event: Optional[str] = None,
    ) -> RetrievalResult:
        config.validate()
        change = _normalize(query["change"].reshape(1, -1))[0]
        sar = _normalize(query["sar"].reshape(1, -1))[0]
        structure = _normalize(query["structure"].reshape(1, -1))[0]
        local = _normalize(query["local"])
        base = base_probabilities.float().to(self.device)
        base = base / base.sum().clamp_min(1e-8)

        proto_similarity = change @ self.prototypes.t()
        route_count = min(config.route_top_prototypes, self.prototypes.shape[0])
        routed = torch.topk(proto_similarity, k=route_count).indices
        candidate_mask = torch.isin(self.prototype_assignment, routed)
        if query_event is not None:
            keep = torch.tensor(
                [event != str(query_event) for event in self.events],
                device=self.device,
                dtype=torch.bool,
            )
            candidate_mask &= keep
        candidates = torch.nonzero(candidate_mask, as_tuple=False)[:, 0]
        if candidates.numel() < config.topk:
            candidates = torch.arange(self.labels.numel(), device=self.device)
            if query_event is not None:
                keep = torch.tensor(
                    [event != str(query_event) for event in self.events],
                    device=self.device,
                    dtype=torch.bool,
                )
                candidates = candidates[keep]
        if candidates.numel() == 0:
            zero = torch.full((NUM_DAMAGE_CLASSES,), 1.0 / NUM_DAMAGE_CLASSES, device=self.device)
            return RetrievalResult(zero, zero, candidates, candidates.float(), 0.0, False, 0, 0, 0, 0, 0)

        weight_total = config.weight_change + config.weight_sar + config.weight_structure + config.weight_local
        coarse = (
            config.weight_change * (self.change[candidates] @ change)
            + config.weight_sar * (self.sar[candidates] @ sar)
            + config.weight_structure * (self.structure[candidates] @ structure)
        ) / weight_total
        rerank_count = min(config.rerank_candidates, candidates.numel())
        rerank_pos = torch.topk(coarse, k=rerank_count).indices
        candidates = candidates[rerank_pos]
        coarse = coarse[rerank_pos]
        if config.weight_local > 0:
            local_scores = torch.stack(
                [self._local_score(local, int(index)) for index in candidates.tolist()]
            )
            scores = coarse + config.weight_local * local_scores / weight_total
        else:
            scores = coarse

        # Event cap prevents one disaster style from monopolizing Top-K.
        order = scores.argsort(descending=True)
        chosen_positions: List[int] = []
        event_use: Dict[str, int] = {}
        for position in order.tolist():
            memory_index = int(candidates[position])
            event = self.events[memory_index]
            if event_use.get(event, 0) >= config.max_neighbors_per_event:
                continue
            chosen_positions.append(position)
            event_use[event] = event_use.get(event, 0) + 1
            if len(chosen_positions) >= config.topk:
                break
        if not chosen_positions:
            chosen_positions = order[: min(config.topk, order.numel())].tolist()
        chosen_pos = torch.tensor(chosen_positions, device=self.device, dtype=torch.long)
        neighbors = candidates[chosen_pos]
        neighbor_scores = scores[chosen_pos]
        weights = torch.softmax(config.beta * neighbor_scores, dim=0)
        probabilities = torch.zeros(NUM_DAMAGE_CLASSES, device=self.device)
        probabilities.scatter_add_(0, self.labels[neighbors], weights)

        proto_weights = torch.softmax(config.beta * proto_similarity, dim=0)
        prototype_probabilities = torch.zeros(NUM_DAMAGE_CLASSES, device=self.device)
        prototype_probabilities.scatter_add_(0, self.prototype_labels, proto_weights)
        agreement = float((probabilities * prototype_probabilities).sum().item())
        sorted_probability = probabilities.sort(descending=True).values
        purity = float(sorted_probability[0].item())
        margin = float((sorted_probability[0] - sorted_probability[1]).item())
        best_similarity = float(neighbor_scores.max().item())
        event_count = len({self.events[int(index)] for index in neighbors.tolist()})
        base_sorted = base.sort(descending=True).values
        base_margin = float((base_sorted[0] - base_sorted[1]).item())

        active = (
            best_similarity >= config.similarity_threshold
            and purity >= config.purity_threshold
            and margin >= config.retrieval_margin_threshold
            and agreement >= config.agreement_threshold
            and event_count >= config.minimum_events
            and base_margin <= config.base_margin_max
        )
        if active:
            components = (
                min(1.0, purity / max(config.purity_threshold, 1e-6)),
                min(1.0, agreement / max(config.agreement_threshold, 1e-6)),
                min(1.0, margin / max(config.retrieval_margin_threshold, 1e-6)),
            )
            gate = float(min(components))
        else:
            gate = 0.0
        return RetrievalResult(
            probabilities=probabilities,
            prototype_probabilities=prototype_probabilities,
            neighbor_indices=neighbors,
            neighbor_scores=neighbor_scores,
            gate=gate,
            active=active,
            purity=purity,
            margin=margin,
            best_similarity=best_similarity,
            agreement=agreement,
            event_count=event_count,
        )


def safe_probability_fusion(
    base_probabilities: torch.Tensor,
    retrieval: RetrievalResult,
    alpha: float,
) -> torch.Tensor:
    """Convex fusion; prototype probabilities only affect the gate, not vote."""

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    base = base_probabilities.float()
    base = base / base.sum().clamp_min(1e-8)
    effective = float(alpha) * float(retrieval.gate)
    output = (1.0 - effective) * base + effective * retrieval.probabilities.to(base.device)
    return output / output.sum().clamp_min(1e-8)


@torch.no_grad()
def extract_raw_building_descriptors(
    mil_head: nn.Module,
    rgb_feature: torch.Tensor,
    sar_feature: torch.Tensor,
    fused_feature: torch.Tensor,
    bag_specs: Sequence[object],
    damage_logits: Optional[torch.Tensor] = None,
    max_local_tokens: int = 16,
) -> List[Dict[str, object]]:
    """Reuse the trained S8 MICB projections to create S9 raw records.

    ``bag_specs`` must expose ``batch_index``, ``mask``, ``label`` and ``area``.
    The function intentionally returns raw descriptors; labels and event ids
    are attached by the caller so validation/test can never enter train memory.
    """

    rgb = mil_head.rgb_proj(rgb_feature)
    sar = mil_head.sar_proj(sar_feature)
    fused = mil_head.fused_proj(fused_feature)
    token_map = mil_head.cross_token(
        torch.cat((rgb, sar, torch.abs(rgb - sar), rgb * sar, fused), dim=1)
    )
    change_energy = torch.abs(rgb - sar).mean(dim=1)
    size = token_map.shape[-2:]
    output: List[Dict[str, object]] = []
    for spec in bag_specs:
        batch_index = int(spec.batch_index)
        full = mil_head._resize_mask(spec.mask, size, token_map.device)
        if float(full.sum().item()) <= 1e-6:
            continue
        try:
            tokens, flat_indices = mil_head._select_tokens(
                token_map[batch_index], full, change_energy[batch_index]
            )
        except RuntimeError:
            continue
        exist_score = mil_head.local_exist(tokens)[:, 0]
        severity_score = mil_head.local_severity(tokens).amax(dim=1)
        combined = torch.sigmoid(exist_score) + 0.25 * torch.sigmoid(severity_score)
        local_count = min(max_local_tokens, int(tokens.shape[0]))
        local_index = torch.topk(combined, k=local_count).indices
        local_tokens = tokens[local_index]
        critical = tokens[exist_score.argmax()]
        top_count = min(int(mil_head.topk), int(tokens.shape[0]))
        soft_top = tokens[torch.topk(exist_score, k=top_count).indices].mean(dim=0)
        global_token = mil_head._weighted_pool(token_map[batch_index], full)
        rgb_structure = mil_head._weighted_pool(rgb[batch_index], full)
        sar_global = mil_head._weighted_pool(sar[batch_index], full)
        _, ring = mil_head._region_weights(full)
        sar_ring = mil_head._weighted_pool(sar[batch_index], ring)
        query = _normalize(mil_head.query(critical[None]))[0]
        keys = _normalize(mil_head.query(tokens))
        attention = torch.softmax((keys @ query) / mil_head.attention_temperature, dim=0)
        attention_bag = (attention[:, None] * mil_head.value(tokens)).sum(dim=0)
        change_global = 0.5 * global_token + 0.2 * critical + 0.15 * soft_top + 0.15 * attention_bag

        if damage_logits is not None:
            mask_logits = mil_head._resize_mask(
                spec.mask, damage_logits.shape[-2:], damage_logits.device
            )
            probability = F.softmax(damage_logits[batch_index].float(), dim=0)[1:4]
            denom = mask_logits.sum().clamp_min(1e-6)
            base_probs = (probability * mask_logits[None]).sum(dim=(1, 2)) / denom
            base_probs = base_probs / base_probs.sum().clamp_min(1e-8)
        else:
            base_probs = token_map.new_full((3,), 1.0 / 3.0)

        feature_attention = token_map.new_zeros(size[0] * size[1])
        feature_attention[flat_indices] = torch.sigmoid(exist_score)
        feature_attention = feature_attention.reshape(size) * full
        output.append(
            {
                "batch_index": batch_index,
                "label": int(getattr(spec, "label", -1)),
                "area": int(getattr(spec, "area", 0)),
                "rgb_structure": rgb_structure.detach(),
                "sar_global": sar_global.detach(),
                "sar_ring": sar_ring.detach(),
                "change_global": change_global.detach(),
                "local_tokens": local_tokens.detach(),
                "base_probs": base_probs.detach(),
                "feature_attention": feature_attention.detach(),
                "mask": spec.mask,
            }
        )
    return output


__all__ = [
    "CrossModalMetricProjector",
    "HierarchicalPrototypeInstanceMemory",
    "MetricConfig",
    "RetrievalConfig",
    "RetrievalResult",
    "build_hierarchical_memory",
    "extract_raw_building_descriptors",
    "greedy_kcenter",
    "hierarchical_log_probs",
    "metric_training_loss",
    "prototype_diversity_loss",
    "safe_probability_fusion",
    "spherical_kmeans",
    "supervised_contrastive_loss",
]
