# -*- coding: utf-8 -*-
"""S7-SIRM: train-only static building-instance retrieval memory.

This module deliberately does not update memory at test time.  It keeps the
base DamageFormer damage logits and adds a reliability-gated, instance-level
cache residual only to the three building classes:

    [Intact, Damaged, Destroyed].

The static cache is built offline from labelled training images.  Each key is
the L2-normalized, mask-pooled ``final_feat_fused_roi`` representation of one
training building region; each value is its damage label.  This is the direct
analogue of Tip-Adapter's key-value cache for BRIGHT semantic segmentation.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_BUILDING_DAMAGE_CLASSES = 3


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _extract_checkpoint_state(payload: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint must contain a state-dict mapping")
    state: object = payload
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(state, Mapping) and key in state and isinstance(state[key], Mapping):
            state = state[key]
            break
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint does not contain a usable model state dict")
    return state  # type: ignore[return-value]


def _canonical_state_key(key: str) -> str:
    prefixes = ("module.", "model.", "base_model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def load_s7_base_model(
    checkpoint_path: str | Path,
    device: torch.device,
    roi_alpha: float = 0.1,
) -> nn.Module:
    """Load S1 or S3 checkpoint into the prototype-free DamageFormer_BG.

    For an S3 checkpoint, all ``damage_proto_memory.*`` entries are dropped.
    The remaining model exactly reproduces S3's prototype-bypass branch.  This
    makes M1 (K=3) versus M2 (SIRM) a same-checkpoint comparison.
    """

    if __package__:
        from .DamageFormer_BG import DamageFormer_BG
    else:
        from model.DamageFormer_BG import DamageFormer_BG

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    source_state = _extract_checkpoint_state(payload)
    state: Dict[str, torch.Tensor] = {}
    dropped_prototype = 0
    for raw_key, value in source_state.items():
        key = _canonical_state_key(str(raw_key))
        if key.startswith("damage_proto_memory."):
            dropped_prototype += 1
            continue
        if isinstance(value, torch.Tensor):
            state[key] = value

    model = DamageFormer_BG(
        roi_alpha=roi_alpha,
        detach_building_prior=True,
        use_pretrained_swin=False,
    )
    incompatible = model.load_state_dict(state, strict=False)
    allowed_unused = (
        ".swin_transformer.norm.weight",
        ".swin_transformer.norm.bias",
        ".swin_transformer.head.weight",
        ".swin_transformer.head.bias",
    )
    serious_missing = [
        key for key in incompatible.missing_keys if not key.endswith(allowed_unused)
    ]
    serious_unexpected = [
        key for key in incompatible.unexpected_keys if not key.startswith("damage_proto_memory.")
    ]
    if serious_missing or serious_unexpected:
        raise RuntimeError(
            "base checkpoint is incompatible with DamageFormer_BG: "
            f"missing={serious_missing}, unexpected={serious_unexpected}"
        )
    model.to(device)
    model.eval()
    model.s7_load_info = {
        "dropped_prototype_keys": dropped_prototype,
        "allowed_missing_keys": list(incompatible.missing_keys),
    }
    return model


@dataclass(frozen=True)
class RetrievalConfig:
    """Inference-only SIRM-M2b hyperparameters selected on validation data.

    ``directional_intact_only`` implements the predeclared M2b safeguard: a
    memory entry may only promote Damaged/Destroyed evidence at pixels that
    the frozen base model currently predicts as Intact.  The memory therefore
    cannot turn an already predicted Damaged/Destroyed pixel into Intact or
    swap one minority class for the other.
    """

    topk: int = 8
    alpha: float = 0.5
    beta: float = 5.5
    similarity_threshold: float = 0.40
    purity_threshold: float = 0.55
    margin_threshold: float = 0.05
    laplace: float = 0.05
    base_margin_threshold: float = 0.25
    directional_intact_only: bool = True

    def validate(self) -> None:
        if self.topk <= 0:
            raise ValueError(f"topk must be positive, got {self.topk}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        for name in ("similarity_threshold", "purity_threshold", "margin_threshold"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0,1), got {value}")
        if self.laplace <= 0:
            raise ValueError(f"laplace must be positive, got {self.laplace}")
        if not 0.0 < self.base_margin_threshold <= 1.0:
            raise ValueError(
                "base_margin_threshold must be in (0,1], got "
                f"{self.base_margin_threshold}"
            )


@dataclass
class InstanceQuery:
    """One connected building region from one image in a mini-batch."""

    batch_index: int
    flat_indices: torch.Tensor  # CPU int64 indices at the full logit resolution.
    feature: torch.Tensor  # Device tensor [D], already L2-normalized.
    area: int


def _fallback_connected_components(mask: np.ndarray) -> List[np.ndarray]:
    """Dependency-free 4-connected components used if scipy is unavailable."""

    mask = np.asarray(mask, dtype=np.bool_)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=np.bool_)
    components: List[np.ndarray] = []
    for y0, x0 in np.argwhere(mask):
        if visited[y0, x0]:
            continue
        visited[y0, x0] = True
        queue: deque[Tuple[int, int]] = deque([(int(y0), int(x0))])
        ys: List[int] = []
        xs: List[int] = []
        while queue:
            y, x = queue.popleft()
            ys.append(y)
            xs.append(x)
            if y > 0 and mask[y - 1, x] and not visited[y - 1, x]:
                visited[y - 1, x] = True
                queue.append((y - 1, x))
            if y + 1 < height and mask[y + 1, x] and not visited[y + 1, x]:
                visited[y + 1, x] = True
                queue.append((y + 1, x))
            if x > 0 and mask[y, x - 1] and not visited[y, x - 1]:
                visited[y, x - 1] = True
                queue.append((y, x - 1))
            if x + 1 < width and mask[y, x + 1] and not visited[y, x + 1]:
                visited[y, x + 1] = True
                queue.append((y, x + 1))
        component = np.zeros_like(mask, dtype=np.bool_)
        component[np.asarray(ys), np.asarray(xs)] = True
        components.append(component)
    return components


def connected_components_4(mask: torch.Tensor | np.ndarray) -> List[np.ndarray]:
    """Return 4-connected boolean component masks on CPU."""

    if isinstance(mask, torch.Tensor):
        array = mask.detach().to(device="cpu").numpy().astype(np.bool_, copy=False)
    else:
        array = np.asarray(mask, dtype=np.bool_)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {array.shape}")
    try:
        from scipy import ndimage  # type: ignore

        labels, count = ndimage.label(
            array,
            structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
        )
        return [labels == index for index in range(1, int(count) + 1)]
    except ImportError:
        return _fallback_connected_components(array)


def extract_instance_queries(
    feature_map: torch.Tensor,
    building_mask: torch.Tensor,
    min_area: int = 32,
    max_area: Optional[int] = None,
) -> List[InstanceQuery]:
    """Pool one normalized feature for every connected building region.

    Args:
        feature_map: ``[B,D,h,w]`` fused ROI feature map.
        building_mask: ``[B,H,W]`` binary GT (cache build) or predicted mask
            (validation/test).  It is never derived from test damage labels.
        min_area: minimum component area at full image resolution.
        max_area: optional upper bound used to reject merged mega-components.
    """

    if feature_map.ndim != 4 or building_mask.ndim != 3:
        raise ValueError(
            f"expected feature [B,D,h,w] and mask [B,H,W], got "
            f"{tuple(feature_map.shape)} and {tuple(building_mask.shape)}"
        )
    if feature_map.shape[0] != building_mask.shape[0]:
        raise ValueError("feature_map and building_mask batch sizes differ")
    if min_area <= 0:
        raise ValueError("min_area must be positive")

    output: List[InstanceQuery] = []
    full_h, full_w = building_mask.shape[-2:]
    feat_h, feat_w = feature_map.shape[-2:]
    for batch_index in range(feature_map.shape[0]):
        for component in connected_components_4(building_mask[batch_index] > 0):
            area = int(component.sum())
            if area < min_area or (max_area is not None and area > max_area):
                continue

            component_tensor = torch.from_numpy(component.astype(np.float32, copy=False))
            component_tensor = component_tensor.to(feature_map.device)[None, None]
            # Area interpolation retains fractional support for small buildings
            # instead of making them disappear after nearest downsampling.
            weights = F.interpolate(component_tensor, size=(feat_h, feat_w), mode="area")[0, 0]
            denom = weights.sum()
            if float(denom.item()) <= 1e-6:
                continue
            pooled = (feature_map[batch_index] * weights.unsqueeze(0)).sum(dim=(1, 2)) / denom
            pooled = F.normalize(pooled.float(), dim=0)

            flat_indices = torch.from_numpy(np.flatnonzero(component.reshape(-1))).long()
            if int(flat_indices.numel()) != area:
                raise RuntimeError("component index extraction produced an inconsistent area")
            output.append(
                InstanceQuery(
                    batch_index=batch_index,
                    flat_indices=flat_indices,
                    feature=pooled,
                    area=area,
                )
            )
    return output


class StaticBuildingInstanceMemory(nn.Module):
    """Read-only train-instance key-value cache."""

    def __init__(self, keys: torch.Tensor, labels: torch.Tensor, metadata: Optional[Mapping] = None):
        super().__init__()
        if keys.ndim != 2:
            raise ValueError(f"keys must be [N,D], got {tuple(keys.shape)}")
        if labels.ndim != 1 or labels.shape[0] != keys.shape[0]:
            raise ValueError("labels must be [N] and match keys")
        if keys.shape[0] == 0:
            raise ValueError("static instance memory cannot be empty")
        labels = labels.long()
        if int(labels.min().item()) < 0 or int(labels.max().item()) >= NUM_BUILDING_DAMAGE_CLASSES:
            raise ValueError("cache labels must use 0=Intact, 1=Damaged, 2=Destroyed")
        self.register_buffer("keys", F.normalize(keys.float(), dim=1), persistent=True)
        self.register_buffer("labels", labels, persistent=True)
        self.metadata: Dict = dict(metadata or {})

    @property
    def feature_dim(self) -> int:
        return int(self.keys.shape[1])

    @classmethod
    def from_file(cls, path: str | Path, map_location: str | torch.device = "cpu"):
        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=map_location)
        if not isinstance(payload, Mapping):
            raise TypeError("cache file must contain a mapping")
        for required in ("keys", "labels"):
            if required not in payload:
                raise KeyError(f"cache file is missing '{required}'")
        return cls(payload["keys"], payload["labels"], payload.get("metadata", {}))

    @torch.no_grad()
    def retrieve(self, queries: torch.Tensor, config: RetrievalConfig) -> Dict[str, torch.Tensor]:
        config.validate()
        if queries.ndim != 2 or queries.shape[1] != self.feature_dim:
            raise ValueError(
                f"queries must be [Q,{self.feature_dim}], got {tuple(queries.shape)}"
            )
        queries = F.normalize(queries.float(), dim=1)
        topk = min(int(config.topk), int(self.keys.shape[0]))
        similarity = queries @ self.keys.t()
        top_similarity, top_indices = similarity.topk(topk, dim=1, largest=True, sorted=True)
        top_labels = self.labels[top_indices]

        # Tip-Adapter affinity: exp[-beta * (1 - cosine_similarity)].
        affinity = torch.exp(-float(config.beta) * (1.0 - top_similarity))
        values = F.one_hot(top_labels, num_classes=NUM_BUILDING_DAMAGE_CLASSES).to(affinity.dtype)
        votes = (affinity.unsqueeze(-1) * values).sum(dim=1)
        probability = (votes + float(config.laplace)) / (
            votes.sum(dim=1, keepdim=True) + NUM_BUILDING_DAMAGE_CLASSES * float(config.laplace)
        )

        sorted_probability = probability.sort(dim=1, descending=True).values
        class_purity = sorted_probability[:, 0]
        class_margin = sorted_probability[:, 0] - sorted_probability[:, 1]
        best_similarity = top_similarity[:, 0]

        def soft_threshold(value: torch.Tensor, threshold: float) -> torch.Tensor:
            return ((value - threshold) / max(1.0 - threshold, 1e-6)).clamp(0.0, 1.0)

        gate = (
            soft_threshold(best_similarity, float(config.similarity_threshold))
            * soft_threshold(class_purity, float(config.purity_threshold))
            * soft_threshold(class_margin, float(config.margin_threshold))
        )
        # Centered log-probabilities keep the cache correction comparable across
        # classes and leave the base model untouched when gate == 0.
        cache_logits = probability.clamp_min(1e-6).log()
        cache_logits = cache_logits - cache_logits.mean(dim=1, keepdim=True)
        return {
            "cache_logits": cache_logits,
            "cache_probability": probability,
            "gate": gate,
            "best_similarity": best_similarity,
            "class_purity": class_purity,
            "class_margin": class_margin,
            "cache_prediction": probability.argmax(dim=1),
            "top_indices": top_indices,
            "top_labels": top_labels,
        }


class DamageFormer_S7_SIRM_M2B(nn.Module):
    """Inference wrapper that inserts SIRM into the building classification branch.

    Validation-grid evaluation uses :func:`refine_damage_logits` directly so a
    single base forward can serve several configurations.  This wrapper is the
    convenient locked-configuration form for deployment or visualization.
    """

    def __init__(
        self,
        base_model: nn.Module,
        memory: StaticBuildingInstanceMemory,
        config: RetrievalConfig,
        min_instance_area: int = 32,
        max_instance_area: Optional[int] = 300000,
    ):
        super().__init__()
        config.validate()
        self.base_model = base_model
        self.memory = memory
        self.config = config
        self.min_instance_area = int(min_instance_area)
        self.max_instance_area = max_instance_area

    def forward(self, pre_data: torch.Tensor, post_data: torch.Tensor, return_aux: bool = False):
        if self.training:
            raise RuntimeError(
                "S7-SIRM M2 is an inference-only static cache. Call model.eval(); "
                "do not train it as a 50k-step end-to-end branch."
            )
        output_loc, base_damage, aux = self.base_model(pre_data, post_data, return_aux=True)
        predicted_building = output_loc.argmax(dim=1) == 1
        instances = extract_instance_queries(
            aux["final_feat_fused_roi"],
            predicted_building,
            min_area=self.min_instance_area,
            max_area=self.max_instance_area,
        )
        refined_damage, retrieval_stats = refine_damage_logits(
            base_damage,
            instances,
            self.memory,
            self.config,
        )
        if not return_aux:
            return output_loc, refined_damage
        aux = dict(aux)
        aux.update(
            {
                "base_damage_logits": base_damage,
                "sirm_instance_count": len(instances),
                "sirm_retrieval": retrieval_stats,
            }
        )
        return output_loc, refined_damage, aux


@torch.no_grad()
def refine_damage_logits(
    damage_logits: torch.Tensor,
    instances: Sequence[InstanceQuery],
    memory: StaticBuildingInstanceMemory,
    config: RetrievalConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Apply a conservative, directional cache correction.

    M2a added one centered three-class cache residual to every pixel in an
    active component.  The validation confusion matrix showed that this did
    reduce D/D->Intact errors, but it also changed many already-correct pixels.
    M2b therefore applies a positive boost only when all of the following hold:

    * the cache predicts Damaged or Destroyed (never Intact);
    * the base damage head currently predicts that pixel as Intact;
    * the base Intact-vs-minority probability margin is below the validation-
      selected ``base_margin_threshold``;
    * the retrieval similarity/purity/margin gate is non-zero.

    Background logits, non-Intact predictions, pixels outside components, and
    the non-target minority logit remain unchanged.  This isolates the intended
    D/D->Intact repair instead of performing a symmetric class rewrite.
    """

    config.validate()
    if damage_logits.ndim != 4 or damage_logits.shape[1] != 4:
        raise ValueError(f"damage_logits must be [B,4,H,W], got {tuple(damage_logits.shape)}")
    if not instances:
        empty = torch.empty(0, device=damage_logits.device)
        return damage_logits.clone(), {
            "gate": empty,
            "best_similarity": empty,
            "class_purity": empty,
            "class_margin": empty,
            "cache_prediction": empty.long(),
            "base_uncertainty_gate": empty,
            "applied_instance": empty.bool(),
            "eligible_pixel_count": empty.long(),
            "modified_pixel_count": empty.long(),
            "flipped_from_intact_count": empty.long(),
        }

    queries = torch.stack([instance.feature for instance in instances], dim=0)
    retrieved = memory.retrieve(queries, config)
    refined = damage_logits.clone()
    count = len(instances)
    base_uncertainty_gate = torch.zeros(count, device=damage_logits.device)
    applied_instance = torch.zeros(count, dtype=torch.bool, device=damage_logits.device)
    eligible_pixel_count = torch.zeros(count, dtype=torch.long, device=damage_logits.device)
    modified_pixel_count = torch.zeros(count, dtype=torch.long, device=damage_logits.device)
    flipped_from_intact_count = torch.zeros(count, dtype=torch.long, device=damage_logits.device)
    correction_norm = torch.zeros(count, device=damage_logits.device)

    for index, instance in enumerate(instances):
        retrieval_gate = retrieved["gate"][index]
        if float(retrieval_gate.item()) <= 0.0 or float(config.alpha) <= 0.0:
            continue
        if not 0 <= instance.batch_index < damage_logits.shape[0]:
            raise IndexError("instance batch_index is outside damage_logits")

        cache_target = int(retrieved["cache_prediction"][index].item())
        if config.directional_intact_only and cache_target == 0:
            # M2b never uses the cache to promote Intact.
            continue

        flat_indices = instance.flat_indices.to(damage_logits.device, non_blocking=True)
        if flat_indices.numel() == 0:
            continue

        # Use the frozen/base logits to decide eligibility.  Decisions are not
        # cascaded from previously refined values.
        full_base_logits = damage_logits[instance.batch_index].reshape(4, -1)[:, flat_indices]
        full_base_prediction = full_base_logits.argmax(dim=0)
        building_logits = refined[instance.batch_index, 1:4].reshape(3, -1)
        base_building_logits = damage_logits[instance.batch_index, 1:4].reshape(3, -1)[
            :, flat_indices
        ]
        base_probability = F.softmax(base_building_logits.float(), dim=0)
        intact_probability = base_probability[0]
        minority_probability = base_probability[1:3].max(dim=0).values
        base_margin = (intact_probability - minority_probability).clamp_min(0.0)

        threshold = float(config.base_margin_threshold)
        uncertainty = ((threshold - base_margin) / threshold).clamp(0.0, 1.0)
        eligible = full_base_prediction == 1
        if not config.directional_intact_only:
            eligible = torch.ones_like(eligible)
        pixel_gate = uncertainty * eligible.to(uncertainty.dtype)

        eligible_pixel_count[index] = eligible.sum()
        base_uncertainty_gate[index] = pixel_gate.mean() if pixel_gate.numel() else 0.0
        if not bool((pixel_gate > 0).any().item()):
            continue

        cache_probability = retrieved["cache_probability"][index]
        if cache_target == 0:
            # Symmetric mode is retained only for explicit diagnostics.  M2b
            # validation/test grids always use directional_intact_only=True.
            target_odds = cache_probability[0] / cache_probability[1:3].max().clamp_min(1e-6)
        else:
            target_odds = cache_probability[cache_target] / cache_probability[0].clamp_min(1e-6)
        evidence = target_odds.clamp_min(1.0).log()
        boost = (
            float(config.alpha)
            * retrieval_gate
            * evidence
            * pixel_gate
        ).to(building_logits.dtype)
        changed = boost > 0
        if not bool(changed.any().item()):
            continue

        target_logits = building_logits[cache_target, flat_indices]
        target_logits += boost
        building_logits[cache_target, flat_indices] = target_logits

        applied_instance[index] = True
        modified_pixel_count[index] = changed.sum()
        correction_norm[index] = boost.float().norm()
        new_prediction = refined[instance.batch_index].reshape(4, -1)[:, flat_indices].argmax(dim=0)
        flipped_from_intact_count[index] = (
            (full_base_prediction == 1) & (new_prediction != 1)
        ).sum()

    stats = {key: value for key, value in retrieved.items() if key != "cache_logits"}
    stats["base_uncertainty_gate"] = base_uncertainty_gate
    stats["applied_instance"] = applied_instance
    stats["eligible_pixel_count"] = eligible_pixel_count
    stats["modified_pixel_count"] = modified_pixel_count
    stats["flipped_from_intact_count"] = flipped_from_intact_count
    stats["correction_norm"] = correction_norm
    return refined, stats


def balance_instance_records(
    records: Sequence[Mapping],
    max_per_event_class: int = 64,
    max_per_class: int = 2048,
    hard_fraction: float = 0.25,
) -> List[Dict]:
    """Class/event-balanced deterministic cache selection.

    Records with ``hard_to_intact=True`` are retained up to ``hard_fraction``
    in each event/class bucket; remaining slots prefer high-confidence, pure
    regions.  Event buckets are then interleaved so a large disaster cannot
    monopolize one class memory.
    """

    if max_per_event_class <= 0 or max_per_class <= 0:
        raise ValueError("memory caps must be positive")
    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be in [0,1]")

    grouped: MutableMapping[Tuple[int, str], List[Dict]] = defaultdict(list)
    for record_raw in records:
        record = dict(record_raw)
        label = int(record["label"])
        event = str(record["event"])
        grouped[(label, event)].append(record)

    capped: MutableMapping[int, Dict[str, deque[Dict]]] = defaultdict(dict)
    for (label, event), items in grouped.items():
        score_key = lambda item: (
            -float(item.get("purity", 0.0)),
            -float(item.get("main_confidence", 0.0)),
            str(item.get("sample_id", "")),
            int(item.get("component_index", 0)),
        )
        hard = sorted((x for x in items if bool(x.get("hard_to_intact", False))), key=score_key)
        ordinary = sorted((x for x in items if not bool(x.get("hard_to_intact", False))), key=score_key)
        hard_quota = min(len(hard), int(round(max_per_event_class * hard_fraction)))
        selected = hard[:hard_quota]
        remaining = max_per_event_class - len(selected)
        selected.extend(ordinary[:remaining])
        remaining = max_per_event_class - len(selected)
        if remaining > 0:
            selected.extend(hard[hard_quota : hard_quota + remaining])
        capped[label][event] = deque(selected)

    balanced: List[Dict] = []
    for label in range(NUM_BUILDING_DAMAGE_CLASSES):
        event_queues = capped.get(label, {})
        chosen_for_class = 0
        while event_queues and chosen_for_class < max_per_class:
            made_progress = False
            for event in sorted(event_queues):
                queue = event_queues[event]
                if queue and chosen_for_class < max_per_class:
                    balanced.append(queue.popleft())
                    chosen_for_class += 1
                    made_progress = True
            event_queues = {event: queue for event, queue in event_queues.items() if queue}
            if not made_progress:
                break
    return balanced


def records_to_cache_payload(records: Sequence[Mapping], metadata: Optional[Mapping] = None) -> Dict:
    if not records:
        raise ValueError("cannot build cache payload from zero records")
    keys = torch.stack([torch.as_tensor(record["key"]).float() for record in records], dim=0)
    keys = F.normalize(keys, dim=1).half().cpu()
    labels = torch.tensor([int(record["label"]) for record in records], dtype=torch.long)
    payload_metadata = dict(metadata or {})
    payload_metadata.update(
        {
            "format": "S7_SIRM_v1",
            "feature_dim": int(keys.shape[1]),
            "num_instances": int(keys.shape[0]),
            "label_encoding": {0: "Intact", 1: "Damaged", 2: "Destroyed"},
            "events": [str(record["event"]) for record in records],
            "areas": [int(record["area"]) for record in records],
            "main_confidences": [float(record["main_confidence"]) for record in records],
            "hard_to_intact": [bool(record["hard_to_intact"]) for record in records],
            "sample_ids": [str(record["sample_id"]) for record in records],
            "purities": [float(record["purity"]) for record in records],
        }
    )
    return {"keys": keys, "labels": labels, "metadata": payload_metadata}


def retrieval_config_to_dict(config: RetrievalConfig) -> Dict[str, float | int]:
    config.validate()
    return asdict(config)
