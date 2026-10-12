"""Learnable metric projection for S7-SIRM M3.

M3 keeps the S3V2 base model and M2a train-only instance cache frozen.  It
learns only a small residual embedding that makes cosine similarity more
damage-discriminative.  The projected memory still uses M2b's directional,
base-uncertainty correction, so the only new experimental factor is the metric
space.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch
from torch import nn
import torch.nn.functional as F

if __package__:
    from .DamageFormer_S7_SIRM_M2B import (  # re-exported for evaluator
        DamageFormer_S7_SIRM_M2B,
        InstanceQuery,
        RetrievalConfig,
        StaticBuildingInstanceMemory,
        balance_instance_records,
        connected_components_4,
        extract_instance_queries,
        load_s7_base_model,
        records_to_cache_payload,
        refine_damage_logits,
        retrieval_config_to_dict,
        sha256_file,
    )
else:  # Preserve direct-file smoke tests from the benchmark root.
    from model.DamageFormer_S7_SIRM_M2B import (
        DamageFormer_S7_SIRM_M2B,
        InstanceQuery,
        RetrievalConfig,
        StaticBuildingInstanceMemory,
        balance_instance_records,
        connected_components_4,
        extract_instance_queries,
        load_s7_base_model,
        records_to_cache_payload,
        refine_damage_logits,
        retrieval_config_to_dict,
        sha256_file,
    )


@dataclass(frozen=True)
class ProjectionConfig:
    input_dim: int = 256
    hidden_dim: int = 64
    dropout: float = 0.10
    gamma_init: float = 1.0

    def validate(self) -> None:
        if self.input_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("projection dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if self.gamma_init < 0.0:
            raise ValueError("gamma_init must be non-negative")


class ResidualMetricProjection(nn.Module):
    """Near-identity 256-D residual metric adapter.

    The final linear layer is initialized with very small weights, so the
    initial projected embedding is almost exactly the normalized raw feature.
    This avoids destroying the frozen base representation at the start of
    training while allowing supervised metric losses to reshape it.
    """

    def __init__(self, config: ProjectionConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.norm = nn.LayerNorm(config.input_dim)
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.input_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.gamma = nn.Parameter(torch.tensor(float(config.gamma_init)))
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.config.input_dim:
            raise ValueError(
                f"features must be [N,{self.config.input_dim}], got {tuple(features.shape)}"
            )
        features = features.float()
        residual = self.fc2(self.dropout(F.gelu(self.fc1(self.norm(features)))))
        return F.normalize(features + self.gamma * residual, dim=1)


class ProjectionClassifier(nn.Module):
    """Training-only classifier used to stabilize the projected embedding."""

    def __init__(self, feature_dim: int, num_classes: int = 3):
        super().__init__()
        self.linear = nn.Linear(int(feature_dim), int(num_classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def save_projection_checkpoint(
    path: str | Path,
    projector: ResidualMetricProjection,
    classifier: ProjectionClassifier,
    metadata: Mapping,
) -> None:
    payload = {
        "format": "S7_SIRM_M3_projection_v1",
        "projection_config": asdict(projector.config),
        "projection_state_dict": projector.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "metadata": dict(metadata),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_projection_checkpoint(
    path: str | Path,
    device: torch.device,
) -> Tuple[ResidualMetricProjection, Dict]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("projection checkpoint must contain a mapping")
    if payload.get("format") != "S7_SIRM_M3_projection_v1":
        raise ValueError(f"unsupported projection format: {payload.get('format')!r}")
    config = ProjectionConfig(**dict(payload["projection_config"]))
    projector = ResidualMetricProjection(config)
    projector.load_state_dict(payload["projection_state_dict"], strict=True)
    projector.to(device)
    projector.eval()
    return projector, dict(payload.get("metadata", {}))


class ProjectedStaticBuildingInstanceMemory(StaticBuildingInstanceMemory):
    """Static memory with projected keys and on-the-fly projected queries."""

    def __init__(
        self,
        raw_memory: StaticBuildingInstanceMemory,
        projector: ResidualMetricProjection,
    ):
        device = next(projector.parameters()).device
        projector.eval()
        with torch.no_grad():
            projected_keys = projector(raw_memory.keys.to(device=device, dtype=torch.float32))
        super().__init__(
            projected_keys,
            raw_memory.labels.to(device),
            metadata=dict(raw_memory.metadata),
        )
        self.projector = projector

    @torch.no_grad()
    def retrieve(self, queries: torch.Tensor, config: RetrievalConfig) -> Dict[str, torch.Tensor]:
        projected_queries = self.projector(queries.to(self.keys.device, dtype=torch.float32))
        return super().retrieve(projected_queries, config)


def build_projected_memory(
    cache_path: str | Path,
    projection_path: str | Path,
    device: torch.device,
) -> Tuple[ProjectedStaticBuildingInstanceMemory, Dict]:
    raw_memory = StaticBuildingInstanceMemory.from_file(cache_path, map_location="cpu")
    projector, metadata = load_projection_checkpoint(projection_path, device=device)
    if raw_memory.feature_dim != projector.config.input_dim:
        raise ValueError(
            f"cache feature dim {raw_memory.feature_dim} does not match projection "
            f"input dim {projector.config.input_dim}"
        )
    cache_hash = sha256_file(cache_path)
    expected_cache_hash = str(metadata.get("source_cache_sha256", ""))
    if expected_cache_hash and cache_hash != expected_cache_hash:
        raise RuntimeError(
            "projection/cache mismatch: train or select the projection from this exact cache"
        )
    return ProjectedStaticBuildingInstanceMemory(raw_memory, projector), metadata


__all__ = [
    "DamageFormer_S7_SIRM_M2B",
    "InstanceQuery",
    "ProjectionClassifier",
    "ProjectionConfig",
    "ProjectedStaticBuildingInstanceMemory",
    "ResidualMetricProjection",
    "RetrievalConfig",
    "StaticBuildingInstanceMemory",
    "build_projected_memory",
    "extract_instance_queries",
    "load_projection_checkpoint",
    "load_s7_base_model",
    "refine_damage_logits",
    "save_projection_checkpoint",
    "sha256_file",
]
