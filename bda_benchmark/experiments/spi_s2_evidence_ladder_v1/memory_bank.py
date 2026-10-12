from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FrozenMemoryBank:
    keys: torch.Tensor
    labels: torch.Tensor
    events: tuple[str, ...]
    image_ids: tuple[str, ...]
    p3_sha256: str
    split_sha256: str

    def validate(self) -> None:
        if self.keys.ndim != 2 or self.keys.shape[0] == 0:
            raise ValueError("memory keys must be non-empty [N,D]")
        if self.labels.shape != (self.keys.shape[0],):
            raise ValueError("memory labels do not align")
        if len(self.events) != self.keys.shape[0] or len(self.image_ids) != self.keys.shape[0]:
            raise ValueError("memory metadata does not align")
        if not set(self.labels.tolist()).issubset({1, 2, 3}):
            raise ValueError("memory contains non-building labels")
        if not bool(torch.isfinite(self.keys).all()):
            raise FloatingPointError("memory contains non-finite keys")
        for label in (1, 2, 3):
            if int((self.labels == label).sum()) == 0:
                raise RuntimeError(f"memory class {label} is empty")
        if not self.p3_sha256 or not self.split_sha256:
            raise RuntimeError("memory provenance is incomplete")

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "protocol": "SPI_S2_CROSS_EVENT_MEMORY_BANK_V1",
            "keys": F.normalize(self.keys.float(), dim=1).cpu().half(),
            "labels": self.labels.long().cpu(),
            "events": list(self.events),
            "image_ids": list(self.image_ids),
            "p3_sha256": self.p3_sha256,
            "train_split_sha256": self.split_sha256,
            "validation_accessed": False,
            "test_accessed": False,
        }


def build_memory_bank(
    keys: torch.Tensor,
    labels: torch.Tensor,
    events: Sequence[str],
    image_ids: Sequence[str],
    *,
    p3_sha256: str,
    split_sha256: str,
) -> FrozenMemoryBank:
    bank = FrozenMemoryBank(
        keys=keys.detach().float().cpu(),
        labels=labels.detach().long().cpu(),
        events=tuple(map(str, events)),
        image_ids=tuple(map(str, image_ids)),
        p3_sha256=str(p3_sha256),
        split_sha256=str(split_sha256),
    )
    bank.validate()
    return bank


def save_memory_bank(bank: FrozenMemoryBank, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(bank.payload(), temporary)
    os.replace(temporary, destination)


def load_memory_bank(
    path: str | Path,
    *,
    expected_p3_sha256: str,
    expected_split_sha256: str,
) -> FrozenMemoryBank:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != "SPI_S2_CROSS_EVENT_MEMORY_BANK_V1":
        raise RuntimeError("incompatible memory-bank protocol")
    if payload.get("p3_sha256") != expected_p3_sha256:
        raise RuntimeError("memory/P3 fingerprint mismatch")
    if payload.get("train_split_sha256") != expected_split_sha256:
        raise RuntimeError("memory/train-split fingerprint mismatch")
    if bool(payload.get("validation_accessed", True)) or bool(payload.get("test_accessed", True)):
        raise RuntimeError("memory bank used a forbidden split")
    return build_memory_bank(
        payload["keys"], payload["labels"], payload["events"], payload["image_ids"],
        p3_sha256=payload["p3_sha256"], split_sha256=payload["train_split_sha256"],
    )

