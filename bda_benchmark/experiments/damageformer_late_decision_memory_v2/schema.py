from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch

LABEL_BG, LABEL_I, LABEL_D, LABEL_X = 0, 1, 2, 3

@dataclass
class MemoryBank:
    keys: torch.Tensor                 # [N,Dg]
    region_tokens: torch.Tensor        # [N,K,Dr]
    region_mask: torch.Tensor          # [N,K]
    region_meta: torch.Tensor          # [N,K,M], first 4: area/conf/entropy/peak-mean
    labels: torch.Tensor               # [N]
    event_ids: torch.Tensor            # [N]
    building_ids: torch.Tensor         # [N]
    base_probs: Optional[torch.Tensor] = None
    source_sha256: Optional[str] = None

    def validate(self):
        n = self.keys.shape[0]
        assert self.region_tokens.shape[0] == n
        assert self.region_mask.shape == self.region_tokens.shape[:2]
        assert self.region_meta.shape[:2] == self.region_tokens.shape[:2]
        assert self.labels.shape[0] == n
        assert self.event_ids.shape[0] == n
        assert self.building_ids.shape[0] == n
        if self.base_probs is not None:
            assert self.base_probs.shape[0] == n

    def to(self, device):
        return MemoryBank(
            self.keys.to(device), self.region_tokens.to(device),
            self.region_mask.to(device), self.region_meta.to(device),
            self.labels.to(device), self.event_ids.to(device),
            self.building_ids.to(device),
            None if self.base_probs is None else self.base_probs.to(device),
            self.source_sha256,
        )

def load_memory_bank(path, map_location="cpu"):
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, MemoryBank):
        bank = obj
    else:
        bank = MemoryBank(
            keys=obj["keys"],
            region_tokens=obj["region_tokens"],
            region_mask=obj["region_mask"].bool(),
            region_meta=obj["region_meta"],
            labels=obj["labels"].long(),
            event_ids=obj["event_ids"].long(),
            building_ids=obj["building_ids"].long(),
            base_probs=obj.get("base_probs"),
            source_sha256=obj.get("source_sha256"),
        )
    bank.validate()
    return bank
