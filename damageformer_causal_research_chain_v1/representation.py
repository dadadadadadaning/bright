from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch

from .tail_models import MultiPrototypeTail


def load_tail_model(path: str, device: torch.device) -> MultiPrototypeTail:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = MultiPrototypeTail(
        int(payload["input_dim"]),
        int(payload["embed_dim"]),
        int(payload["prototypes"]),
        float(payload.get("temperature", 0.10)),
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def transform_features(features: torch.Tensor, tail_ckpt: str = "", device: str = "cuda:0") -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not tail_ckpt:
        return features.float(), None
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = load_tail_model(tail_ckpt, dev)
    embeds, scores = [], []
    for i in range(0, len(features), 4096):
        xb = features[i:i+4096].float().to(dev)
        s, h = model(xb)
        embeds.append(h.cpu())
        scores.append(s.cpu())
    return torch.cat(embeds), torch.cat(scores)
