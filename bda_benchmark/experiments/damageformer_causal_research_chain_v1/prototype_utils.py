from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .core import normalize


def kmeans_torch(x: torch.Tensor, k: int, *, iterations: int = 30, seed: int = 3407) -> torch.Tensor:
    """Small deterministic cosine k-means used for oracle diagnostics."""
    x = normalize(x.float())
    if x.shape[0] < k:
        k = int(x.shape[0])
    if k <= 0:
        raise ValueError("cannot build prototype from empty class")
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    # deterministic-ish farthest-first after one seeded point
    first = int(torch.randint(0, x.shape[0], (1,), generator=g, device=x.device))
    centers = [x[first]]
    for _ in range(1, k):
        sim = x @ torch.stack(centers).T
        dist = 1.0 - sim.max(1).values
        centers.append(x[int(dist.argmax())])
    centers = torch.stack(centers)
    for _ in range(iterations):
        assign = (x @ centers.T).argmax(1)
        new = []
        for j in range(k):
            m = assign == j
            if bool(m.any()):
                new.append(normalize(x[m].mean(0, keepdim=True))[0])
            else:
                new.append(centers[j])
        newc = torch.stack(new)
        if float((newc - centers).abs().max()) < 1e-5:
            centers = newc
            break
        centers = newc
    return normalize(centers)


def build_class_prototypes(features: torch.Tensor, labels: torch.Tensor, k: int, seed: int = 3407) -> List[torch.Tensor]:
    rows = []
    for cls in range(3):
        x = features[labels == cls]
        rows.append(kmeans_torch(x, k, seed=seed + 37 * cls))
    return rows


def score_prototypes(features: torch.Tensor, prototypes: Sequence[torch.Tensor], temperature: float = 0.10) -> torch.Tensor:
    x = normalize(features.float())
    scores = []
    tau = max(float(temperature), 1e-4)
    for p in prototypes:
        sim = x @ normalize(p.float()).T
        s = tau * torch.logsumexp(sim / tau, dim=1) - tau * np.log(float(p.shape[0]))
        scores.append(s)
    return torch.stack(scores, dim=1)
