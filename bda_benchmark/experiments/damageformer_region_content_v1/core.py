import copy
import hashlib
from pathlib import Path

import torch
from torch import nn
from bda_benchmark.experiments.damageformer_modulation_controls_v1.core import capture
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.model import (
    RegionEmbedding, project_probabilities,
)


class FullRegionReadout(nn.Module):
    """Exact V2 weighted pooling + mean/std bypass + classifier architecture.

    Fresh parameters, not the old trained region head. Q and content are frozen.
    No leading pixel-wise LayerNorm is added to the input feature.
    """
    def __init__(self, dim=128):
        super().__init__()
        self.embedding = RegionEmbedding(channels=256, dim=dim)
        self.classifier = nn.Linear(dim, 4)

    def forward(self, feature, reference, output_size):
        q = reference['q'].detach().float()
        ids = reference['ids']
        k = reference['tokens'].shape[1]
        tokens, _ = self.embedding(feature.detach().float(), q, ids, k)
        return project_probabilities(self.classifier(tokens), q, ids,
                                     reference['shape'], output_size)


def make_heads(sources, dim, device):
    prototype = FullRegionReadout(dim).to(device)
    return {s: copy.deepcopy(prototype) for s in sources}


def implementation_hash():
    """Hash experiment plus imported diagnostic helpers; front separately hashed."""
    repo = Path(__file__).resolve().parents[3]
    paths = list(Path(__file__).parent.glob('*.py')) + [
        repo/'bda_benchmark/experiments/damageformer_modulation_controls_v1/core.py',
        repo/'bda_benchmark/experiments/damageformer_region_evidence_v1/core.py',
        repo/'bda_benchmark/experiments/damageformer_region_evidence_v1/train_heads.py',
    ]
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.relative_to(repo).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


@torch.no_grad()
def verify_original_readout(model, reference, output_size):
    """Check captured normal content reproduces the original full region output."""
    tokens, _ = model.embedding(reference['features']['normal'], reference['q'],
                                reference['ids'], reference['tokens'].shape[1])
    replay = project_probabilities(model.classifier(tokens), reference['q'],
                                   reference['ids'], reference['shape'], output_size)
    if not torch.allclose(replay, reference['prob'], atol=1e-5, rtol=1e-5):
        raise RuntimeError('Original full-region replay failed')
    return float((replay-reference['prob']).abs().max())
