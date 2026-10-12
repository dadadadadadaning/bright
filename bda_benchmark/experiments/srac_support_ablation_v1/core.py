from __future__ import annotations

import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_records(path, split, sources):
    data = torch.load(path, map_location='cpu', weights_only=False)
    meta = data.get('metadata', {})
    if data.get('test_accessed') is True or meta.get('test_accessed') is True:
        raise ValueError('Test-accessed cache forbidden')
    if data.get('split') != split or data.get('test_accessed', meta.get('test_accessed')) is not False:
        raise ValueError('Explicit train/validation split and test_accessed=false required')
    if split == 'train' and data.get('allowed_for_training', True) is not True:
        raise ValueError('Cache forbids training')
    y = data['labels'].long()
    mask = data['token_mask'].bool()
    if y.ndim != 1 or not len(y) or not torch.isin(y, torch.tensor([1, 2, 3])).all():
        raise ValueError('Expected nonempty labels in 1,2,3 (Intact/Damaged/Destroyed)')
    if mask.ndim != 2 or len(mask) != len(y) or not mask.any(1).all():
        raise ValueError('Every region must contain a valid sampled token')
    for s in sources:
        x = data['tokens'][s]
        if x.ndim != 3 or x.shape[:2] != mask.shape or not torch.isfinite(x).all():
            raise ValueError(f'Invalid token source {s}')
    for name in ('events', 'images'):
        if len(data[name]) != len(y) or any(not str(v) for v in data[name]):
            raise ValueError(f'Invalid {name}')
    return data


def batch(data, ids, sources, device):
    x = torch.cat([data['tokens'][s][ids].float() for s in sources], -1).to(device)
    return x, data['token_mask'][ids].to(device)


class Pooler(nn.Module):
    def __init__(self, channels, dim=128, weighted=False):
        super().__init__()
        self.weighted = weighted
        self.project = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, dim), nn.GELU())
        self.main = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        if weighted:
            self.score = nn.Linear(dim, 1)
            self.skip = nn.Sequential(nn.LayerNorm(2 * channels), nn.Linear(2 * channels, dim))
            self.alpha = nn.Parameter(torch.zeros(()))
        self.classifier = nn.Linear(dim, 3)

    def forward(self, x, mask):
        w = mask.float().unsqueeze(-1)
        n = w.sum(1).clamp_min(1)
        h = self.project(x)
        if self.weighted:
            a = self.score(h).squeeze(-1).masked_fill(~mask, -torch.inf).softmax(1)
            main = self.main((h * a[..., None]).sum(1))
            mean = (x * w).sum(1) / n
            std = (((x - mean[:, None]).square() * w).sum(1) / n + 1e-6).sqrt()
            z = main + self.alpha.tanh() * self.skip(torch.cat([mean, std], -1))
        else:
            z = self.main((h * w).sum(1) / n)
        return z, self.classifier(z)


class SupportHead(nn.Module):
    """Same active parameters in E2/E3. E2 derives support slots from query only."""
    def __init__(self, dim=128):
        super().__init__()
        self.labels = nn.Embedding(3, dim)
        self.attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.out = nn.Linear(dim, 3)

    def forward(self, q, support=None, labels=None):
        if support is None:
            # Three learned class slots; query predictions, no GT or external memory.
            slots = q[:, None] + self.labels.weight[None]
            keys = values = slots
        else:
            keys = support
            values = support + self.labels(labels)
        context, _ = self.attention(q[:, None], keys, values, need_weights=False)
        h = self.norm(q + context[:, 0])
        return self.out(h + self.ffn(h))


def select_bank(data, per_bucket, seed):
    buckets = {}
    for i, (e, c) in enumerate(zip(data['events'], data['labels'].tolist())):
        buckets.setdefault((str(e), c), []).append(i)
    gen = torch.Generator().manual_seed(seed)
    out = []
    for key in sorted(buckets):
        rows = torch.tensor(buckets[key])
        out.extend(rows[torch.randperm(len(rows), generator=gen)[:per_bucket]].tolist())
    return torch.tensor(out)


def retrieve(query, bank, query_events, query_images, bank_events, bank_images, k,
             random=False, generator=None):
    score = F.normalize(query, dim=-1) @ F.normalize(bank, dim=-1).T
    allowed = torch.tensor([[e != be and im != bi for be, bi in zip(bank_events, bank_images)]
                            for e, im in zip(query_events, query_images)], device=query.device)
    if (allowed.sum(1) < k).any():
        raise ValueError('Insufficient cross-event support: reduce topk or expand Train bank')
    if random:
        score = torch.rand(score.shape, generator=generator).to(score.device)
    return score.masked_fill(~allowed, -torch.inf).topk(k, dim=1).indices


def metrics(y, pred):
    cm = torch.bincount(y * 3 + pred, minlength=9).reshape(3, 3).double()
    tp = cm.diag()
    precision = tp / cm.sum(0).clamp_min(1)
    recall = tp / cm.sum(1).clamp_min(1)
    f1 = 2 * tp / (cm.sum(0) + cm.sum(1)).clamp_min(1)
    return dict(accuracy=float(tp.sum() / cm.sum().clamp_min(1)), macro_f1=float(f1.mean()),
                precision=precision.tolist(), recall=recall.tolist(), f1=f1.tolist(),
                confusion=cm.long().tolist(), count=int(cm.sum()))
