"""GT-free retrieval and two-scalar probability readers, not new superpixels."""
import math
import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def shuffled_distribution(bank, event_id, image_id, generator=None):
    allowed = (bank['events'] != int(event_id)) & (bank['images'] != int(image_id))
    eligible = allowed.nonzero().flatten()
    order = torch.arange(len(bank['tokens']), device=bank['tokens'].device)
    order[eligible] = eligible[torch.randperm(len(eligible), device=order.device, generator=generator)]
    return bank['distribution'][order]


@torch.no_grad()
def select_neighbors(query, bank, event_id, image_id, topk=8,
                     intervention='normal', generator=None, distribution=None):
    """No query labels accepted. Same image AND same event are always excluded.

    Random intervention changes neighbor identities, not the cosine weighting
    rule. Supply a once-per-image shuffled distribution to keep labels stable
    across chunks. Empty support is represented by valid=False, never NaNs.
    """
    if intervention not in ('normal', 'random', 'shuffle_labels') or topk < 1:
        raise ValueError('Invalid retrieval request')
    tokens = bank['tokens']
    if query.ndim != 2 or tokens.ndim != 2 or query.shape[1] != tokens.shape[1] or not len(tokens):
        raise ValueError('Expected nonempty, dimension-compatible bank')
    if not torch.isfinite(query).all() or not torch.isfinite(tokens).all():
        raise ValueError('Nonfinite retrieval features')
    allowed = (bank['images'] != int(image_id)) & (bank['events'] != int(event_id))
    normalized = bank.get('normalized_tokens')
    if normalized is None:
        normalized = F.normalize(tokens.float(), dim=-1)
    cosine = F.normalize(query.detach().float(), dim=-1) @ normalized.T
    ranking = torch.rand(cosine.shape, device=query.device, generator=generator) if intervention == 'random' else cosine
    ranked, ids = ranking.masked_fill(~allowed[None], -torch.inf).topk(min(topk, len(tokens)), -1)
    valid = torch.isfinite(ranked)
    if distribution is None:
        distribution = (shuffled_distribution(bank, event_id, image_id, generator)
                        if intervention == 'shuffle_labels' else bank['distribution'])
    labels = distribution[ids].detach()
    labels = torch.where(valid[..., None], labels, torch.zeros_like(labels))
    scores = torch.where(valid, cosine.gather(1, ids), torch.zeros_like(ranked))
    return dict(scores=scores, labels=labels, valid=valid, indices=ids)


def weighted_labels(scores, labels, valid, temperature):
    if scores.ndim != 2 or valid.shape != scores.shape or labels.shape != (*scores.shape, 4):
        raise ValueError('Expected NK scores/mask and NK4 support labels')
    if valid.dtype != torch.bool or scores.shape[1] < 1:
        raise ValueError('Nonempty support axis and bool mask required')
    t = torch.as_tensor(temperature, dtype=scores.dtype, device=scores.device)
    if t.numel() != 1 or not bool(torch.isfinite(t)) or not bool(t > 0):
        raise ValueError('Temperature must be positive and finite')
    # torch.where BEFORE softmax/multiplication prevents masked NaNs leaking.
    safe_scores = torch.where(valid, scores, torch.zeros_like(scores))
    safe_labels = torch.where(valid[..., None], labels, torch.zeros_like(labels))
    if (not torch.isfinite(safe_scores).all() or not torch.isfinite(safe_labels).all()
            or (safe_labels < 0).any()):
        raise ValueError('Invalid valid-support data')
    if valid.any() and not torch.allclose(safe_labels.sum(-1)[valid], torch.ones_like(scores[valid]), atol=1e-5):
        raise ValueError('Support labels must be normalized soft distributions')
    logits = (safe_scores / t).masked_fill(~valid, -1e9)
    weights = logits.softmax(-1) * valid
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    return (weights[..., None] * safe_labels).sum(1), valid.any(-1)


class LabelMixer(nn.Module):
    """Both controls have two scalar parameters and identical initialization.

    Memory: base + weighted Train labels. No-memory: temperature and foreground
    logit bias calibration. Equal parameter COUNT/init, not function/FLOPs.
    Memory alpha=0 preserves the base exactly, including unavailable supports. A
    learned alpha staying zero is a valid negative result, not a failed run.
    """
    def __init__(self, mode):
        super().__init__()
        if mode not in ('memory', 'nomemory'):
            raise ValueError(mode)
        self.mode = mode
        self.alpha = nn.Parameter(torch.tensor(0.))
        self.log_temperature = nn.Parameter(torch.tensor(0.))

    def forward(self, base_prob, scores=None, labels=None, valid=None):
        if base_prob.ndim != 2 or base_prob.shape[-1] != 4:
            raise ValueError('Expected N4 base probability')
        if (not torch.isfinite(base_prob).all() or (base_prob < 0).any()
                or not torch.allclose(base_prob.sum(-1), torch.ones_like(base_prob[:, 0]), atol=1e-5)):
            raise ValueError('Invalid base probability')
        t = self.log_temperature.clamp(math.log(.02), math.log(4.)).exp()
        if self.mode == 'memory':
            evidence, supported = weighted_labels(scores, labels, valid, t)
        else:
            # No GT/bank/support is consumed by the independent calibration.
            # An own-probability mixture with alpha=0,T=1 would give ZERO
            # gradients to both parameters; direct calibration avoids that.
            bias = self.alpha.clamp(-.5, .5) * base_prob.new_tensor([0., 1., 1., 1.])
            return (base_prob.clamp_min(1e-12).log() / t + bias).softmax(-1)
        a = self.alpha.clamp(0., .5)
        result = base_prob + a * (evidence - base_prob)
        return torch.where(supported[:, None], result, base_prob)

    @torch.no_grad()
    def project_(self):
        self.alpha.clamp_(0. if self.mode == 'memory' else -.5, .5)
        self.log_temperature.clamp_(math.log(.02), math.log(4.))

    def values(self):
        return dict(alpha=float(self.alpha.detach().clamp(0 if self.mode == 'memory' else -.5, .5)),
                    temperature=float(self.log_temperature.detach().clamp(math.log(.02), math.log(4.)).exp()))


def mix_fixed(base_prob, evidence, supported, alpha):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Invalid fixed mixture')
    if alpha == 0:
        return base_prob
    # A literal endpoint avoids cancellation/tie changes contaminating knn_only.
    result = evidence if alpha == 1 else base_prob + alpha*(evidence-base_prob)
    return torch.where(supported[:, None], result, base_prob)
