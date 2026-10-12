"""Frozen inference interventions on external attention allocation only.

The normal prediction always uses the original need_weights=False forward.
Identity reconstructs that forward using observed per-head weights and the
actual value input/projections. Uniform and cosine preserve every head's self
weight and external total. Query GT is never accepted by this module.
"""
import torch
from torch.nn import functional as F


TEMPERATURE = .07
CASES = ('normal', 'identity', 'uniform', 'cosine')
ATOL, RTOL = 2e-5, 2e-4


def _max_abs(value):
    return float(value.abs().max()) if value.numel() else 0.


def _empty_replay():
    return dict(
        max_abs_context=0., max_abs_logits=0., identity_region_argmax_disagreements=0,
        observation_max_abs_context=0., observation_max_abs_logits=0.,
        observation_region_argmax_disagreements=0, max_self_weight_error=0.,
        max_external_mass_error=0., max_invalid_weight=0., max_normalization_error=0.,
    )


def _check_weight_inputs(weights, valid, cosine):
    if (weights.ndim != 3 or valid.ndim != 2 or valid.dtype != torch.bool
            or weights.shape[0] != valid.shape[0]
            or weights.shape[-1] != valid.shape[-1] + 1
            or cosine.shape != valid.shape or not weights.is_floating_point()
            or not cosine.is_floating_point()):
        raise ValueError('Expected weights [N,H,K+1], valid/cosine [N,K]')
    if valid.device != weights.device or cosine.device != weights.device:
        raise ValueError('Weight intervention tensors must share a device')
    if not torch.isfinite(weights).all() or not torch.isfinite(cosine[valid]).all():
        raise ValueError('Weights and valid cosine scores must be finite')
    if (weights < 0).any():
        raise ValueError('Attention weights must be nonnegative')
    if (weights[..., :-1].masked_select(~valid[:, None])).any():
        raise ValueError('Masked supports must have exactly zero attention')
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., -1]),
                               atol=1e-5, rtol=1e-5)


@torch.no_grad()
def redistribute_external_weights(weights, valid, cosine, mode):
    """Return [N,H,K+1] weights; self is the LAST slot, never head-averaged.

    Cosine temperature is deliberately fixed, not a validation-search argument.
    Undefined padding scores are safe; a query without supports keeps its self
    fallback. Inputs are never changed.
    """
    if mode not in ('identity', 'uniform', 'cosine'):
        raise ValueError('Unknown external allocation mode')
    _check_weight_inputs(weights, valid, cosine)
    if mode == 'identity' or valid.shape[-1] == 0:
        return weights.clone()
    if mode == 'uniform':
        relative = valid.to(weights.dtype) / valid.sum(-1, keepdim=True).clamp_min(1)
    else:
        # All-masked queries are made finite before softmax, then zeroed. No
        # undefined padding participates in either the scores or normalization.
        scores = torch.where(valid, cosine, torch.zeros_like(cosine)).to(weights.dtype)
        scores = (scores / TEMPERATURE).masked_fill(~valid, -torch.inf)
        scores = torch.where(valid.any(-1, keepdim=True), scores, torch.zeros_like(scores))
        relative = scores.softmax(-1).masked_fill(~valid, 0.)
    external_mass = weights[..., :-1].sum(-1, keepdim=True)
    external = external_mass * relative[:, None]
    result = torch.cat((external, weights[..., -1:]), -1)
    if not torch.isfinite(result).all():
        raise RuntimeError('Nonfinite redistributed attention')
    return result


def _check_reader(head, query, base, support, labels, valid, cosine):
    if any(module.training for module in head.modules()):
        raise ValueError('Reader and every submodule must be in eval mode')
    if any(p.requires_grad or p.grad is not None for p in head.parameters()):
        raise ValueError('Reader must be frozen and without gradients')
    attention = head.attention
    if (attention.dropout != 0 or not attention.batch_first
            or not attention._qkv_same_embed_dim or attention.in_proj_weight is None
            or attention.bias_k is not None or attention.bias_v is not None
            or attention.add_zero_attn):
        raise ValueError('Unsupported MHA layout; require trained batch-first dropout=0 reader')
    if (query.ndim != 2 or query.shape[-1] != head.dim
            or base.shape != (len(query), 4)
            or support.ndim != 3 or support.shape[0] != len(query)
            or support.shape[-1] != head.dim or labels.shape != (*support.shape[:2], 4)
            or valid.shape != support.shape[:2] or valid.dtype != torch.bool
            or cosine.shape != valid.shape):
        raise ValueError('Invalid query/base/support/label/mask/cosine shapes')
    values = (query, base, support, labels, cosine)
    if any(not value.is_floating_point() for value in values):
        raise ValueError('Feature, label, logit and cosine tensors must be floating point')
    if any(value.device != query.device for value in (*values, valid)):
        raise ValueError('Reader inputs must share a device')
    if any(value.requires_grad for value in values):
        raise ValueError('Frozen audit inputs must not require gradients')
    if any(not torch.isfinite(value).all() for value in
           (query, base, support[valid], labels[valid], cosine[valid])):
        raise ValueError('Queries/base and valid support/label/cosine values must be finite')


def _project_values(attention, values):
    """Use the true packed MHA V parameters, including their learned bias."""
    dim, heads = attention.embed_dim, attention.num_heads
    bias = attention.in_proj_bias
    value_bias = None if bias is None else bias[2 * dim:]
    # Match MHA's sequence-first linear and head splitting. Retain the value
    # input captured from production (which already includes support labels).
    projected = F.linear(values.transpose(0, 1), attention.in_proj_weight[2 * dim:], value_bias)
    return projected.reshape(values.shape[1], values.shape[0], heads, dim // heads).permute(1, 2, 0, 3)


def _context(attention, values_by_head, weights):
    per_head = torch.matmul(weights.unsqueeze(-2), values_by_head)
    concatenated = per_head.transpose(1, 2).contiguous().reshape(weights.shape[0], 1, attention.embed_dim)
    return F.linear(concatenated, attention.out_proj.weight, attention.out_proj.bias)


@torch.no_grad()
def trace_weight_cases(head, query, base, support, labels, valid, cosine):
    """Return ``{'traces': {case: {'logits', 'weights'}}, 'replay': scalars}``.

    Each logit tensor is [N,4]; each weight tensor is [N,H,K+1]. Normal logits
    are the actual production output. Identity/uniform/cosine use identical
    captured V, MHA output projection, residual, LayerNorm and classification
    head. Numerical closeness is checked here; the calling audit additionally
    requires ZERO identity disagreements in both region and pixel argmax.
    A zero-query batch returns empty traces without invoking MHA, whose
    production kernel cannot reshape an empty batch on all supported versions.
    """
    _check_reader(head, query, base, support, labels, valid, cosine)
    if len(query) == 0:
        weights = query.new_empty((0, head.attention.num_heads, valid.shape[-1] + 1))
        return dict(traces={case: dict(logits=base.clone(), weights=weights.clone())
                            for case in CASES}, replay=_empty_replay())
    captured = {}

    def before(module, args, kwargs):
        if captured:
            raise RuntimeError('Expected exactly one production attention call')
        captured['args'], captured['kwargs'] = args, dict(kwargs)

    def after(module, args, output):
        captured['context'] = output[0]

    pre = head.attention.register_forward_pre_hook(before, with_kwargs=True)
    post = head.attention.register_forward_hook(after)
    try:
        normal = head(query, base, support, labels, valid, mode='memory')
    finally:
        pre.remove()
        post.remove()
    if len(captured.get('args', ())) != 3 or captured['kwargs'].get('need_weights') is not False:
        raise RuntimeError('Production reader must use explicit Q/K/V and need_weights=False')
    actual_query, keys, values = captured['args']
    if actual_query.shape != (len(query), 1, head.dim) or keys.shape != values.shape:
        raise RuntimeError('Unexpected production MHA inputs')
    if values.shape != (len(query), valid.shape[-1] + 1, head.dim):
        raise RuntimeError('Support/self slot layout changed')
    expected_mask = torch.cat((~valid, torch.zeros((len(query), 1), dtype=torch.bool, device=query.device)), -1)
    if not torch.equal(captured['kwargs'].get('key_padding_mask'), expected_mask):
        raise RuntimeError('Production support mask/self fallback changed')
    options = dict(captured['kwargs'], need_weights=True, average_attn_weights=False)
    observed_context, observed_weights = head.attention(*captured['args'], **options)
    weights = observed_weights[:, :, 0]
    _check_weight_inputs(weights, valid, cosine)
    observation_logits = base + head.ffn(head.norm(query + observed_context[:, 0]))
    projected_values = _project_values(head.attention, values)
    traces = dict(normal=dict(logits=normal, weights=weights))
    contexts = {}
    for case in CASES[1:]:
        case_weights = redistribute_external_weights(weights, valid, cosine, case)
        contexts[case] = _context(head.attention, projected_values, case_weights)
        logits = base + head.ffn(head.norm(query + contexts[case][:, 0]))
        if not torch.isfinite(logits).all():
            raise RuntimeError('Nonfinite intervention logits')
        traces[case] = dict(logits=logits, weights=case_weights)
    if not torch.isfinite(normal).all():
        raise RuntimeError('Nonfinite production logits')
    for name, context, logits in (
            ('observation', observed_context, observation_logits),
            ('identity', contexts['identity'], traces['identity']['logits'])):
        torch.testing.assert_close(context, captured['context'], atol=ATOL, rtol=RTOL,
                                   msg=name + ' context differs from production reader')
        torch.testing.assert_close(logits, normal, atol=ATOL, rtol=RTOL,
                                   msg=name + ' logits differ from production reader')
    replay = dict(
        max_abs_context=_max_abs(contexts['identity'] - captured['context']),
        max_abs_logits=_max_abs(traces['identity']['logits'] - normal),
        identity_region_argmax_disagreements=int((traces['identity']['logits'].argmax(-1) != normal.argmax(-1)).sum()),
        observation_max_abs_context=_max_abs(observed_context - captured['context']),
        observation_max_abs_logits=_max_abs(observation_logits - normal),
        observation_region_argmax_disagreements=int((observation_logits.argmax(-1) != normal.argmax(-1)).sum()),
        max_self_weight_error=0., max_external_mass_error=0., max_invalid_weight=0.,
        max_normalization_error=0.,
    )
    for trace in traces.values():
        current = trace['weights']
        replay['max_self_weight_error'] = max(replay['max_self_weight_error'], _max_abs(current[..., -1] - weights[..., -1]))
        replay['max_external_mass_error'] = max(replay['max_external_mass_error'], _max_abs(current[..., :-1].sum(-1) - weights[..., :-1].sum(-1)))
        replay['max_invalid_weight'] = max(replay['max_invalid_weight'], _max_abs(current[..., :-1].masked_select(~valid[:, None])))
        replay['max_normalization_error'] = max(replay['max_normalization_error'], _max_abs(current.sum(-1) - 1))
    if replay['max_self_weight_error'] != 0 or replay['max_invalid_weight'] != 0:
        raise RuntimeError('Self weights or support masks changed during redistribution')
    if replay['max_external_mass_error'] > 1e-5 or replay['max_normalization_error'] > 1e-5:
        raise RuntimeError('External mass or weight normalization changed')
    return dict(traces=traces, replay=replay)
