"""Matched fixed/learned Q branches on immutable V2 pixel-input ROI content.

Only this new module bypasses the historical no-grad capture and Q-detached
Readout: frozen capture finishes first; a copied decoder and differentiable
gather/read path run afterward. No Memory, backbone, gate, or auxiliary loss.
"""
import copy
import math

import torch
from torch import nn

from bda_benchmark.experiments.damageformer_iterative_partition_v1.model import (
    candidate_map, iterative_assignment,
)
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import partition
from .common import capture, state_digest
from .routing import forward_separate, route_from_q


@torch.no_grad()
def capture_q_inputs(front, pre, post):
    """Capture FIRST RGB encoder call, not the modulated second RGB call."""
    if front.training or any(p.requires_grad or p.grad is not None for p in front.parameters()):
        raise RuntimeError('Q input extraction requires an immutable eval-only front')
    rgb, decoder_inputs = [], []
    calls = 0

    def first_rgb(_module, _inputs, output):
        nonlocal calls
        calls += 1
        if calls == 1:
            rgb.extend(front.base._to_nchw(x).detach() for x in output[1:])

    def original_decoder_inputs(_module, inputs):
        decoder_inputs.append(tuple(x.detach() for x in inputs[0]))

    handles = [front.base.encoder_1.register_forward_hook(first_rgb),
               front.decoder.register_forward_pre_hook(original_decoder_inputs)]
    try:
        reference = capture(front, pre, post)
    finally:
        for handle in handles:
            handle.remove()
    if calls != 2 or len(rgb) != 4 or len(decoder_inputs) != 1:
        raise RuntimeError('Expected V2: two RGB calls, one four-stage decoder call')
    if any(not torch.equal(a, b) for a, b in zip(rgb, decoder_inputs[0])):
        raise RuntimeError('First RGB pyramid differs from original partition input')
    if any(x.requires_grad or x.grad_fn is not None for x in rgb):
        raise RuntimeError('Frozen RGB pyramid retained a gradient graph')
    return reference, tuple(rgb)


def decoder_q(decoder, pyramid, reference, config):
    """Replay original ungrouped clustering; B only splits subsequent routing."""
    embedding = decoder(tuple(x.detach() for x in pyramid))
    shape = tuple(embedding.shape[-2:])
    if shape != tuple(reference['shape']):
        raise RuntimeError('Copied decoder changed native Q resolution')
    ids, valid, gh, gw = candidate_map(*shape, config['stride'], embedding.device)
    if not torch.equal(ids, reference['ids']) or gh*gw != reference['tokens'].shape[1]:
        raise RuntimeError('Copied decoder changed ordered base slot IDs')
    # This zeros_like is essential: passing predicted B would silently introduce
    # a second hard split INSIDE clustering, unlike the frozen SRACSwin V2.
    return iterative_assignment(embedding, ids, valid,
                                torch.zeros_like(reference['building']), gh*gw,
                                config['stride'], config['iterations'])


class QBranch(nn.Module):
    """Owns ONLY copied decoder and complete source Readout parameters."""
    def __init__(self, decoder, readout, config, learn_q):
        super().__init__()
        self.decoder = copy.deepcopy(decoder)
        self.readout = copy.deepcopy(readout)
        self.config = {k: config[k] for k in ('stride', 'iterations')}
        self.learn_q = bool(learn_q)
        self.decoder.requires_grad_(self.learn_q)
        self.readout.requires_grad_(True)
        for parameter in self.parameters():
            parameter.grad = None
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        if not self.learn_q:
            self.decoder.eval()
        return self

    def forward(self, reference, pyramid, output_size, *, pool_mode='hard',
                read_mode='hard', epsilon=.05, threshold=.5):
        if pool_mode not in ('hard', 'exchange') or read_mode not in ('hard', 'exchange'):
            raise ValueError('Q branches require hard/exchange gather and read routing')
        q = (decoder_q(self.decoder, pyramid, reference, self.config) if self.learn_q
             else reference['q'].detach())
        def route(mode):
            return route_from_q(q, reference['ids'], reference['tokens'].shape[1],
                                reference['shape'], reference['building'],
                                epsilon=0. if mode == 'hard' else epsilon,
                                threshold=threshold)
        gather, read = route(pool_mode), route(read_mode)
        result = forward_separate(self.readout, reference['features']['roi'],
                                  gather, read, output_size)
        return dict(result, q=q, pool_route=gather, read_route=read)


def make_branches(front, source_readout):
    branches = {name: QBranch(front.decoder, source_readout, front.config, learn)
                for name, learn in (('fixed_q', False), ('learn_q', True))}
    for member in ('decoder', 'readout'):
        a, b = (getattr(branches[name], member) for name in ('fixed_q', 'learn_q'))
        if state_digest(a) != state_digest(b):
            raise RuntimeError('Q branches did not initialize identically: '+member)
    modules = [front, source_readout, *branches.values()]
    storages = [{p.untyped_storage().data_ptr() for p in m.parameters()} for m in modules]
    if any(a & b for i, a in enumerate(storages) for b in storages[i+1:]):
        raise RuntimeError('Q branches share parameter storage with each other or source')
    return branches


@torch.no_grad()
def verify_initial_replay(branches, source_head, reference, pyramid, output_size,
                          *, pool_mode='hard', read_mode='hard', epsilon=.05,
                          threshold=.5, atol=2e-5, rtol=2e-4):
    """Use real inputs before any update; reject accidentally changed clustering."""
    errors = {}
    kwargs = dict(pool_mode=pool_mode, read_mode=read_mode,
                  epsilon=epsilon, threshold=threshold)
    outputs = {}
    for name, branch in branches.items():
        q = decoder_q(branch.decoder, pyramid, reference, branch.config)
        torch.testing.assert_close(q, reference['q'], atol=atol, rtol=rtol,
                                   msg=name+' initial base Q replay')
        errors[name+'_q_max_abs'] = float((q-reference['q']).abs().max())
        outputs[name] = branch(reference, pyramid, output_size, **kwargs)
    a, b = outputs['fixed_q'], outputs['learn_q']
    torch.testing.assert_close(a['prob'], b['prob'], atol=atol, rtol=rtol,
                               msg='Initial fixed/learned full-readout replay')
    errors['branch_prob_max_abs'] = float((a['prob']-b['prob']).abs().max())
    replay = forward_separate(source_head, reference['features']['roi'],
                              a['pool_route'], a['read_route'], output_size)['prob']
    torch.testing.assert_close(a['prob'], replay, atol=atol, rtol=rtol,
                               msg='Initial copied complete source Readout replay')
    errors['source_prob_max_abs'] = float((a['prob']-replay).abs().max())
    if pool_mode == read_mode:
        legacy_route = partition(reference, pool_mode, epsilon=epsilon,
                                 threshold=threshold, stride=branches['fixed_q'].config['stride'])
        legacy = source_head(reference['features']['roi'], legacy_route, output_size)
        torch.testing.assert_close(a['prob'], legacy, atol=atol, rtol=rtol,
                                   msg='New differentiable readout versus historical Readout')
        errors['legacy_prob_max_abs'] = float((a['prob']-legacy).abs().max())
    return dict(passed=True, atol=atol, rtol=rtol, **errors)


def gradient_report(branches, front):
    """Inspect actual backward results; a trainable flag alone is not evidence."""
    if front.training or any(p.requires_grad or p.grad is not None for p in front.parameters()):
        raise RuntimeError('Frozen front received gradients or became trainable')
    report = {}
    for name, branch in branches.items():
        row = {}
        for member in ('decoder', 'readout'):
            module = getattr(branch, member)
            named = list(module.named_parameters())
            if member == 'decoder' and not branch.learn_q:
                if module.training or any(p.requires_grad or p.grad is not None for _, p in named):
                    raise RuntimeError('Fixed-Q copied decoder changed gradient policy')
                row[member] = dict(parameters_with_gradient=0, norm=0., nonzero=False)
                continue
            if any(p.grad is None for _, p in named):
                missing = [key for key, p in named if p.grad is None]
                raise RuntimeError(name+' disconnected '+member+': '+str(missing))
            if any(not torch.isfinite(p.grad).all() for _, p in named):
                raise RuntimeError(name+' nonfinite '+member+' gradient')
            norm = math.sqrt(sum(float(p.grad.detach().double().square().sum()) for _, p in named))
            row[member] = dict(parameters_with_gradient=len(named), norm=norm, nonzero=norm > 0)
        report[name] = row
    return report
