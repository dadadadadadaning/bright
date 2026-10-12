"""Three independent experiments. No GT in forward; no original model edits."""
import copy
import math

import torch
from torch import nn

from bda_benchmark.experiments.damageformer_roi_followup_v1.common import state_digest
from bda_benchmark.experiments.damageformer_roi_followup_v1.q_model import (
    capture_q_inputs, decoder_q,
)
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import (
    route_from_q, forward_separate,
)
from bda_benchmark.experiments.damageformer_iterative_partition_v1.model import candidate_map
from .generator import dual_assignment
from .exchange import exchange_pool_route

SUITES = {
    'structure': ('fixed_q', 'learn_q', 'learn_q_structure'),
    'multimodal': ('rgb_rgb', 'rgb_sar'),
    'support': ('hard', 'uniform', 'sar'),
}
PAIRS = {
    'structure': (('fixed_q', 'learn_q'), ('learn_q', 'learn_q_structure')),
    'multimodal': (('rgb_rgb', 'rgb_sar'),),
    'support': (('hard', 'uniform'), ('uniform', 'sar'), ('hard', 'sar')),
}


@torch.no_grad()
def capture_inputs(front, pre, post):
    """Reuse ONE old forward; capture real SAR pyramid and pre-fusion SAR content.

    RGB pyramid capture is already restricted to the FIRST (original) RGB call.
    The old second modulated call is only used for the immutable old_region check.
    """
    sar, sar_fused = [], []

    def sar_encoder(_module, _inputs, output):
        sar.append(tuple(front.base._to_nchw(x).detach() for x in output[1:]))

    def sar_fusion(_module, _inputs, output):
        sar_fused.append(output.detach())

    handles = [front.base.encoder_2.register_forward_hook(sar_encoder),
               front.base.fusion_layer_2.register_forward_hook(sar_fusion)]
    try:
        reference, rgb = capture_q_inputs(front, pre, post)
    finally:
        for handle in handles:
            handle.remove()
    if len(sar) != 1 or len(sar[0]) != 4 or len(sar_fused) != 1:
        raise RuntimeError('Expected one four-stage SAR encoder and one SAR fusion call')
    if any(a.shape != b.shape for a, b in zip(rgb, sar[0])):
        raise RuntimeError('RGB/SAR pyramids do not share the expected native grids')
    if sar_fused[0].shape[-2:] != tuple(reference['shape']):
        raise RuntimeError('SAR support descriptor is not on the native Q grid')
    if any(x.requires_grad or x.grad_fn is not None for x in (*rgb, *sar[0], sar_fused[0])):
        raise RuntimeError('Frozen feature capture retained an autograd graph')
    return reference, dict(rgb=rgb, sar=sar[0], sar_feature=sar_fused[0])


def hard_route(reference, q, threshold):
    return route_from_q(q, reference['ids'], reference['tokens'].shape[1],
                        reference['shape'], reference['building'], epsilon=0., threshold=threshold)


class Branch(nn.Module):
    def __init__(self, front, source_head, variant, *, beta=.2,
                 epsilon=.05, threshold=.5, support_temperature=.2):
        super().__init__()
        if variant not in sum(SUITES.values(), ()):
            raise ValueError('Unknown variant: '+variant)
        if not math.isfinite(beta) or not 0 < beta < 1:
            raise ValueError('Both modality branches require 0 < beta < 1')
        if not math.isfinite(epsilon) or not 0 <= epsilon <= .5 or not 0 < threshold < 1:
            raise ValueError('Invalid source routing settings')
        if not math.isfinite(support_temperature) or support_temperature <= 0:
            raise ValueError('Invalid support temperature')
        self.variant, self.beta = variant, float(beta)
        self.epsilon, self.threshold = float(epsilon), float(threshold)
        self.support_temperature = float(support_temperature)
        self.config = {k: front.config[k] for k in ('stride', 'iterations')}
        self.readout = copy.deepcopy(source_head).requires_grad_(True)
        if variant in SUITES['structure']:
            self.decoder = copy.deepcopy(front.decoder)
            self.decoder.requires_grad_(variant != 'fixed_q')
        elif variant in SUITES['multimodal']:
            self.decoder = copy.deepcopy(front.decoder).requires_grad_(True)
            self.second_decoder = copy.deepcopy(front.decoder).requires_grad_(True)
        for parameter in self.parameters():
            parameter.grad = None
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        if self.variant == 'fixed_q':
            self.decoder.eval()
        return self

    def forward(self, reference, inputs, output_size):
        q = reference['q'].detach()
        if self.variant in ('learn_q', 'learn_q_structure'):
            q = decoder_q(self.decoder, inputs['rgb'], reference, self.config)
        elif self.variant in SUITES['multimodal']:
            rgb = self.decoder(tuple(x.detach() for x in inputs['rgb']))
            second_input = inputs['sar'] if self.variant == 'rgb_sar' else inputs['rgb']
            other = self.second_decoder(tuple(x.detach() for x in second_input))
            if tuple(rgb.shape[-2:]) != tuple(reference['shape']):
                raise RuntimeError('Descriptor decoder changed the native Q grid')
            ids, valid, gh, gw = candidate_map(*reference['shape'], self.config['stride'], rgb.device)
            if not torch.equal(ids, reference['ids']) or gh*gw != reference['tokens'].shape[1]:
                raise RuntimeError('Changed ordered base seed IDs')
            q = dual_assignment(rgb, other, ids, valid, gh*gw, self.config['stride'],
                                iterations=self.config['iterations'], beta=self.beta)
        read = hard_route(reference, q, self.threshold)
        pool, extra = read, {}
        if self.variant in ('uniform', 'sar'):
            pool, extra = exchange_pool_route(reference, inputs['sar_feature'],
                epsilon=self.epsilon, threshold=self.threshold,
                temperature=self.support_temperature, mode=self.variant)
        result = forward_separate(self.readout, reference['features']['roi'], pool, read, output_size)
        return dict(result, q=q, pool_route=pool, read_route=read, exchange_diagnostics=extra)


def make_branches(front, source_head, suite, **kwargs):
    if suite not in SUITES:
        raise ValueError('Unknown suite')
    branches = {name: Branch(front, source_head, name, **kwargs) for name in SUITES[suite]}
    for branch in branches.values():
        if state_digest(branch.readout) != state_digest(source_head):
            raise RuntimeError('Readout did not initialize from the same source')
        for name in ('decoder', 'second_decoder'):
            if hasattr(branch, name) and state_digest(getattr(branch, name)) != state_digest(front.decoder):
                raise RuntimeError('Copied decoder initialization changed')
    modules = [front, source_head, *branches.values()]
    storages = [{p.untyped_storage().data_ptr() for p in module.parameters()} for module in modules]
    if any(a & b for i, a in enumerate(storages) for b in storages[i+1:]):
        raise RuntimeError('Independent branches share parameter storage')
    return branches


def member_hashes(branches):
    return {name: {key: state_digest(module) for key, module in branch.named_children()}
            for name, branch in branches.items()}


def gradients(branches, front):
    if front.training or any(p.requires_grad or p.grad is not None for p in front.parameters()):
        raise RuntimeError('Original frozen front received gradients')
    result = {}
    for name, branch in branches.items():
        result[name] = {}
        for member, module in branch.named_children():
            parameters = list(module.parameters())
            active = [p for p in parameters if p.requires_grad]
            if not active:
                if module.training or any(p.grad is not None for p in parameters):
                    raise RuntimeError('Frozen member changed policy: '+name+'/'+member)
            elif any(p.grad is None for p in active):
                raise RuntimeError('Disconnected gradient: '+name+'/'+member)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise RuntimeError('Nonfinite gradient: '+name+'/'+member)
            norm = math.sqrt(sum(float(p.grad.detach().double().square().sum())
                                 for p in active if p.grad is not None))
            result[name][member] = dict(norm=norm, nonzero=norm > 0,
                                        parameters_with_gradient=sum(p.grad is not None for p in parameters))
    return result
