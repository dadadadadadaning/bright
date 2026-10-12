"""One-factor readout comparison; Q and region construction are not retrained."""
from __future__ import annotations

import copy

import torch
from torch import nn

from bda_benchmark.experiments.damageformer_srac_joint_context_v1.model import (
    ContextMemoryReader, IdentityJointClassifier,
)
from .core import LocalSelectiveReadout


ARMS = ('p', 'q_read', 'content_read')


class SelectiveArm(nn.Module):
    def __init__(self, classifier, dim=128, score_dim=32, chunk_size=4096):
        super().__init__()
        self.reader = ContextMemoryReader(dim=dim, num_events=1, heads=4, slots=8)
        self.readout = LocalSelectiveReadout(256, dim, score_dim, chunk_size)
        self.classifier = IdentityJointClassifier(classifier, dim)
        self.mode = 'content_read'

    def configure(self, mode):
        if mode not in ARMS:
            raise ValueError(mode)
        self.mode = mode
        self.requires_grad_(True)
        # These metadata branches are not used by a no-Memory reader.
        for part in ('disaster_embedding', 'event_embedding', 'geometry_embedding'):
            getattr(self.reader, part).requires_grad_(False)
        if mode != 'content_read':
            self.readout.requires_grad_(False)
        if mode == 'p':
            self.reader.requires_grad_(False)
            self.classifier.context.requires_grad_(False)
        return self

    def forward(self, fixed):
        pixel, tokens = fixed['pixel_feature'], fixed['tokens']
        if tuple(pixel.shape[-2:]) != tuple(fixed['shape']):
            raise ValueError('A1 pixel feature and Q grid differ; do not silently resize evidence')
        if self.mode == 'p':
            context = pixel.new_zeros(pixel.shape[0], tokens.shape[-1], *pixel.shape[-2:])
            stats = {}
        else:
            queries = tokens.reshape(-1, tokens.shape[-1])
            chunks = [self.reader(queries[start:start + 256], mode='nomemory')
                      for start in range(0, len(queries), 256)]
            enhanced = torch.cat(chunks).reshape_as(tokens)
            enhanced = enhanced.masked_fill(~fixed['valid_regions'][..., None], 0)
            context, stats = self.readout(
                pixel, enhanced, fixed['q'], fixed['ids'], fixed['valid_regions'],
                mode='q' if self.mode == 'q_read' else 'content',
            )
        return dict(logits=self.classifier(pixel, context), context=context, routing=stats)


def make_arms(front, score_dim=32, chunk_size=4096):
    template = SelectiveArm(front.base.clf_2, front.config['dim'], score_dim, chunk_size)
    return {name: copy.deepcopy(template).configure(name) for name in ARMS}


def gradient_norms(arm):
    result = {}
    for part in ('classifier', 'reader', 'readout'):
        values = [p.grad.detach().float().norm().square()
                  for p in getattr(arm, part).parameters() if p.grad is not None]
        result[part] = float(torch.stack(values).sum().sqrt()) if values else 0.0
    return result
