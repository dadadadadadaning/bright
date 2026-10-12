"""Frozen V2 front: raw/normal/ROI x pixel/region, identical fresh heads.

This is a diagnostic, not full-model training or a claim of SRAC reproduction.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import torch
from .core import capture
from bda_benchmark.experiments.damageformer_region_evidence_v1.core import (
    MatchedHead, load_fixed, lists, loader, parse_batch, metrics, cm_update,
    seed_everything, new_output, save_json, save_pt, source_hash, digest,
)
from bda_benchmark.experiments.damageformer_region_evidence_v1.train_heads import objective


@torch.no_grad()
def evaluate(model, heads, data, device):
    model.eval()
    for h in heads.values():
        h.eval()
    cms = {k: torch.zeros(4, 4, dtype=torch.long) for k in ['old_pixel', 'old_region', *heads]}
    count, error = 0, 0.
    for batch in data:
        pre, post, _, gt, _ = parse_batch(batch, device)
        r = capture(model, pre, post)
        error = max(error, r['identity_max_abs'])
        cm_update(cms['old_pixel'], gt, r['pixel'].argmax(1))
        cm_update(cms['old_region'], gt, r['prob'].argmax(1))
        for name, head in heads.items():
            source, kind = name.split('__')
            prob = head(r['features'][source], r, kind, gt.shape[-2:])
            cm_update(cms[name], gt, prob.argmax(1))
        count += len(pre)
        if count % 25 == 0:
            print('evaluate', count, flush=True)
    return dict(metrics={k: metrics(v) for k, v in cms.items()}, image_count=count,
                identity_max_abs=error, test_accessed=False)


def main():
    p = argparse.ArgumentParser(__doc__)
    for key in ('checkpoint', 'dataset', 'train_list', 'val_list', 'output'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--sources', nargs='+', choices=['raw', 'normal', 'roi'], default=['raw', 'normal', 'roi'])
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--accum', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if min(a.steps, a.eval_every, a.batch, a.accum, a.lr) <= 0 or a.batch*a.accum != 16:
        p.error('Positive settings and batch*accum=16 required')
    if len(set(a.sources)) != len(a.sources):
        p.error('Duplicate sources')
    train, val = lists(a.train_list, a.val_list)
    out = new_output(a.output)
    device = torch.device(a.device)
    seed_everything(a.seed)
    model, checkpoint, sha = load_fixed(a.checkpoint, device)
    proto = MatchedHead().to(device)
    heads = {s+'__'+k: copy.deepcopy(proto) for s in a.sources for k in ('pixel', 'region')}
    del proto
    opts = {k: torch.optim.AdamW(h.parameters(), lr=a.lr, weight_decay=.005) for k, h in heads.items()}
    own = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        own.update(path.name.encode()); own.update(path.read_bytes())
    manifest = dict(arguments=vars(a), checkpoint_sha256=sha, checkpoint_step=checkpoint['step'],
                    frontend_source_hash=source_hash(), experiment_hash=own.hexdigest(),
                    train_sha256=digest(a.train_list), val_sha256=digest(a.val_list),
                    frozen_front=True, fixed_q=True, test_accessed=False,
                    parameters={k: sum(x.numel() for x in h.parameters()) for k, h in heads.items()},
                    sources=dict(raw='first fused feature; no input modulation, no Soft ROI',
                                 normal='second fused feature; input modulation, no Soft ROI',
                                 roi='exact original pixel classifier input, including Soft ROI'),
                    note='Same fresh MLP; NOT equal to old pixel head; mean pooling only')
    save_json(out/'manifest.json', manifest)
    tr = loader(a.dataset, train, True, a.batch, a.workers, seed=a.seed+2)
    vl = loader(a.dataset, val[:2] if a.smoke else val, workers=a.workers, seed=a.seed+3)
    initial = evaluate(model, heads, vl, device)
    save_json(out/'initial.json', initial)
    if not a.smoke:
        if initial['image_count'] != 450 or abs(initial['metrics']['old_region']['valid_miou']-checkpoint['valid_miou']) > 1e-5:
            raise RuntimeError('Full Val checkpoint reproduction failed')
    best = {k: -1. for k in heads}
    it = iter(tr)
    for step in range(1, a.steps+1):
        warm = min(100, max(1, a.steps//10))
        factor = step/warm if step <= warm else .5*(1+math.cos(math.pi*(step-warm)/max(1, a.steps-warm)))
        totals = {k: 0. for k in heads}
        for k, h in heads.items():
            h.train(); opts[k].zero_grad(set_to_none=True)
            for g in opts[k].param_groups:
                g['lr'] = a.lr*factor
        for _ in range(a.accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(tr); batch = next(it)
            pre, post, _, gt, _ = parse_batch(batch, device)
            r = capture(model, pre, post)
            for name, head in heads.items():
                source, kind = name.split('__')
                loss = objective(head(r['features'][source], r, kind, gt.shape[-2:]), gt)/a.accum
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss: '+name)
                loss.backward(); totals[name] += float(loss.detach())
        for name, head in heads.items():
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            opts[name].step()
        if any(x.grad is not None for x in model.parameters()):
            raise RuntimeError('Frozen front has gradients')
        if step == 1 or step % 20 == 0:
            row = dict(step=step, crops=step*16, losses=totals, lr=a.lr*factor)
            with (out/'training.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            print(row, flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            result = evaluate(model, heads, vl, device)
            with (out/'validation.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=step, **result))+'\n')
            for name, head in heads.items():
                score = result['metrics'][name]['valid_miou']
                state = dict(head=head.state_dict(), step=step, name=name, metrics=result['metrics'][name], manifest=manifest)
                if score > best[name]:
                    best[name] = score; save_pt(out/(name+'_best.pth'), state)
                if step == a.steps:
                    save_pt(out/(name+'_last.pth'), state)
            print('VAL', step, {k: v['valid_miou'] for k, v in result['metrics'].items()}, flush=True)
    save_json(out/'summary.json', dict(complete=True, best=best, smoke=a.smoke, test_accessed=False))


if __name__ == '__main__':
    main()
