"""Frozen V2/Soft ROI content; independent S0/S1/S2/C2 full region readouts."""
import argparse
import json
import math
import time
import torch
from .core import (VARIANTS, capture, partition, make_heads, routing_stats, implementation_hash)
from bda_benchmark.experiments.damageformer_region_content_v1.core import verify_original_readout
from bda_benchmark.experiments.damageformer_region_evidence_v1.core import (
    load_fixed, lists, loader, parse_batch, metrics, cm_update, seed_everything,
    new_output, save_json, save_pt, source_hash, digest,
)
from bda_benchmark.experiments.damageformer_region_evidence_v1.train_heads import objective


def route(r, name, model, args):
    return partition(r, name, args.soft_strength, args.threshold, model.config['stride'])


@torch.no_grad()
def evaluate(model, heads, data, device, args):
    model.eval()
    for head in heads.values():
        head.eval()
    cms = {name: torch.zeros(4, 4, dtype=torch.long) for name in ('old_pixel', 'old_region', *heads)}
    stats = {name: {} for name in heads}
    count, identity_error, replay_error = 0, 0., 0.
    for batch in data:
        pre, post, _, gt, _ = parse_batch(batch, device)
        r = capture(model, pre, post)
        identity_error = max(identity_error, r['identity_max_abs'])
        replay_error = max(replay_error, verify_original_readout(model, r, gt.shape[-2:]))
        cm_update(cms['old_pixel'], gt, r['pixel'].argmax(1))
        cm_update(cms['old_region'], gt, r['prob'].argmax(1))
        for name, head in heads.items():
            routing = route(r, name, model, args)
            pred = head(r['features']['roi'], routing, gt.shape[-2:]).argmax(1)
            cm_update(cms[name], gt, pred)
            for key, value in routing_stats(routing, r['building']).items():
                stats[name][key] = stats[name].get(key, 0.) + value * len(pre)
        count += len(pre)
        if count % 25 == 0:
            print('evaluate', count, flush=True)
    return dict(metrics={k: metrics(cm) for k, cm in cms.items()}, image_count=count,
                routing={k: {j: v/max(1, count) for j, v in row.items()} for k, row in stats.items()},
                identity_max_abs=identity_error, region_replay_max_abs=replay_error, test_accessed=False)


def main():
    p = argparse.ArgumentParser(__doc__)
    for key in ('checkpoint', 'dataset', 'train_list', 'val_list', 'output'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--accum', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--soft_strength', type=float, default=1.)
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if min(a.steps, a.eval_every, a.batch, a.accum) <= 0 or a.batch*a.accum != 16:
        p.error('Positive settings and single-device batch*accum=16 required')
    if not math.isfinite(a.lr) or a.lr <= 0 or a.workers < 0:
        p.error('Invalid lr/workers')
    if not math.isfinite(a.soft_strength) or a.soft_strength < 0 or not 0 < a.threshold < 1:
        p.error('Invalid ROI constraint settings')
    if len(set(a.variants)) != len(a.variants):
        p.error('Duplicate variants')
    train, val = lists(a.train_list, a.val_list)
    device = torch.device(a.device)
    seed_everything(a.seed)
    model, checkpoint, sha = load_fixed(a.checkpoint, device)
    out = new_output(a.output)
    heads = make_heads(a.variants, model.config['dim'], device)
    opts = {k: torch.optim.AdamW(h.parameters(), lr=a.lr, weight_decay=.005) for k, h in heads.items()}
    frozen_buffers = {k: b.detach().cpu().clone() for k, b in model.named_buffers()}
    manifest = dict(arguments=vars(a), format='ROI_PARTITION_CONTROL_V1',
                    checkpoint_sha256=sha, checkpoint_step=checkpoint['step'],
                    frontend_source_hash=source_hash(), experiment_hash=implementation_hash(),
                    train_sha256=digest(a.train_list), val_sha256=digest(a.val_list),
                    feature_source='exact original pixel-head input including existing Soft ROI',
                    frozen_front=True, frozen_building=True, frozen_base_q=True,
                    learned_partition=False, test_accessed=False, smoke=a.smoke,
                    dim=model.config['dim'], stride=model.config['stride'],
                    parameters={k: sum(x.numel() for x in h.parameters()) for k, h in heads.items()},
                    variants=dict(s0='original Q', s1='one-shot probability compatibility reweight',
                                  s2='per-region predicted building/background split',
                                  c2='per-seed x-midpoint spatial split; ignores building'),
                    loss='region NLL+Lovasz; no auxiliary new loss or Memory',
                    budget_caution='S2/C2 same 2K slot ceiling, NOT guaranteed equal occupied token count',
                    checkpoint_policy='best+last readout-only per variant, no optimizer/resume/front/cache copy')
    save_json(out/'manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    tr = loader(a.dataset, train, True, a.batch, a.workers, seed=a.seed+2)
    vl = loader(a.dataset, val[:2] if a.smoke else val, workers=a.workers, seed=a.seed+3)
    initial = evaluate(model, heads, vl, device, a)
    save_json(out/'initial.json', initial)
    if not a.smoke and (initial['image_count'] != 450 or
            abs(initial['metrics']['old_region']['valid_miou']-checkpoint['valid_miou']) > 1e-5):
        raise RuntimeError('Full Val checkpoint reproduction failed; do not bypass')
    best = {k: -1. for k in heads}
    best_steps = {k: 0 for k in heads}
    it = iter(tr)
    start = time.monotonic()
    for step in range(1, a.steps+1):
        model.eval()
        warm = min(100, max(1, a.steps//10))
        factor = step/warm if step <= warm else .5*(1+math.cos(math.pi*(step-warm)/max(1, a.steps-warm)))
        totals = {k: 0. for k in heads}
        for name, head in heads.items():
            head.train(); opts[name].zero_grad(set_to_none=True)
            for group in opts[name].param_groups:
                group['lr'] = a.lr*factor
        for _ in range(a.accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(tr); batch = next(it)
            pre, post, _, gt, _ = parse_batch(batch, device)
            r = capture(model, pre, post)
            for name, head in heads.items():
                routing = route(r, name, model, a)
                loss = objective(head(r['features']['roi'], routing, gt.shape[-2:]), gt)/a.accum
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss: '+name)
                loss.backward(); totals[name] += float(loss.detach())
        for name, head in heads.items():
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            opts[name].step()
        if any(x.grad is not None for x in model.parameters()):
            raise RuntimeError('Frozen front has gradients')
        if step == 1 or step % 20 == 0:
            row = dict(step=step, crops=step*16, losses=totals, lr=a.lr*factor,
                       elapsed_seconds=time.monotonic()-start)
            with (out/'training.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            print(row, flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            result = evaluate(model, heads, vl, device, a)
            for key, value in model.named_buffers():
                if not torch.equal(value.detach().cpu(), frozen_buffers[key]):
                    raise RuntimeError('Frozen buffer changed: '+key)
            with (out/'validation.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=step, **result))+'\n')
            for name, head in heads.items():
                score = result['metrics'][name]['valid_miou']
                state = dict(head=head.state_dict(), step=step, name=name,
                             metrics=result['metrics'][name], manifest=manifest)
                if score > best[name]:
                    best[name] = score; best_steps[name] = step
                    save_pt(out/(name+'_best.pth'), state)
                if step == a.steps:
                    save_pt(out/(name+'_last.pth'), state)
            print('VAL', step, {k: v['valid_miou'] for k, v in result['metrics'].items()}, flush=True)
            print('ROUTING', result['routing'], flush=True)
    save_json(out/'summary.json', dict(complete=True, best=best, best_steps=best_steps,
                                     smoke=a.smoke, test_accessed=False))
    print('ROI PARTITION COMPLETE. Front/Q frozen. Train/Val only. No Test.', flush=True)


if __name__ == '__main__':
    main()
