"""Paired P/Q-read/Content-read training on the SAME frozen snapshot and crops."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from bda_benchmark.experiments.damageformer_srac_fourstage_v2.label_audit import validate_batch_labels
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import (
    lists, loader, new_output, parse_batch, save_json, save_pt,
)
from .models import ARMS, gradient_norms, make_arms
from .runtime import (
    FrozenEvidence, assert_independent, cosine_warmup, load_fixed, module_fingerprint,
    objective, reproduction_gate, upsample_probability,
)
from .evidence import Evidence

FORMAT = 'SRAC_SELECTIVE_READOUT_DIAGNOSTIC_V1'
PAIRS = (('a1', 'region'), ('a1', 'p'), ('p', 'q_read'),
         ('p', 'content_read'), ('q_read', 'content_read'))


def parser():
    p = argparse.ArgumentParser(__doc__)
    for key in ('checkpoint', 'a1_checkpoint', 'dataset', 'train_list', 'val_list', 'output'):
        p.add_argument('--' + key, required=True)
    for key, default in dict(steps=5000, eval_every=500, batch=2, accum=8, workers=4,
                             seed=3407, score_dim=32, chunk_size=4096).items():
        p.add_argument('--' + key, type=int, default=default)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=5e-3)
    p.add_argument('--amp', choices=('none', 'bf16'), default='none')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--smoke', action='store_true')
    return p


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, allow_nan=False) + '\n')


@torch.no_grad()
def evaluate(source, arms, data, device, check_initial=False):
    for arm in arms.values():
        arm.eval()
    evidence = Evidence(('a1', 'region', *ARMS), PAIRS)
    max_initial_error = {name: 0.0 for name in ARMS}
    route_rows = {name: [] for name in ARMS}
    for index, batch in enumerate(data, 1):
        if not validate_batch_labels(batch, 0, index)['valid']:
            raise ValueError('Invalid validation labels')
        pre, post, _loc, gt, names = parse_batch(batch, device)
        fixed = source(pre, post)
        probabilities = dict(a1=fixed['old_pixel'].float().softmax(1), region=fixed['prob'])
        for name, arm in arms.items():
            output = arm(fixed)
            probability = upsample_probability(output['logits'], gt.shape[-2:])
            if not torch.isfinite(probability).all():
                raise RuntimeError('Non-finite validation probability: ' + name)
            probabilities[name] = probability
            route_rows[name].append(output['routing'])
            if check_initial:
                error = float((probability - probabilities['a1']).abs().max())
                max_initial_error[name] = max(error, max_initial_error[name])
                if error > 2e-5:
                    raise RuntimeError('Initial arm does not reproduce A1: ' + name)
        evidence.update(gt, {key: p.argmax(1) for key, p in probabilities.items()}, fixed['building'], names[0])
        if index % 25 == 0:
            print(f'evaluate selective readout {index}', flush=True)
    result = evidence.finalize()
    result['routing_mean_per_image'] = {
        name: {key: sum(float(row[key]) for row in rows) / len(rows) for key in rows[0]}
        if rows else {} for name, rows in route_rows.items()
    }
    if check_initial:
        result['initial_max_abs_probability_error_to_a1'] = max_initial_error
    return result


def validate_args(args):
    if args.smoke:
        args.steps, args.eval_every = 20, 20
    if min(args.steps, args.eval_every, args.batch, args.accum, args.score_dim, args.chunk_size) <= 0:
        raise ValueError('Counts and dimensions must be positive')
    if args.workers < 0 or args.batch * args.accum != 16:
        raise ValueError('Use nonnegative workers and batch*accum=16')
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError('Invalid optimizer settings')
    if Path(args.output).exists():
        raise ValueError('Output exists; preserve it and choose a new path: ' + args.output)


def run(args):
    validate_args(args)
    started = time.perf_counter()
    device = torch.device(args.device)
    if args.amp == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('BF16 unavailable; use --amp none')
    train_names, val_names = lists(args.train_list, args.val_list)
    front, payload, provenance = load_fixed(args)
    # CUDA is initialized by the loaded model, before peak-statistics reset.
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    source = FrozenEvidence(front)
    source_before = module_fingerprint(front)
    arms = {key: value.to(device) for key, value in make_arms(front, args.score_dim, args.chunk_size).items()}
    assert_independent(front, arms)
    before = {key: module_fingerprint(value) for key, value in arms.items()}
    if len(set(before.values())) != 1:
        raise RuntimeError('Arm initial weights differ')
    optimizers = {key: torch.optim.AdamW([p for p in value.parameters() if p.requires_grad],
                                       lr=args.lr, weight_decay=args.weight_decay)
                  for key, value in arms.items()}
    manifest = dict(**provenance, format=FORMAT, complete=False, memory_enabled=False,
                    weights_updated='new independent no-Memory readers/classifiers and content scorer only',
                    steps_are='optimizer updates per arm, shared crop stream', effective_batch=16,
                    expected_crops_per_arm=args.steps * 16, initial_parameter_fingerprints=before,
                    trainable_parameters={key: sum(p.numel() for p in arm.parameters() if p.requires_grad)
                                          for key, arm in arms.items()},
                    allocated_parameters={key: sum(p.numel() for p in arm.parameters()) for key, arm in arms.items()},
                    capacity_warning='Same allocation/initialization; NOT equal active capacity. Content-read adds trainable score projections; capacity control needed if positive.',
                    loss='unweighted probability NLL + Lovasz-Softmax; same target, no class/edge/Memory loss',
                    boundary='unchanged native boundary in all arms',
                    schedule='warmup=min(1000,steps//10), cosine to zero',
                    classifier='A1-copied pixel block + zero context block; both learn except P context disabled')
    out = new_output(args.output)
    save_json(out / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    train_data = loader(args.dataset, train_names, train=True, batch=args.batch,
                        workers=args.workers, seed=args.seed + 2)
    val_data = loader(args.dataset, val_names[:2] if args.smoke else val_names,
                      workers=args.workers, seed=args.seed + 3)
    try:
        initial = evaluate(source, arms, val_data, device, check_initial=True)
    except Exception as error:
        save_json(out / 'step0_failure.json', dict(complete=False, optimizer_updates=0,
                  exception_type=type(error).__name__, message=str(error), test_accessed=False))
        raise
    gate = reproduction_gate(initial, payload, args.smoke)
    save_json(out / 'step0.json', dict(**initial, reproduction=gate))
    if gate['passed'] is False:
        raise RuntimeError('Step-0 native reproduction failed. Evidence saved; no optimizer update performed')
    iterator, crops = iter(train_data), 0
    crop_stream = hashlib.sha256()
    best = {key: dict(score=-1.0, step=None) for key in ARMS}
    gradients_verified = False
    for step in range(1, args.steps + 1):
        lr = args.lr * cosine_warmup(step, args.steps)
        for name, arm in arms.items():
            arm.train()
            optimizers[name].zero_grad(set_to_none=True)
            for group in optimizers[name].param_groups:
                group['lr'] = lr
        losses, gradients = {key: 0.0 for key in ARMS}, {}
        routing = {}
        for micro in range(args.accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_data)
                batch = next(iterator)
            label_report = validate_batch_labels(batch, step, micro)
            if not label_report['valid']:
                save_json(out / 'invalid_labels.json', label_report)
                raise ValueError('Invalid labels; no remapping permitted')
            pre, post, _loc, target, names = parse_batch(batch, device)
            crops += len(pre)
            crop_stream.update(json.dumps([step, micro, names]).encode())
            fixed = source(pre, post)
            q_before = fixed['q'].clone()
            first = step == 1 and micro == 0
            saved_gradients = {}
            for index, (name, arm) in enumerate(arms.items()):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp == 'bf16'):
                    result = arm(fixed)
                    probability = upsample_probability(result['logits'], target.shape[-2:])
                    loss = objective(probability, target)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss in ' + name)
                (loss / args.accum).backward()
                losses[name] += float(loss.detach()) / args.accum
                routing[name] = result['routing']
                if first:
                    for other in ARMS[index + 1:]:
                        if any(p.grad is not None for p in arms[other].parameters()):
                            raise RuntimeError('Backward leaked into another arm')
                    for previous, values in saved_gradients.items():
                        parameters = dict(arms[previous].named_parameters())
                        if any(not torch.equal(parameters[key].grad, grad) for key, grad in values.items()):
                            raise RuntimeError('Backward changed another arm gradient')
                    saved_gradients[name] = {key: p.grad.clone() for key, p in arm.named_parameters() if p.grad is not None}
            if not torch.equal(q_before, fixed['q']) or any(p.grad is not None for p in front.parameters()):
                raise RuntimeError('Frozen evidence was changed or accumulated gradients')
            if first:
                gradients_verified = True
        clipping = {}
        for name, arm in arms.items():
            gradients[name] = gradient_norms(arm)
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in arm.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True,
            )
            clipping[name] = float(norm)
            optimizers[name].step()
        row = dict(step=step, crops_per_arm=crops, loss=losses, lr=lr,
                   gradient_norm_by_module=gradients, preclip_grad_norm=clipping,
                   routing_last_microbatch=routing)
        append_json(out / 'training.jsonl', row)
        if step == 1 or step % 20 == 0:
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            evaluation = evaluate(source, arms, val_data, device)
            validation_gate = reproduction_gate(evaluation, payload, args.smoke)
            append_json(out / 'validation.jsonl', dict(step=step, metrics=evaluation['metrics'],
                        per_event=evaluation['per_event'], routing=evaluation['routing_mean_per_image'],
                        reproduction=validation_gate))
            if validation_gate['passed'] is False:
                save_json(out / 'failed_validation.json', evaluation)
                raise RuntimeError('Frozen baseline reproduction changed during validation')
            for name, arm in arms.items():
                score = evaluation['metrics'][name]['valid_miou']
                state = dict(format=FORMAT, mode=name, model=arm.state_dict(), step=step, valid_miou=score,
                             manifest=manifest, run_complete=False, reusable_as_parent=False)
                if score > best[name]['score']:
                    best[name] = dict(score=score, step=step)
                    save_pt(out / f'best_{name}.pth', state)
                if step == args.steps:
                    save_pt(out / f'last_{name}.pth', state)
            print('VAL ' + json.dumps(dict(step=step, scores={key: value['valid_miou']
                      for key, value in evaluation['metrics'].items()})), flush=True)
    unchanged = module_fingerprint(front) == source_before
    summary = dict(complete=unchanged, smoke=args.smoke, test_accessed=False, diagnostic_only=True,
                   reusable_as_parent=False, best=best, final_metrics=evaluation['metrics'],
                   final_step=args.steps, crops_per_arm=crops, shared_crop_name_stream_sha256=crop_stream.hexdigest(),
                   source_weights_and_buffers_unchanged=unchanged, independent_gradients_verified=gradients_verified,
                   seconds=time.perf_counter() - started,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                   initial_fingerprints=before, final_fingerprints={key: module_fingerprint(arm) for key, arm in arms.items()},
                   checkpoint_storage='readers/classifiers/scorer only; no optimizer or A1 copy; inference artifacts, NOT exact-resume states')
    save_json(out / 'audit_final.json', dict(**evaluation, **summary))
    save_json(out / 'summary.json', summary)
    if not unchanged:
        raise RuntimeError('Source parameters/buffers changed; run is invalid')
    for name in ARMS:
        for prefix in ('best', 'last'):
            path = out / f'{prefix}_{name}.pth'
            state = torch.load(path, map_location='cpu', weights_only=False)
            state.update(run_complete=True, completed_steps=args.steps)
            save_pt(path, state)
    manifest['complete'] = True
    save_json(out / 'manifest.json', manifest)
    print('SELECTIVE READOUT COMPLETE. Frozen parent unchanged. No Memory. No Test.', flush=True)


if __name__ == '__main__':
    run(parser().parse_args())
