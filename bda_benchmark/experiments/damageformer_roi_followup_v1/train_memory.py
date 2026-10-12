"""Paired Memory/no-memory heads on one frozen token space and routing."""
import argparse
import math
import time

import torch

from . import common
from .memory import (HEAD_FORMAT, read_bank, device_bank, make_heads, classify_regions,
                     add_statistics, finish_statistics)
from .routing import encode_regions, decode_regions
from bda_benchmark.experiments.damageformer_roi_exchange_v1.diagnostics import DiagnosticAccumulator


@torch.no_grad()
def evaluate(bundle, heads, bank, data, device, args, detailed=False):
    front, source = bundle['front'], bundle['heads'][args.source_head]
    front.eval()
    source.eval()
    for head in heads.values():
        head.eval()
    cases = [('nomemory', 'nomemory', 'normal'), ('memory', 'memory', 'normal')]
    if detailed:
        cases += [('memory_' + action, 'memory', action) for action in ('random', 'shuffle_labels', 'off')]
    all_names = ['old_pixel', 'original_region', 'frozen_region'] + [case[0] for case in cases]
    matrices = {name: torch.zeros((4, 4), dtype=torch.long) for name in all_names}
    support = {name: {} for name, mode, _ in cases if mode == 'memory'}
    pairs = [('nomemory', 'memory'), ('frozen_region', 'nomemory'), ('frozen_region', 'memory'),
             ('original_region', 'frozen_region')]
    if detailed:
        pairs += [('memory', 'memory_' + action) for action in ('random', 'shuffle_labels', 'off')]
    diagnostic = DiagnosticAccumulator(threshold=bundle['manifest']['arguments']['threshold'], pairs=pairs) if detailed else None
    generators = {name: torch.Generator(device=device).manual_seed(args.seed + 1000 + index)
                  for index, (name, _, _) in enumerate(cases)}
    image_count, original_replay_max_abs = 0, 0.
    for batch in data:
        pre, post, _, gt, names = common.parse_batch(batch, device)
        reference = common.capture(front, pre, post)
        pool_route, read_route = common.routes(reference, bundle['manifest'], args.pool_mode, args.read_mode)
        encoded = encode_regions(source, reference['features']['roi'], pool_route)
        tokens, mass = encoded['tokens'], encoded['mass']
        base_prob = decode_regions(source.classifier(tokens), read_route, mass, gt.shape[-2:])
        original_route, _ = common.routes(reference, bundle['manifest'], args.source_head, args.source_head)
        original_prob = source(reference['features']['roi'], original_route, gt.shape[-2:])
        if args.pool_mode == args.read_mode == args.source_head:
            error = float((base_prob - original_prob).abs().max())
            original_replay_max_abs = max(original_replay_max_abs, error)
            if not torch.allclose(base_prob, original_prob, atol=2e-5, rtol=2e-4):
                raise RuntimeError('Separated frozen readout does not reproduce the original head')
        predictions = dict(old_pixel=reference['pixel'].argmax(1), original_region=original_prob.argmax(1),
                           frozen_region=base_prob.argmax(1))
        event_names = [common.event(name) for name in names]
        for name, mode, intervention in cases:
            logits, stats = classify_regions(heads[mode], tokens, mass, bank, names, event_names, mode,
                args.topk, args.query_chunk, not args.allow_same_event, intervention, generators[name])
            prob = decode_regions(logits, read_route, mass, gt.shape[-2:])
            if not torch.isfinite(prob).all():
                raise RuntimeError('Nonfinite validation probabilities: ' + name)
            predictions[name] = prob.argmax(1)
            if mode == 'memory':
                add_statistics(support[name], stats)
        for name, prediction in predictions.items():
            common.cm_update(matrices[name], gt, prediction)
        if diagnostic is not None:
            diagnostic.update(gt, predictions, reference['building'], names)
        image_count += len(names)
        if image_count % 25 == 0:
            print('evaluate memory', image_count, flush=True)
    expected = 2 if args.smoke else 450
    if image_count != expected:
        raise RuntimeError(f'Validation count changed: expected {expected}, received {image_count}')
    result = dict(metrics={name: common.metrics(matrix) for name, matrix in matrices.items()},
                  support={name: finish_statistics(row) for name, row in support.items()},
                  image_count=image_count, original_replay_max_abs=original_replay_max_abs,
                  separated_replay_checked=args.pool_mode == args.read_mode == args.source_head,
                  test_accessed=False)
    if diagnostic is not None:
        result['diagnostics'] = diagnostic.finalize()
    return result


def verify_initial_reproduction(bundle, result, args):
    """Always replay the original selected head, even under changed routing."""
    if args.smoke:
        return dict(checked=False, reason='Val2 smoke cannot verify the full-Val450 source score')
    expected = bundle['summary']['final_metrics']
    checks = {'old_pixel': 'old_pixel', 'original_region': args.source_head}
    errors = {}
    for current, original in checks.items():
        actual, wanted = result['metrics'][current]['valid_miou'], expected[original]['valid_miou']
        if not math.isfinite(actual) or abs(actual - wanted) > 1e-5:
            raise RuntimeError('Full-Val source reproduction failed: ' + current)
        errors[current] = abs(actual - wanted)
    return dict(checked=True, absolute_miou_errors=errors)


def main():
    parser = argparse.ArgumentParser(__doc__)
    common.add_arguments(parser, training=True)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--query_chunk', type=int, default=256)
    parser.add_argument('--allow_same_event', action='store_true',
                        help='Explicit ablation only; same image is always forbidden')
    args = parser.parse_args()
    common.validate_training_args(args)
    if args.smoke and args.batch > 2:
        parser.error('Smoke uses exactly two Train images; batch must be at most two')
    if min(args.topk, args.query_chunk) < 1:
        parser.error('Positive topk and query_chunk required')
    begin = time.monotonic()
    common.seed_everything(args.seed)
    bundle = common.load_source(args)
    identity = common.source_identity(bundle, args)
    bank_cpu, bank_manifest = read_bank(args.bank, identity, args.smoke, bundle['train'])
    if not args.smoke and set(bank_cpu['image_names']) != set(bundle['train']):
        raise ValueError('Formal bank must be extracted from the entire Train split')
    front, source = bundle['front'], bundle['heads'][args.source_head]
    device = next(front.parameters()).device
    bank = device_bank(bank_cpu, device)
    heads = make_heads(bank['tokens'].shape[1], device)
    initial = {name: common.state_digest(head) for name, head in heads.items()}
    if len(set(initial.values())) != 1 or set(map(id, heads['memory'].parameters())) & set(map(id, heads['nomemory'].parameters())):
        raise RuntimeError('Controls must have identical initial values and independent parameters')
    optimizers = {name: torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=.005)
                  for name, head in heads.items()}
    out = common.new_output(args.output)
    bank_hash = common.digest(str(args.bank) + '/bank.pt')
    manifest = dict(format=HEAD_FORMAT, arguments=vars(args), identity=identity,
                    source_hash=common.experiment_hash(), smoke=args.smoke, test_accessed=False,
                    bank_sha256=bank_hash, bank_manifest_sha256=common.digest(str(args.bank) + '/manifest.json'),
                    bank_regions=len(bank['tokens']), bank_images=len(bank['image_names']),
                    bank_limit=bank_manifest['bank_limit'], initial_head_sha256=initial,
                    parameters={name: sum(p.numel() for p in head.parameters()) for name, head in heads.items()},
                    frozen_front=True, frozen_building=True, frozen_base_q=True, frozen_embedding=True,
                    fresh_independent_heads=True, metadata_injected=False,
                    control='identical trainable attention/label/norm/linear parameters; no-memory learned slots; shared fixed token input',
                    exclusion='same image forbidden; ' + ('same event allowed (explicit ablation)' if args.allow_same_event else 'same event forbidden'),
                    loss='same original image-level NLL+Lovasz after dynamic routing unpool; no new auxiliary loss',
                    training='paired identical images/order; effective batch16; warmup/cosine AdamW; clip1',
                    extraction='supports full-image1024; queries use the same frozen embedding on Train crops/Val1024',
                    primary_endpoint='same-step final memory minus independently trained nomemory; best secondary',
                    checkpoint_policy='best+last classifier head only; no optimizer/front/embedding/cache copy',
                    intervention_policy='final Val random/shuffle_labels/off use the fixed trained Memory head; not substitute controls')
    common.save_json(out / 'manifest.json', manifest)
    common.save_json(out / 'summary.json', dict(complete=False, smoke=args.smoke, test_accessed=False))
    train_names = bank['image_names'] if args.smoke else bundle['train']
    training = common.loader(args.dataset, train_names, True, args.batch, args.workers, seed=args.seed + 2)
    validation = common.loader(args.dataset, bundle['val'][:2] if args.smoke else bundle['val'],
                               workers=args.workers, seed=args.seed + 3)
    if len(training) == 0:
        raise ValueError('Training loader is empty; smoke batch must fit two Train images')
    initial_result = evaluate(bundle, heads, bank, validation, device, args)
    initial_result['source_reproduction'] = verify_initial_reproduction(bundle, initial_result, args)
    common.assert_source(bundle)
    common.save_json(out / 'initial.json', initial_result)
    best = {name: -1. for name in heads}
    best_steps = {name: 0 for name in heads}
    iterator = iter(training)
    for step in range(1, args.steps + 1):
        front.eval()
        source.eval()
        warm = min(100, max(1, args.steps // 10))
        factor = step / warm if step <= warm else .5 * (1 + math.cos(math.pi * (step - warm) / max(1, args.steps - warm)))
        losses = {name: 0. for name in heads}
        support = {}
        for name, head in heads.items():
            head.train()
            optimizers[name].zero_grad(set_to_none=True)
            for group in optimizers[name].param_groups:
                group['lr'] = args.lr * factor
        for _ in range(args.accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(training)
                batch = next(iterator)
            pre, post, _, gt, names = common.parse_batch(batch, device)
            with torch.no_grad():
                reference = common.capture(front, pre, post)
                pool_route, read_route = common.routes(reference, bundle['manifest'], args.pool_mode, args.read_mode)
                encoded = encode_regions(source, reference['features']['roi'], pool_route)
            for name, head in heads.items():
                logits, stats = classify_regions(head, encoded['tokens'], encoded['mass'], bank,
                    names, [common.event(value) for value in names], name, args.topk, args.query_chunk,
                    not args.allow_same_event)
                probability = decode_regions(logits, read_route, encoded['mass'], gt.shape[-2:])
                loss = common.objective(probability, gt) / args.accum
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite matched-head loss: ' + name)
                loss.backward()
                losses[name] += float(loss.detach())
                if name == 'memory':
                    add_statistics(support, stats)
        for name, head in heads.items():
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizers[name].step()
        if any(p.grad is not None or p.requires_grad for model in (front, source) for p in model.parameters()):
            raise RuntimeError('Frozen front/readout embedding received gradients or became trainable')
        if step == 1 or step % 20 == 0:
            row = dict(step=step, crops=step * 16, losses=losses, lr=args.lr * factor,
                       support=finish_statistics(support), elapsed_seconds=time.monotonic() - begin)
            common.append_json(out / 'training.jsonl', row)
            print(row, flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(bundle, heads, bank, validation, device, args, detailed=step == args.steps)
            common.assert_source(bundle)
            if 'diagnostics' in result:
                common.save_json(out / 'audit_final.json', result.pop('diagnostics'))
                result['diagnostics_file'] = 'audit_final.json'
            common.append_json(out / 'validation.jsonl', dict(step=step, **result))
            for name, head in heads.items():
                score = result['metrics'][name]['valid_miou']
                state = dict(format=HEAD_FORMAT, head=head.state_dict(), name=name, step=step,
                             metrics=result['metrics'][name], manifest=manifest)
                if score > best[name]:
                    best[name], best_steps[name] = score, step
                    common.save_pt(out / (name + '_best.pth'), state)
                if step == args.steps:
                    common.save_pt(out / (name + '_last.pth'), state)
            print('VAL memory', step, {name: value['valid_miou'] for name, value in result['metrics'].items()}, flush=True)
    common.assert_source(bundle)
    if common.digest(str(args.bank) + '/bank.pt') != bank_hash or common.digest(str(args.bank) + '/manifest.json') != manifest['bank_manifest_sha256']:
        raise RuntimeError('Bank changed during paired training')
    if common.experiment_hash() != manifest['source_hash']:
        raise RuntimeError('Followup implementation changed during paired training')
    common.save_json(out / 'summary.json', dict(complete=True, final_step=args.steps,
        final_metrics=result['metrics'], support=result['support'], best=best, best_steps=best_steps,
        final_memory_minus_nomemory=result['metrics']['memory']['valid_miou'] - result['metrics']['nomemory']['valid_miou'],
        smoke=args.smoke, test_accessed=False, frozen_source_verified=True,
        elapsed_seconds=time.monotonic() - begin, diagnostics_file='audit_final.json',
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None))
    print('PAIRED MEMORY COMPLETE. Independent no-memory control; frozen embedding; Train/Val only.', flush=True)


if __name__ == '__main__':
    main()
