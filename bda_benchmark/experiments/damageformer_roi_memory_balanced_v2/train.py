"""Matched residual heads isolate support occupancy and class balance."""
import argparse
import math
import time
from pathlib import Path

import torch

from . import common
from .memory import (BANK_FORMAT, HEAD_FORMAT, MODES, add_statistics,
                     assert_global_matches_v1, assert_no_query_target_interfaces,
                     classify_regions, device_bank, finish_statistics, make_heads,
                     read_banks)
from bda_benchmark.experiments.damageformer_region_memory_v2.common import (
    OutputLock, baseline_policy, initial_gate, require_hard)
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import (
    decode_regions, encode_regions)
from bda_benchmark.experiments.damageformer_roi_exchange_v1.diagnostics import DiagnosticAccumulator


def schedule_factor(step, steps):
    warm = min(100, max(1, steps // 10))
    return (step / warm if step <= warm else
            .5 * (1 + math.cos(math.pi * (step - warm) /
                               max(1, steps - warm))))


@torch.no_grad()
def evaluate(bundle, heads, banks, data, args, detailed=False, initial=False):
    front, source_head, device = (bundle['front'], bundle['heads']['hard'],
                                  bundle['device'])
    front.eval(); source_head.eval()
    for head in heads.values():
        head.eval()
    cases = [(name, name, 'normal') for name in MODES]
    if detailed:
        cases += [('balanced_hard_' + action, 'balanced_hard', action)
                  for action in ('random', 'shuffle_labels', 'off')]
    prediction_names = ['old_pixel', 'old_region', 'source_hard'] + [x[0] for x in cases]
    matrices = {name: torch.zeros((4, 4), dtype=torch.long)
                for name in prediction_names}
    statistics = {name: {} for name, _, _ in cases}
    changed_pixels = {name: 0 for name, _, _ in cases}
    pairs = [('nomemory', 'global_all'), ('global_all', 'global_hard'),
             ('global_hard', 'balanced_hard'), ('nomemory', 'balanced_hard'),
             ('source_hard', 'nomemory'), ('source_hard', 'global_all'),
             ('source_hard', 'global_hard'), ('source_hard', 'balanced_hard')]
    if detailed:
        pairs += [('balanced_hard', 'balanced_hard_' + action)
                  for action in ('random', 'shuffle_labels', 'off')]
    diagnostic = DiagnosticAccumulator(
        threshold=bundle['manifest']['arguments']['threshold'], pairs=pairs) if detailed else None
    generators = {name: torch.Generator(device=device).manual_seed(args.seed + 500 + i)
                  for i, (name, _, _) in enumerate(cases)}
    image_count = valid_pixels = 0
    replay_max = 0.
    for batch in data:
        pre, post, _, gt, image_names = common.parse_batch(batch, device)
        reference = common.capture(front, pre, post)
        pool_route, read_route = common.routes(
            reference, bundle['manifest'], 'hard', 'hard')
        encoded = encode_regions(source_head, reference['features']['roi'], pool_route)
        tokens, mass = encoded['tokens'], encoded['mass']
        base_logits = source_head.classifier(tokens)
        base_probability = decode_regions(base_logits, read_route, mass, gt.shape[-2:])
        legacy = source_head(reference['features']['roi'], pool_route, gt.shape[-2:])
        replay_max = max(replay_max, float((base_probability - legacy).abs().max()))
        torch.testing.assert_close(base_probability, legacy, atol=2e-5, rtol=2e-4,
                                   msg='Separated frozen hard readout replay')
        predictions = dict(old_pixel=reference['pixel'].argmax(1),
                           old_region=reference['prob'].argmax(1),
                           source_hard=base_probability.argmax(1))
        valid = (gt >= 0) & (gt < 4)
        event_names = [common.event(name) for name in image_names]
        for case_name, mode, intervention in cases:
            logits, stats = classify_regions(
                heads[mode], tokens, mass, base_logits, banks, image_names,
                event_names, mode, args.topk, args.query_chunk,
                not args.allow_same_event, intervention, generators[case_name])
            if initial and not torch.equal(logits, base_logits):
                raise RuntimeError('Step0 residual identity failed: ' + case_name)
            probability = decode_regions(logits, read_route, mass, gt.shape[-2:])
            if not torch.isfinite(probability).all():
                raise RuntimeError('Nonfinite validation probability: ' + case_name)
            predictions[case_name] = probability.argmax(1)
            changed_pixels[case_name] += int(
                ((predictions[case_name] != predictions['source_hard']) & valid).sum())
            add_statistics(statistics[case_name], stats)
        for name, prediction in predictions.items():
            common.cm_update(matrices[name], gt, prediction)
        if diagnostic is not None:
            diagnostic.update(gt, predictions, reference['building'], image_names)
        image_count += len(image_names)
        valid_pixels += int(valid.sum())
        if image_count % 25 == 0:
            print('evaluate quality memory', image_count, flush=True)
    expected = 2 if args.smoke else 450
    if image_count != expected:
        raise RuntimeError(f'Expected {expected} Val images, received {image_count}')
    result = dict(
        metrics={name: common.metrics(cm) for name, cm in matrices.items()},
        support={name: finish_statistics(row) for name, row in statistics.items()},
        changed_pixel_fraction_vs_source={
            name: count / max(1, valid_pixels) for name, count in changed_pixels.items()},
        image_count=image_count, hard_replay_max_abs=replay_max,
        zero_residual_identity_checked=initial, test_accessed=False)
    if diagnostic is not None:
        result['diagnostics'] = diagnostic.finalize()
    return result


def independent_identical(heads):
    hashes = {name: common.state_digest(head) for name, head in heads.items()}
    counts = {name: sum(p.numel() for p in head.parameters())
              for name, head in heads.items()}
    pointers = {name: {p.data_ptr() for p in head.parameters()}
                for name, head in heads.items()}
    if len(set(hashes.values())) != 1 or len(set(counts.values())) != 1:
        raise RuntimeError('All controls need identical initial weights and parameter counts')
    for i, first in enumerate(MODES):
        for second in MODES[i + 1:]:
            if pointers[first] & pointers[second]:
                raise RuntimeError('Control heads share trainable storage')
    return hashes, counts


def run(args):
    require_hard(args)
    common.validate_training_args(args)
    if args.topk < 4 or args.topk % 4 or args.query_chunk < 1:
        raise ValueError('topk must be divisible by four; query_chunk must be positive')
    if args.smoke and args.batch > 2:
        raise ValueError('Smoke batch must be <=2')
    assert_no_query_target_interfaces()
    begin = time.monotonic()
    common.seed_everything(args.seed)
    code_hash = common.experiment_hash()
    bundle = common.load_source(args)
    policy = baseline_policy(bundle, args)
    identity = common.source_identity(bundle, args)
    banks_cpu, bank_manifest = read_banks(
        args.bank, identity, args.smoke, common.digest, bundle['train'])
    if not args.smoke:
        for bank in banks_cpu.values():
            if set(bank['image_names']) != set(bundle['train']):
                raise ValueError('Formal dual banks must cover complete Train2918')
    device = bundle['device']
    front, source_head = bundle['front'], bundle['heads']['hard']
    banks = {name: device_bank(bank, device) for name, bank in banks_cpu.items()}
    for bank in banks.values():
        assert_global_matches_v1(bank['tokens'][:1], bank, -1, -1, args.topk, True)
    heads = make_heads(banks['all']['tokens'].shape[1], device)
    initial_hash, parameter_counts = independent_identical(heads)
    optimizers = {name: torch.optim.AdamW(head.parameters(), lr=args.lr,
                                          weight_decay=.005)
                  for name, head in heads.items()}
    bank_files = {str(Path(args.bank) / filename): common.digest(Path(args.bank) / filename)
                  for filename in ('bank_all.pt', 'bank_hard.pt', 'manifest.json')}
    out = common.new_output(args.output)
    manifest = dict(
        format=HEAD_FORMAT, complete=False, arguments=vars(args), identity=identity,
        code_hash=code_hash, bank_format=bank_manifest['format'],
        bank_file_sha256=bank_files,
        all_bank_regions=len(banks['all']['tokens']),
        hard_bank_regions=len(banks['hard']['tokens']),
        bank_limit_per_policy=bank_manifest['bank_limit_per_policy'],
        head_initial_sha256=initial_hash, head_parameters=parameter_counts,
        smoke=args.smoke, test_accessed=False,
        frozen_front=True, frozen_building=True, frozen_q=True,
        frozen_embedding=True, frozen_source_classifier=True,
        output='frozen source_hard REGION logits + zero-initialized learned REGION correction',
        controls='nomemory; global_all(V1 eligible); global_hard(hard_count>0); balanced_hard(per-class quota)',
        isolation='global_hard-global_all tests occupancy quality; balanced_hard-global_hard tests class allocation',
        hard_eligibility='threshold-free argmax-Q hard_count>0, selected before each separate per-bucket cap',
        balanced_policy=f'equal {args.topk // 4} supports per stored dominant class; missing classes masked, no backfill',
        exclusion='same image forbidden; ' +
                  ('same event allowed (explicit ablation)' if args.allow_same_event else 'same event forbidden'),
        loss='same pixel NLL+Lovasz after unchanged hard regional unpool',
        protocol='same batches/order; effective batch16; AdamW wd.005; warmup/cosine; clip1; FP32',
        step0='all four correction heads exactly reproduce source_hard logits',
        interventions='final-only balanced_hard random/shuffle_labels/off',
        checkpoint_policy='best+last correction heads only; no optimizer/front/embedding/bank copy')
    common.save_json(out / 'manifest.json', manifest)
    common.save_json(out / 'baseline_policy.json', policy)
    common.save_json(out / 'summary.json', dict(
        complete=False, phase='initial_evaluation', optimizer_steps=0,
        smoke=args.smoke, test_accessed=False))
    train_names = banks['all']['image_names'] if args.smoke else bundle['train']
    training = common.loader(args.dataset, train_names, True, args.batch,
                             args.workers, seed=args.seed + 2)
    validation = common.loader(args.dataset,
        bundle['val'][:2] if args.smoke else bundle['val'],
        workers=args.workers, seed=args.seed + 3)
    if not len(training):
        raise ValueError('Empty training loader')
    initial_result = evaluate(bundle, heads, banks, validation, args, initial=True)
    gate = initial_gate(bundle, initial_result, args, out, policy)
    common.assert_v2_immutable(bundle, code_hash)
    print('STEP0', {name: row['valid_miou']
                    for name, row in initial_result['metrics'].items()}, flush=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    best = {name: -1. for name in MODES}
    best_steps = {name: 0 for name in MODES}
    grad_steps = {name: 0 for name in MODES}
    module_grad_steps = {name: {module: 0 for module in
        ('labels', 'attention', 'norm', 'ffn')} for name in MODES}
    iterator = iter(training)
    for step in range(1, args.steps + 1):
        front.eval(); source_head.eval()
        factor = schedule_factor(step, args.steps)
        losses = {name: 0. for name in MODES}
        statistics = {name: {} for name in MODES}
        for name, head in heads.items():
            head.train(); optimizers[name].zero_grad(set_to_none=True)
            for group in optimizers[name].param_groups:
                group['lr'] = args.lr * factor
        for _ in range(args.accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(training); batch = next(iterator)
            pre, post, _, gt, image_names = common.parse_batch(batch, device)
            with torch.no_grad():
                reference = common.capture(front, pre, post)
                pool_route, read_route = common.routes(
                    reference, bundle['manifest'], 'hard', 'hard')
                encoded = encode_regions(source_head, reference['features']['roi'], pool_route)
                base_logits = source_head.classifier(encoded['tokens'])
            event_names = [common.event(name) for name in image_names]
            for name, head in heads.items():
                logits, stats = classify_regions(
                    head, encoded['tokens'], encoded['mass'], base_logits,
                    banks, image_names, event_names, name, args.topk,
                    args.query_chunk, not args.allow_same_event)
                probability = decode_regions(
                    logits, read_route, encoded['mass'], gt.shape[-2:])
                loss = common.objective(probability, gt) / args.accum
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss: ' + name)
                loss.backward(); losses[name] += float(loss.detach())
                add_statistics(statistics[name], stats)
        grad_norm = {}
        for name, head in heads.items():
            for module_name in module_grad_steps[name]:
                has_gradient = any(p.grad is not None and bool((p.grad != 0).any())
                                   for p in getattr(head, module_name).parameters())
                module_grad_steps[name][module_name] += int(has_gradient)
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.,
                                                  error_if_nonfinite=True)
            grad_norm[name] = float(norm)
            grad_steps[name] += int(float(norm) > 0)
            optimizers[name].step()
        if any(p.requires_grad or p.grad is not None
               for model in (front, source_head) for p in model.parameters()):
            raise RuntimeError('Frozen source gradient boundary violated')
        if step == 1 or step % 20 == 0:
            row = dict(step=step, crops=step * 16, losses=losses,
                       lr=args.lr * factor, grad_norm=grad_norm,
                       support={name: finish_statistics(stats)
                                for name, stats in statistics.items()},
                       elapsed_seconds=time.monotonic() - begin)
            common.append_json(out / 'training.jsonl', row)
            print(row, flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(bundle, heads, banks, validation, args,
                              detailed=step == args.steps)
            common.assert_v2_immutable(bundle, code_hash)
            if 'diagnostics' in result:
                common.save_json(out / 'audit_final.json', result.pop('diagnostics'))
                result['diagnostics_file'] = 'audit_final.json'
            common.append_json(out / 'validation.jsonl', dict(step=step, **result))
            for name, head in heads.items():
                score = result['metrics'][name]['valid_miou']
                checkpoint = dict(
                    format=HEAD_FORMAT, name=name, step=step,
                    head=head.state_dict(), metrics=result['metrics'][name],
                    source_identity=identity, code_hash=code_hash,
                    bank_file_sha256=bank_files)
                if score > best[name]:
                    best[name], best_steps[name] = score, step
                    common.save_pt(out / (name + '_best.pth'), checkpoint)
                if step == args.steps:
                    common.save_pt(out / (name + '_last.pth'), checkpoint)
            common.save_json(out / 'summary.json', dict(
                complete=False, phase='training', optimizer_steps=step,
                best=best, best_steps=best_steps, smoke=args.smoke,
                test_accessed=False))
            print('VAL quality memory', step,
                  {name: row['valid_miou'] for name, row in result['metrics'].items()},
                  flush=True)
    common.assert_v2_immutable(bundle, code_hash)
    for path, expected in bank_files.items():
        if common.digest(path) != expected:
            raise RuntimeError('Dual bank changed during training: ' + path)
    changed = {name: common.state_digest(head) != initial_hash[name]
               for name, head in heads.items()}
    if not all(changed.values()) or not all(grad_steps.values()):
        raise RuntimeError('A residual head never changed or received gradients')
    final = result['metrics']
    summary = dict(
        complete=True, phase='complete', final_step=args.steps,
        optimizer_steps=args.steps, smoke=args.smoke, test_accessed=False,
        best=best, best_steps=best_steps, final_metrics=final,
        support=result['support'],
        changed_pixel_fraction_vs_source=result['changed_pixel_fraction_vs_source'],
        final_global_hard_minus_global_all=(final['global_hard']['valid_miou'] -
                                            final['global_all']['valid_miou']),
        final_balanced_hard_minus_global_hard=(final['balanced_hard']['valid_miou'] -
                                               final['global_hard']['valid_miou']),
        final_global_all_minus_nomemory=(final['global_all']['valid_miou'] -
                                         final['nomemory']['valid_miou']),
        final_balanced_hard_minus_nomemory=(final['balanced_hard']['valid_miou'] -
                                            final['nomemory']['valid_miou']),
        baseline_reproduction_pass=gate['passed'], frozen_source_verified=True,
        bank_immutable_verified=True, v2_code_immutable_verified=True,
        training_verification=dict(nonzero_gradient_steps=grad_steps,
            module_nonzero_gradient_steps=module_grad_steps,
            weights_changed=changed),
        elapsed_seconds=time.monotonic() - begin,
        peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                                  if device.type == 'cuda' else None))
    common.save_json(out / 'summary.json', summary)
    manifest['complete'] = True
    common.save_json(out / 'manifest.json', manifest)
    print('QUALITY-AWARE RESIDUAL MEMORY COMPLETE. Train/Val only; source unchanged.',
          flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    common.add_arguments(parser, training=True)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--query_chunk', type=int, default=256)
    parser.add_argument('--allow_same_event', action='store_true',
                        help='Explicit ablation; same image always forbidden')
    args = parser.parse_args()
    with OutputLock(args.output):
        run(args)


if __name__ == '__main__':
    main()
