"""Train isolated selective-Memory controls on frozen hard-region tokens.

Stage ``retrieval`` tests whether quality filtering plus a per-class ceiling is
better than no Memory, unrestricted hard-bank retrieval, or a forced balanced
quota.  Stage ``posterior`` tests whether an explicit support-label posterior
is useful, and whether its reliability gate needs retrieval-quality evidence.
No stage updates the front end, Q, source readout, or Train Memory bank.
"""
import argparse
import math
import time
from pathlib import Path

import torch

from . import common
from .memory import (
    BANK_FORMAT,
    HEAD_FORMAT,
    POSTERIOR_MODES,
    RETRIEVAL_MODES,
    add_statistics,
    assert_no_query_target_interfaces,
    build_quality_view,
    classify_posterior_regions,
    classify_residual_regions,
    device_bank,
    finish_statistics,
    make_posterior_heads,
    make_retrieval_heads,
    read_banks,
)
from bda_benchmark.experiments.damageformer_region_memory_v2.common import (
    OutputLock,
    baseline_policy,
    require_hard,
)
from bda_benchmark.experiments.damageformer_roi_exchange_v1.diagnostics import (
    DiagnosticAccumulator,
)
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import (
    decode_regions,
    encode_regions,
)


def schedule_factor(step, steps):
    warmup = min(100, max(1, steps // 10))
    if step <= warmup:
        return step / warmup
    return .5 * (1 + math.cos(math.pi * (step - warmup) /
                              max(1, steps - warmup)))


def stage_modes(stage):
    if stage == 'retrieval':
        return tuple(RETRIEVAL_MODES)
    if stage == 'posterior':
        return tuple(POSTERIOR_MODES)
    raise ValueError(stage)


def _classify(head, mode, stage, tokens, mass, base_logits, bank,
              image_names, event_names, args, intervention='normal',
              generator=None):
    shared = dict(
        head=head, tokens=tokens, mass=mass, base_logits=base_logits,
        bank=bank, names=image_names, event_names=event_names,
        topk=args.topk, preselect=args.preselect,
        per_class_cap=args.per_class_cap, min_cosine=args.min_cosine,
        query_chunk=args.query_chunk,
        exclude_same_event=not args.allow_same_event,
        intervention=intervention, generator=generator)
    if stage == 'retrieval':
        return classify_residual_regions(mode=mode, **shared)
    return classify_posterior_regions(
        mode=mode, retrieval_policy=args.posterior_retrieval,
        temperature=args.temperature, **shared)


def _cases(stage, detailed):
    cases = [(name, name, 'normal') for name in stage_modes(stage)]
    primary = ('adaptive_cap' if stage == 'retrieval'
               else 'posterior_quality_gate')
    if detailed:
        cases += [(primary + '_' + action, primary, action)
                  for action in ('random', 'shuffle_labels', 'off')]
    return cases


@torch.no_grad()
def evaluate(bundle, heads, bank, data, args, detailed=False, initial=False):
    front = bundle['front']
    source_head = bundle['heads']['hard']
    device = bundle['device']
    front.eval()
    source_head.eval()
    for head in heads.values():
        head.eval()
    cases = _cases(args.stage, detailed)
    prediction_names = ['old_pixel', 'old_region', 'source_hard'] + [
        name for name, _, _ in cases]
    matrices = {name: torch.zeros((4, 4), dtype=torch.long)
                for name in prediction_names}
    statistics = {name: {} for name, _, _ in cases}
    changed_pixels = {name: 0 for name, _, _ in cases}
    modes = stage_modes(args.stage)
    primary = ('adaptive_cap' if args.stage == 'retrieval'
               else 'posterior_quality_gate')
    pairs = [('source_hard', name) for name in modes]
    pairs += [(modes[0], name) for name in modes[1:]]
    if detailed:
        pairs += [(primary, primary + '_' + action)
                  for action in ('random', 'shuffle_labels', 'off')]
    diagnostic = DiagnosticAccumulator(
        threshold=bundle['manifest']['arguments']['threshold'],
        pairs=pairs) if detailed else None
    generators = {
        name: torch.Generator(device=device).manual_seed(args.seed + 500 + i)
        for i, (name, _, _) in enumerate(cases)
    }
    image_count = valid_pixels = 0
    replay_max = 0.
    for batch in data:
        pre, post, _, gt, image_names = common.parse_batch(batch, device)
        reference = common.capture(front, pre, post)
        pool_route, read_route = common.routes(
            reference, bundle['manifest'], 'hard', 'hard')
        encoded = encode_regions(
            source_head, reference['features']['roi'], pool_route)
        tokens, mass = encoded['tokens'], encoded['mass']
        base_logits = source_head.classifier(tokens)
        base_probability = decode_regions(
            base_logits, read_route, mass, gt.shape[-2:])
        legacy = source_head(
            reference['features']['roi'], pool_route, gt.shape[-2:])
        replay_max = max(replay_max,
                         float((base_probability - legacy).abs().max()))
        torch.testing.assert_close(
            base_probability, legacy, atol=2e-5, rtol=2e-4,
            msg='Separated frozen hard readout replay')
        predictions = dict(
            old_pixel=reference['pixel'].argmax(1),
            old_region=reference['prob'].argmax(1),
            source_hard=base_probability.argmax(1))
        valid = (gt >= 0) & (gt < 4)
        event_names = [common.event(name) for name in image_names]
        for case_name, mode, intervention in cases:
            logits, stats = _classify(
                heads[mode], mode, args.stage, tokens, mass, base_logits,
                bank, image_names, event_names, args, intervention,
                generators[case_name])
            if initial and not torch.equal(logits, base_logits):
                raise RuntimeError(
                    'Step0 must exactly reproduce source logits: ' + case_name)
            probability = decode_regions(
                logits, read_route, mass, gt.shape[-2:])
            if not torch.isfinite(probability).all():
                raise RuntimeError('Nonfinite validation output: ' + case_name)
            predictions[case_name] = probability.argmax(1)
            changed_pixels[case_name] += int(
                ((predictions[case_name] != predictions['source_hard']) &
                 valid).sum())
            add_statistics(statistics[case_name], stats)
        for name, prediction in predictions.items():
            common.cm_update(matrices[name], gt, prediction)
        if diagnostic is not None:
            diagnostic.update(
                gt, predictions, reference['building'], image_names)
        image_count += len(image_names)
        valid_pixels += int(valid.sum())
        if image_count % 25 == 0:
            print('evaluate selective memory', image_count, flush=True)
    expected = 2 if args.smoke else 450
    if image_count != expected:
        raise RuntimeError(
            f'Expected {expected} Val images, received {image_count}')
    result = dict(
        metrics={name: common.metrics(cm) for name, cm in matrices.items()},
        support={name: finish_statistics(row)
                 for name, row in statistics.items()},
        changed_pixel_fraction_vs_source={
            name: count / max(1, valid_pixels)
            for name, count in changed_pixels.items()},
        image_count=image_count,
        hard_replay_max_abs=replay_max,
        zero_residual_identity_checked=initial,
        test_accessed=False)
    if diagnostic is not None:
        result['diagnostics'] = diagnostic.finalize()
    return result


def independent_identical(heads):
    modes = tuple(heads)
    hashes = {name: common.state_digest(head)
              for name, head in heads.items()}
    counts = {name: sum(parameter.numel() for parameter in head.parameters())
              for name, head in heads.items()}
    pointers = {name: {parameter.data_ptr()
                       for parameter in head.parameters()}
                for name, head in heads.items()}
    if len(set(hashes.values())) != 1 or len(set(counts.values())) != 1:
        raise RuntimeError(
            'Controls need identical initialization and parameter counts')
    for index, first in enumerate(modes):
        for second in modes[index + 1:]:
            if pointers[first] & pointers[second]:
                raise RuntimeError('Control heads share trainable storage')
    return hashes, counts


def _thresholds_for_json(bank):
    thresholds = bank.get('quality_thresholds', [])
    if not isinstance(thresholds, list):
        raise ValueError('quality_thresholds must be the classwise Train list')
    return [dict(row) for row in thresholds]


def _validate_args(args):
    require_hard(args)
    common.validate_training_args(args)
    if args.topk < 1 or args.preselect < args.topk:
        raise ValueError('Require topk>=1 and preselect>=topk')
    if args.stage == 'retrieval' and (args.topk < 4 or args.topk % 4):
        raise ValueError('Retrieval stage fixed_balanced control requires topk divisible by four')
    if args.stage == 'retrieval' and (args.topk < 4 or args.topk % 4):
        raise ValueError(
            'Retrieval controls include fixed_balanced, so topk must be '
            'divisible by four')
    if args.per_class_cap < 1 or args.query_chunk < 1:
        raise ValueError('per_class_cap/query_chunk must be positive')
    if not 0 <= args.quality_quantile < .5:
        raise ValueError('quality_quantile must lie in [0,.5)')
    if args.min_cosine != -1.:
        raise ValueError(
            'Selective V1 fixes min_cosine=-1 so its random control is fair')
    if args.temperature <= 0 or not math.isfinite(args.temperature):
        raise ValueError('temperature must be finite and positive')
    if args.posterior_retrieval not in ('global_hard', 'adaptive_cap'):
        raise ValueError('Unsupported posterior retrieval policy')
    if args.smoke and args.batch > 2:
        raise ValueError('Smoke batch must be <=2')


def run(args):
    _validate_args(args)
    assert_no_query_target_interfaces()
    begin = time.monotonic()
    common.seed_everything(args.seed)
    code_hash = common.experiment_hash()
    parent_hash = common.parent_experiment_hash()
    bundle = common.load_source(args)
    policy = baseline_policy(bundle, args)
    identity = common.source_identity(bundle, args)
    banks_cpu, bank_manifest = read_banks(
        args.bank, identity, args.smoke, common.digest, bundle['train'])
    if not args.smoke:
        for loaded_bank in banks_cpu.values():
            if set(loaded_bank['image_names']) != set(bundle['train']):
                raise ValueError('Formal bank must cover complete Train2918')
    bank_cpu = build_quality_view(
        banks_cpu['hard'], quantile=args.quality_quantile)
    bank = device_bank(bank_cpu, bundle['device'])
    front, source_head = bundle['front'], bundle['heads']['hard']
    if args.stage == 'retrieval':
        heads = make_retrieval_heads(bank['tokens'].shape[1], bundle['device'])
    else:
        heads = make_posterior_heads(bank['tokens'].shape[1], bundle['device'])
    initial_hash, parameter_counts = independent_identical(heads)
    optimizers = {
        name: torch.optim.AdamW(head.parameters(), lr=args.lr,
                                weight_decay=.005)
        for name, head in heads.items()
    }
    bank_files = {
        str(Path(args.bank) / filename): common.digest(Path(args.bank) / filename)
        for filename in ('bank_all.pt', 'bank_hard.pt', 'manifest.json')
    }
    out = common.new_output(args.output)
    manifest = dict(
        format=HEAD_FORMAT, complete=False, stage=args.stage,
        arguments=vars(args), identity=identity, code_hash=code_hash,
        parent_code_hash=parent_hash, bank_format=bank_manifest['format'],
        expected_bank_format=BANK_FORMAT, bank_file_sha256=bank_files,
        hard_bank_regions=len(bank['tokens']),
        quality_eligible_regions=int(bank['quality_eligible'].sum().item()),
        quality_thresholds=_thresholds_for_json(bank),
        head_initial_sha256=initial_hash, head_parameters=parameter_counts,
        smoke=args.smoke, test_accessed=False,
        frozen_front=True, frozen_building=True, frozen_q=True,
        frozen_embedding=True, frozen_source_classifier=True,
        output='frozen source_hard logits plus an isolated learned correction',
        retrieval_controls=(
            'nomemory; global_hard; fixed_balanced; adaptive_cap'
            if args.stage == 'retrieval' else None),
        posterior_controls=(
            'query_only; posterior_nogate; posterior_query_gate; '
            'posterior_quality_gate' if args.stage == 'posterior' else None),
        adaptive_policy=(
            'Train-only classwise quality thresholds; global cosine order; '
            f'at most {args.per_class_cap}/class and {args.topk}/query; '
            'no minimum quota and no low-quality backfill'),
        posterior=(
            'temperature-weighted support-label posterior; log-posterior '
            'correction clamped to [-5,5], class-centered, and scaled by a '
            'projected [0,1] strength; unsupported queries '
            'remain on the source/query-only path'),
        exclusion='same image forbidden; ' + (
            'same event allowed (explicit ablation)'
            if args.allow_same_event else 'same event forbidden'),
        loss='same pixel NLL+Lovasz after unchanged hard regional unpool',
        protocol=(
            'same batches/order; effective batch16; AdamW wd.005; '
            'warmup/cosine; clip1; FP32'),
        step0='every fresh head exactly reproduces source_hard logits',
        interventions='final-only primary Memory random/shuffle_labels/off',
        checkpoint_policy=(
            'best+last head only; no optimizer/front/embedding/bank copy'))
    common.save_json(out / 'manifest.json', manifest)
    common.save_json(out / 'baseline_policy.json', policy)
    common.save_json(out / 'summary.json', dict(
        complete=False, phase='initial_evaluation', stage=args.stage,
        optimizer_steps=0, smoke=args.smoke, test_accessed=False))
    train_names = (banks_cpu['all']['image_names']
                   if args.smoke else bundle['train'])
    training = common.loader(
        args.dataset, train_names, True, args.batch, args.workers,
        seed=args.seed + 2)
    validation = common.loader(
        args.dataset, bundle['val'][:2] if args.smoke else bundle['val'],
        workers=args.workers, seed=args.seed + 3)
    if not len(training):
        raise ValueError('Empty training loader')
    initial_result = evaluate(
        bundle, heads, bank, validation, args, initial=True)
    gate = common.selective_initial_gate(
        bundle, initial_result, args, out, policy)
    common.assert_selective_immutable(bundle, code_hash, parent_hash)
    print('STEP0', {name: row['valid_miou']
                    for name, row in initial_result['metrics'].items()},
          flush=True)
    if bundle['device'].type == 'cuda':
        torch.cuda.reset_peak_memory_stats(bundle['device'])
    modes = stage_modes(args.stage)
    best = {name: -1. for name in modes}
    best_steps = {name: 0 for name in modes}
    grad_steps = {name: 0 for name in modes}
    iterator = iter(training)
    for step in range(1, args.steps + 1):
        front.eval()
        source_head.eval()
        factor = schedule_factor(step, args.steps)
        losses = {name: 0. for name in modes}
        statistics = {name: {} for name in modes}
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
            pre, post, _, gt, image_names = common.parse_batch(
                batch, bundle['device'])
            with torch.no_grad():
                reference = common.capture(front, pre, post)
                pool_route, read_route = common.routes(
                    reference, bundle['manifest'], 'hard', 'hard')
                encoded = encode_regions(
                    source_head, reference['features']['roi'], pool_route)
                base_logits = source_head.classifier(encoded['tokens'])
            event_names = [common.event(name) for name in image_names]
            for name, head in heads.items():
                logits, stats = _classify(
                    head, name, args.stage, encoded['tokens'],
                    encoded['mass'], base_logits, bank, image_names,
                    event_names, args)
                probability = decode_regions(
                    logits, read_route, encoded['mass'], gt.shape[-2:])
                loss = common.objective(probability, gt) / args.accum
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss: ' + name)
                loss.backward()
                losses[name] += float(loss.detach())
                add_statistics(statistics[name], stats)
        grad_norm = {}
        for name, head in heads.items():
            norm = torch.nn.utils.clip_grad_norm_(
                head.parameters(), 1., error_if_nonfinite=True)
            grad_norm[name] = float(norm)
            grad_steps[name] += int(float(norm) > 0)
            optimizers[name].step()
            if args.stage == 'posterior':
                head.clamp_strength_()
        if any(parameter.requires_grad or parameter.grad is not None
               for model in (front, source_head)
               for parameter in model.parameters()):
            raise RuntimeError('Frozen source gradient boundary violated')
        if step == 1 or step % 20 == 0:
            row = dict(
                step=step, crops=step * 16, losses=losses,
                lr=args.lr * factor, grad_norm=grad_norm,
                support={name: finish_statistics(stats)
                         for name, stats in statistics.items()},
                elapsed_seconds=time.monotonic() - begin)
            common.append_json(out / 'training.jsonl', row)
            print(row, flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(
                bundle, heads, bank, validation, args,
                detailed=step == args.steps)
            common.assert_selective_immutable(bundle, code_hash, parent_hash)
            if 'diagnostics' in result:
                common.save_json(
                    out / 'audit_final.json', result.pop('diagnostics'))
                result['diagnostics_file'] = 'audit_final.json'
            common.append_json(
                out / 'validation.jsonl', dict(step=step, **result))
            for name, head in heads.items():
                score = result['metrics'][name]['valid_miou']
                checkpoint = dict(
                    format=HEAD_FORMAT, stage=args.stage, name=name,
                    step=step, head=head.state_dict(),
                    metrics=result['metrics'][name],
                    source_identity=identity, code_hash=code_hash,
                    bank_file_sha256=bank_files)
                if score > best[name]:
                    best[name], best_steps[name] = score, step
                    common.save_pt(out / (name + '_best.pth'), checkpoint)
                if step == args.steps:
                    common.save_pt(out / (name + '_last.pth'), checkpoint)
            common.save_json(out / 'summary.json', dict(
                complete=False, phase='training', stage=args.stage,
                optimizer_steps=step, best=best, best_steps=best_steps,
                smoke=args.smoke, test_accessed=False))
            print('VAL selective memory', step,
                  {name: row['valid_miou']
                   for name, row in result['metrics'].items()}, flush=True)
    common.assert_selective_immutable(bundle, code_hash, parent_hash)
    for path, expected in bank_files.items():
        if common.digest(path) != expected:
            raise RuntimeError('Memory bank changed during training: ' + path)
    changed = {name: common.state_digest(head) != initial_hash[name]
               for name, head in heads.items()}
    if not all(changed.values()) or not all(grad_steps.values()):
        raise RuntimeError('A head never changed or received gradients')
    final = result['metrics']
    base_mode = modes[0]
    summary = dict(
        complete=True, phase='complete', stage=args.stage,
        final_step=args.steps, optimizer_steps=args.steps,
        smoke=args.smoke, test_accessed=False,
        best=best, best_steps=best_steps, final_metrics=final,
        support=result['support'],
        changed_pixel_fraction_vs_source=(
            result['changed_pixel_fraction_vs_source']),
        same_step_gains_vs_control={
            name: final[name]['valid_miou'] - final[base_mode]['valid_miou']
            for name in modes[1:]},
        # ``passed`` is this experiment's blocking gate, not the assertion
        # that every diagnostic metric cell was bitwise/reduction stable.
        baseline_reproduction_pass=gate['passed'],
        blocking_baseline_reproduction_pass=gate.get('blocking_passed'),
        baseline_gate_status=gate.get('status'),
        all_baseline_metric_cells_pass=gate.get(
            'all_metric_checks_passed'),
        nonblocking_baseline_drift=gate.get('nonblocking_failures', []),
        frozen_source_verified=True, bank_immutable_verified=True,
        selective_code_immutable_verified=True,
        training_verification=dict(
            nonzero_gradient_steps=grad_steps, weights_changed=changed),
        elapsed_seconds=time.monotonic() - begin,
        peak_cuda_allocated_bytes=(
            torch.cuda.max_memory_allocated(bundle['device'])
            if bundle['device'].type == 'cuda' else None))
    common.save_json(out / 'summary.json', summary)
    manifest['complete'] = True
    common.save_json(out / 'manifest.json', manifest)
    print('SELECTIVE MEMORY ' + args.stage.upper() +
          ' COMPLETE. Train/Val only; frozen source/bank unchanged.',
          flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    common.add_arguments(parser, training=True)
    parser.add_argument('--stage', choices=('retrieval', 'posterior'),
                        required=True)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--preselect', type=int, default=32)
    parser.add_argument('--per_class_cap', type=int, default=4)
    parser.add_argument('--quality_quantile', type=float, default=.25)
    parser.add_argument('--min_cosine', type=float, default=-1.)
    parser.add_argument('--temperature', type=float, default=.07)
    parser.add_argument('--query_chunk', type=int, default=256)
    parser.add_argument('--posterior_retrieval',
                        choices=('global_hard', 'adaptive_cap'),
                        default='adaptive_cap')
    parser.add_argument('--allow_same_event', action='store_true',
                        help='Explicit ablation; same image always forbidden')
    args = parser.parse_args()
    with OutputLock(args.output):
        run(args)


if __name__ == '__main__':
    main()
