"""Three isolated matched experiments. Original front immutable; Train/Val only."""
import argparse
import hashlib
import math
import time
from pathlib import Path

import torch

from bda_benchmark.experiments.damageformer_roi_followup_v1.common import (
    append_json, assert_source, cm_update, experiment_hash as parent_hash,
    load_source, loader, metrics, new_output, objective, parse_batch, save_json,
    save_pt, seed_everything, validate_training_args,
)
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import routing_stats
from bda_benchmark.experiments.damageformer_roi_exchange_v1.diagnostics import DiagnosticAccumulator
from bda_benchmark.experiments.damageformer_region_content_v1.core import verify_original_readout
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import forward_separate
from .generator import structure_reconstruction_loss
from .model import SUITES, PAIRS, capture_inputs, hard_route, make_branches, member_hashes, gradients
from .reproduction import load_history, build_policy, compare_baselines, enforce, json_safe_evidence

FORMAT = 'PARTITION_LEARNING_V1'


def implementation_hash():
    h = hashlib.sha256(parent_hash().encode())
    for path in sorted(Path(__file__).parent.iterdir()):
        if path.is_file() and path.suffix in ('.py', '.sh'):
            h.update(path.name.encode()); h.update(path.read_bytes())
    return h.hexdigest()


def arguments(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    for key in ('checkpoint', 'run_root', 'dataset', 'train_list', 'val_list', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--suite', choices=tuple(SUITES), required=True)
    parser.add_argument('--source_head', choices=('hard',), default='hard')
    for key, default in (('steps', 5000), ('eval_every', 500), ('batch', 2), ('accum', 8),
                         ('workers', 4), ('seed', 3407)):
        parser.add_argument('--'+key, type=int, default=default)
    for key, default in (('lr', 1e-4), ('structure_weight', .1), ('sar_weight', .2),
                         ('support_temperature', .2)):
        parser.add_argument('--'+key, type=float, default=default)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    try:
        validate_training_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.smoke and (args.steps != 5000 or args.eval_every != 500):
        parser.error('Formal experiments require 5000 optimizer steps and Val every 500')
    if not all(math.isfinite(x) and x > 0 for x in
               (args.structure_weight, args.sar_weight, args.support_temperature)) or args.sar_weight >= 1:
        parser.error('Positive finite weights/temperature required; 0 < sar_weight < 1')
    # Preserve load_source interface; no accidental stale pool/read environment.
    args.pool_mode = args.read_mode = 'hard'
    if Path(args.output).exists():
        parser.error('Output exists; preserve it and choose a new --output')
    return args


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(bundle, branches, data, args, *, initial=False, detailed=False):
    front, device = bundle['front'], bundle['device']
    for branch in branches.values():
        branch.eval()
    names = ('old_pixel', 'old_region', 'source_hard', *branches)
    cms = {name: torch.zeros(4, 4, dtype=torch.long) for name in names}
    diagnostic = DiagnosticAccumulator(
        threshold=bundle['manifest']['arguments']['threshold'],
        pairs=[*PAIRS[args.suite], *(('source_hard', n) for n in branches)]) if detailed else None
    routing, drift, durations = ({name: {} for name in branches},
                                {name: dict(mean_abs=0., max_abs=0.) for name in branches},
                                {name: 0. for name in branches})
    quality = {name: {} for name in branches}
    base_quality = {name: {} for name in branches}
    count, front_seconds, pixel_error, region_error, replay_error = 0, 0., 0., 0., {}
    for batch in data:
        pre, post, _, gt, image_names = parse_batch(batch, device)
        synchronize(device); started = time.perf_counter()
        reference, inputs = capture_inputs(front, pre, post)
        synchronize(device); front_seconds += time.perf_counter()-started
        pixel_error = max(pixel_error, reference['identity_max_abs'])
        region_error = max(region_error, verify_original_readout(front, reference, gt.shape[-2:]))
        route = hard_route(reference, reference['q'], bundle['manifest']['arguments']['threshold'])
        source_prob = forward_separate(bundle['heads']['hard'], reference['features']['roi'],
                                       route, route, gt.shape[-2:])['prob']
        if initial and count == 0:
            legacy = bundle['heads']['hard'](reference['features']['roi'], route, gt.shape[-2:])
            torch.testing.assert_close(source_prob, legacy, atol=2e-5, rtol=2e-4)
        predictions = dict(old_pixel=reference['pixel'].argmax(1),
                           old_region=reference['prob'].argmax(1), source_hard=source_prob.argmax(1))
        for name, branch in branches.items():
            synchronize(device); started = time.perf_counter()
            output = branch(reference, inputs, gt.shape[-2:])
            synchronize(device); durations[name] += time.perf_counter()-started
            predictions[name] = output['prob'].argmax(1)
            if initial and count == 0 and name not in ('rgb_sar', 'uniform', 'sar'):
                torch.testing.assert_close(output['q'], reference['q'], atol=2e-5, rtol=2e-4,
                                           msg=name+' original Q initialization')
                torch.testing.assert_close(output['prob'], source_prob, atol=2e-5, rtol=2e-4,
                                           msg=name+' source readout initialization')
                replay_error[name] = float((output['prob']-source_prob).abs().max())
            delta = (output['q']-reference['q']).abs()
            drift[name]['mean_abs'] += float(delta.mean())*len(pre)
            drift[name]['max_abs'] = max(drift[name]['max_abs'], float(delta.max()))
            row = routing_stats(output['pool_route'], reference['building'])
            row.update(output['exchange_diagnostics'])
            row['base_q_entropy'] = float(-(output['q']*output['q'].clamp_min(1e-8).log()).sum(1).mean())
            for key, value in row.items():
                routing[name][key] = routing[name].get(key, 0.)+float(value)*len(pre)
            if detailed:
                from .quality import partition_quality
                values = partition_quality(output['read_route'], gt)
                for key, value in values.items():
                    quality[name][key] = quality[name].get(key, 0.)+value
                base_route = dict(q=output['q'],
                    ids=reference['ids'][None].expand(len(pre), -1, -1),
                    k=reference['tokens'].shape[1], shape=reference['shape'])
                for key, value in partition_quality(base_route, gt).items():
                    base_quality[name][key] = base_quality[name].get(key, 0.)+value
            del output
        for name, pred in predictions.items():
            cm_update(cms[name], gt, pred)
        if diagnostic is not None:
            diagnostic.update(gt, predictions, reference['building'], image_names)
        count += len(pre)
        if count % 25 == 0:
            print('evaluate', count, flush=True)
    if count != (2 if args.smoke else 450):
        raise RuntimeError('Expected smoke Val2 or full Val450')
    if count == 0:
        raise RuntimeError('Empty validation')
    for row in drift.values():
        row['mean_abs'] /= count
    result = dict(image_count=count, metrics={n: metrics(cm) for n, cm in cms.items()},
        routing={n: {k: v/count for k, v in row.items()} for n, row in routing.items()},
        q_drift=drift, identity_max_abs=pixel_error, region_replay_max_abs=region_error,
        branch_forward_seconds_per_image={n: t/count for n, t in durations.items()},
        shared_front_seconds_per_image=front_seconds/count, test_accessed=False)
    if initial:
        result['initial_replay_max_abs'] = replay_error
        result['initial_replay_exempt'] = 'rgb_sar has different inputs; support variants have different gathering'
    if diagnostic is not None:
        result['diagnostics'] = diagnostic.finalize()
        result['partition_quality_totals'] = quality
        result['base_partition_quality_totals'] = base_quality
    return result


def verify_baselines(bundle, result, args, policy=None):
    if args.smoke:
        return dict(passed=None, status='smoke_not_full_val', checks=[], warnings=[])
    expected = bundle['summary']['final_metrics']
    report = compare_baselines(result['metrics'], expected, policy or build_policy(expected))
    enforce(report)
    return report


def reproduction_policy(bundle, args):
    """Calibrate BEFORE new inference, from the completed frozen-reference history only."""
    if args.smoke:
        return None
    rows, warnings, sha = load_history(args.run_root, bundle['summary']['final_step'])
    path = str(Path(args.run_root)/'validation.jsonl')
    if sha is not None:
        # Subsequent assert_source checks history too; never edit the source log.
        bundle['provenance']['source_file_sha256'][path] = sha
    policy = build_policy(bundle['summary']['final_metrics'], rows, warnings)
    policy['history_file'] = path
    policy['history_sha256'] = sha
    return policy


def initial_gate(bundle, result, args, out, policy):
    """Persist ALL baseline evidence even when the gate rejects this run."""
    save_json(out/'initial.json', json_safe_evidence(result))
    if args.smoke:
        report = dict(passed=None, status='smoke_not_full_val', checks=[], warnings=[])
    else:
        report = compare_baselines(result['metrics'], bundle['summary']['final_metrics'], policy)
        report['status'] = 'passed' if report['passed'] else 'failed'
    report['metric_checks_passed'] = report['passed']
    report['source_integrity_pass'] = None
    save_json(out/'reproduction.json', report)
    for warning in report.get('warnings', []):
        print('REPRO WARNING:', warning, flush=True)
    for row in report.get('checks', []):
        if not row['passed']:
            print('REPRO FAIL:', row, flush=True)
        elif row.get('tolerance', 0.) > 1e-5:
            print('REPRO HISTORICAL TOLERANCE:', row, flush=True)
    try:
        assert_source(bundle)
        report['source_integrity_pass'] = True
        save_json(out/'reproduction.json', report)
        if not args.smoke:
            enforce(report)
    except Exception as exc:
        if report['source_integrity_pass'] is None:
            report['source_integrity_pass'] = False
            report['passed'] = False
        report['status'] = 'failed'
        report['error'] = str(exc)
        save_json(out/'reproduction.json', report)
        save_json(out/'summary.json', dict(complete=False, suite=args.suite, smoke=args.smoke,
            phase='baseline_check_failed', optimizer_steps=0, test_accessed=False,
            reproduction_file='reproduction.json', error=str(exc)))
        print('Baseline check stopped BEFORE training. Details:', out/'reproduction.json', flush=True)
        raise
    print('BASELINE REPRODUCTION '+('SKIPPED (smoke only)' if args.smoke else 'PASS'), flush=True)
    return report


def checkpoint_payload(branch, name, step, metric, manifest):
    return dict(format=FORMAT, variant=name, step=step, branch=branch.state_dict(),
                metrics=metric, manifest=manifest)


def main(argv=None):
    wall_start = time.monotonic()
    args = arguments(argv)
    seed_everything(args.seed)
    bundle = load_source(args)
    if bundle['manifest'].get('smoke', False):
        raise ValueError('Use a completed formal source, even for this suite smoke')
    front, device = bundle['front'], bundle['device']
    settings = bundle['manifest']['arguments']
    if args.suite == 'support' and not 0 < settings['epsilon'] <= .5:
        raise ValueError('Support comparison needs positive source exchange epsilon')
    branches = make_branches(front, bundle['heads']['hard'], args.suite,
        beta=args.sar_weight, epsilon=settings['epsilon'], threshold=settings['threshold'],
        support_temperature=args.support_temperature)
    hashes = member_hashes(branches)
    policy = reproduction_policy(bundle, args)
    out = new_output(args.output)
    code_hash = implementation_hash()
    manifest = dict(format=FORMAT, suite=args.suite, variants=list(branches), arguments=vars(args),
        source_provenance=bundle['provenance'], implementation_hash=code_hash,
        reproduction_gate_version='history_bounded_v1', reproduction_policy=policy,
        runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32),
        initial_state_sha256=hashes, frozen_front=True, frozen_building=True,
        no_memory=True, no_new_backbone=True, test_accessed=False, smoke=args.smoke,
        paired_crop_batches=True, effective_batch=16,
        source_head_step=bundle['provenance']['source_head_step'],
        epsilon=settings['epsilon'], threshold=settings['threshold'], front_config=dict(front.config),
        parameter_count={n: dict(total=sum(p.numel() for p in b.parameters()),
            trainable=sum(p.numel() for p in b.parameters() if p.requires_grad),
            by_member={k: sum(p.numel() for p in m.parameters()) for k,m in b.named_children()})
            for n,b in branches.items()},
        content='frozen original RGB+SAR pixel-head input with original SoftROI',
        initialization='independent exact copies of same hard_last readout and front decoder',
        objective='existing region NLL+Lovasz; only learn_q_structure adds weighted binary reconstruction',
        structure_target='Train GT foreground/valid area fractions; reconstruction on unsplit Q, never forward input',
        support_control='uniform and SAR share hard-reference nonempty support, epsilon budget and fallback; not legacy exchange',
        schedule='AdamW .005 decay, 100-step warmup, cosine, clip1; smoke warmup2',
        checkpoint_policy='small branch best+last only; no front/cache/optimizer; not exact-resumable',
        primary_endpoint='same optimizer-step final pairs; best checkpoints secondary',
        resource_scope='peak memory is combined resident front and all suite branches; branch inference timed separately')
    save_json(out/'manifest.json', manifest)
    save_json(out/'summary.json', dict(complete=False, suite=args.suite, smoke=args.smoke,
                                     phase='initial_validation', optimizer_steps=0, test_accessed=False))
    print('SUITE', args.suite, 'variants', list(branches), 'source step', manifest['source_head_step'], flush=True)
    training = loader(args.dataset, bundle['train'], True, args.batch, args.workers, seed=args.seed+2)
    validation = loader(args.dataset, bundle['val'][:2] if args.smoke else bundle['val'],
                        workers=args.workers, seed=args.seed+3)
    initial = evaluate(bundle, branches, validation, args, initial=True)
    reproduction = initial_gate(bundle, initial, args, out, policy)
    # No optimizer exists until the immutable baselines have been checked.
    optimizers = {n: torch.optim.AdamW((p for p in b.parameters() if p.requires_grad),
                                     lr=args.lr, weight_decay=.005) for n, b in branches.items()}
    save_json(out/'summary.json', dict(complete=False, suite=args.suite, smoke=args.smoke,
        phase='training', progress_file='training.jsonl', baseline_reproduction_pass=reproduction['passed'],
        reproduction_file='reproduction.json', test_accessed=False))
    best, best_steps = {n: -1. for n in branches}, {n: 0 for n in branches}
    nonzero = {n: {m: 0 for m in row} for n,row in hashes.items()}
    iterator, started = iter(training), time.monotonic()
    for step in range(1, args.steps+1):
        warm = min(100, max(1, args.steps//10))
        factor = step/warm if step <= warm else .5*(1+math.cos(math.pi*(step-warm)/max(1,args.steps-warm)))
        totals = {n: dict(task=0., structure=0., total=0.) for n in branches}
        for name, branch in branches.items():
            branch.train(); optimizers[name].zero_grad(set_to_none=True)
            for group in optimizers[name].param_groups:
                group['lr'] = args.lr*factor
        for _ in range(args.accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(training); batch = next(iterator)
            pre, post, _, gt, _ = parse_batch(batch, device)
            reference, inputs = capture_inputs(front, pre, post)
            for name, branch in branches.items():
                output = branch(reference, inputs, gt.shape[-2:])
                if name in ('learn_q', 'learn_q_structure', 'rgb_rgb', 'rgb_sar') and not output['q'].requires_grad:
                    raise RuntimeError('Trainable Q detached: '+name)
                task = objective(output['prob'], gt)
                structure = task.new_zeros(())
                if name == 'learn_q_structure':
                    structure = structure_reconstruction_loss(output['q'], reference['ids'],
                        reference['tokens'].shape[1], reference['shape'], gt)
                loss = (task+args.structure_weight*structure)/args.accum
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss: '+name)
                loss.backward()
                totals[name]['task'] += float(task.detach())/args.accum
                totals[name]['structure'] += float(structure.detach())/args.accum
                totals[name]['total'] += float(loss.detach())
                del output, task, structure, loss
        report = gradients(branches, front)
        for name, branch in branches.items():
            for member, row in report[name].items():
                nonzero[name][member] += int(row['nonzero'])
            torch.nn.utils.clip_grad_norm_((p for p in branch.parameters() if p.requires_grad),
                                          1., error_if_nonfinite=True)
            optimizers[name].step()
        if step == 1 or step % 20 == 0 or step == args.steps:
            row = dict(step=step, crops=step*16, losses=totals, lr=args.lr*factor,
                       gradients=report, elapsed_seconds=time.monotonic()-started)
            append_json(out/'training.jsonl', row); print(row, flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(bundle, branches, validation, args, detailed=step == args.steps)
            assert_source(bundle)
            if implementation_hash() != code_hash:
                raise RuntimeError('Experiment code changed during training')
            if args.suite == 'structure' and member_hashes(branches)['fixed_q']['decoder'] != hashes['fixed_q']['decoder']:
                raise RuntimeError('Frozen copied decoder changed')
            if 'diagnostics' in result:
                save_json(out/'audit_final.json', result.pop('diagnostics'))
                result['diagnostics_file'] = 'audit_final.json'
            append_json(out/'validation.jsonl', dict(step=step, **result))
            for name, branch in branches.items():
                value = result['metrics'][name]['valid_miou']
                if not math.isfinite(value):
                    raise RuntimeError('Nonfinite metric: '+name)
                payload = checkpoint_payload(branch, name, step, result['metrics'][name], manifest)
                if value > best[name]:
                    best[name], best_steps[name] = value, step
                    save_pt(out/(name+'_best.pth'), payload)
                if step == args.steps:
                    save_pt(out/(name+'_last.pth'), payload)
            print('VAL', step, {n: m['valid_miou'] for n,m in result['metrics'].items()}, flush=True)
    assert_source(bundle)
    final_hash = member_hashes(branches)
    changed = {n: {m: final_hash[n][m] != old for m,old in row.items()} for n,row in hashes.items()}
    for name, branch in branches.items():
        for member, module in branch.named_children():
            active = any(p.requires_grad for p in module.parameters())
            if active and (not changed[name][member] or not nonzero[name][member]):
                raise RuntimeError('Trainable member never learned: '+name+'/'+member)
            if not active and (changed[name][member] or nonzero[name][member]):
                raise RuntimeError('Frozen member changed: '+name+'/'+member)
    gains = {a+'__'+b: 100*(result['metrics'][b]['valid_miou']-result['metrics'][a]['valid_miou'])
             for a,b in PAIRS[args.suite]}
    save_json(out/'summary.json', dict(complete=True, suite=args.suite, final_step=args.steps,
        phase='complete', optimizer_steps=args.steps,
        baseline_reproduction_pass=reproduction['passed'], reproduction_file='reproduction.json',
        smoke=args.smoke, test_accessed=False, best=best, best_steps=best_steps,
        final_metrics=result['metrics'], same_step_gains_pp=gains, q_drift=result['q_drift'],
        final_routing=result['routing'], partition_quality_totals=result['partition_quality_totals'],
        base_partition_quality_totals=result['base_partition_quality_totals'],
        final_state_sha256=final_hash, weights_changed=changed, nonzero_gradient_steps=nonzero,
        immutable_front_verified=True, diagnostics_file='audit_final.json',
        elapsed_seconds=time.monotonic()-wall_start,
        branch_forward_seconds_per_image=result['branch_forward_seconds_per_image'],
        shared_front_seconds_per_image=result['shared_front_seconds_per_image'],
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        resource_scope=manifest['resource_scope']))
    print('COMPLETE', args.suite, 'Train/Val only. No next suite launched.', flush=True)


if __name__ == '__main__':
    main()
