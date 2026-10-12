"""Independent matched fixed-Q / trainable-Q experiment; Train/Val only."""
import argparse
import math
import time

import torch

from bda_benchmark.experiments.damageformer_region_content_v1.core import verify_original_readout
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import routing_stats
from bda_benchmark.experiments.damageformer_roi_exchange_v1.diagnostics import DiagnosticAccumulator
from .common import (
    add_arguments, append_json, assert_source, cm_update, load_source, loader,
    metrics, new_output, objective, parse_batch, save_json, save_pt,
    seed_everything, source_identity, state_digest, validate_training_args,
)
from .q_model import capture_q_inputs, gradient_report, make_branches, verify_initial_replay
from .routing import route_diagnostics


def branch_arguments(bundle, args):
    return dict(pool_mode=args.pool_mode, read_mode=args.read_mode,
                epsilon=bundle['manifest']['arguments']['epsilon'],
                threshold=bundle['manifest']['arguments']['threshold'])


@torch.no_grad()
def evaluate(bundle, branches, data, args, *, detailed=False, check_replay=False):
    front, device = bundle['front'], bundle['device']
    kwargs = branch_arguments(bundle, args)
    for branch in branches.values():
        branch.eval()
    names = ('old_pixel', 'old_region', *branches)
    cms = {name: torch.zeros(4, 4, dtype=torch.long) for name in names}
    stats = {name: {} for name in branches}
    diagnostic = (DiagnosticAccumulator(threshold=kwargs['threshold'],
                    pairs=[('fixed_q', 'learn_q'), ('old_pixel', 'fixed_q'),
                           ('old_pixel', 'learn_q'), ('old_region', 'fixed_q'),
                           ('old_region', 'learn_q')]) if detailed else None)
    count, identity_error, region_error, replay = 0, 0., 0., None
    q_drift_sum, q_drift_max = 0., 0.
    for batch in data:
        pre, post, _, gt, image_names = parse_batch(batch, device)
        reference, pyramid = capture_q_inputs(front, pre, post)
        if check_replay and replay is None:
            replay = verify_initial_replay(branches, bundle['heads'][args.source_head],
                reference, pyramid, gt.shape[-2:], **kwargs)
        identity_error = max(identity_error, reference['identity_max_abs'])
        region_error = max(region_error, verify_original_readout(front, reference, gt.shape[-2:]))
        predictions = dict(old_pixel=reference['pixel'].argmax(1),
                           old_region=reference['prob'].argmax(1))
        for name, branch in branches.items():
            output = branch(reference, pyramid, gt.shape[-2:], **kwargs)
            predictions[name] = output['prob'].argmax(1)
            row = routing_stats(output['pool_route'], reference['building'])
            row.update(route_diagnostics(output['pool_route'], output['read_route'],
                       output['mass'], output['attention_mass'], reference['building']))
            row['base_q_entropy'] = float(-(output['q']*output['q'].clamp_min(1e-8).log()).sum(1).mean())
            for key, value in row.items():
                stats[name][key] = stats[name].get(key, 0.) + float(value)*len(pre)
            if name == 'learn_q':
                delta = (output['q']-reference['q']).abs()
                q_drift_sum += float(delta.mean())*len(pre)
                q_drift_max = max(q_drift_max, float(delta.max()))
        for name, prediction in predictions.items():
            cm_update(cms[name], gt, prediction)
        if diagnostic is not None:
            diagnostic.update(gt, predictions, reference['building'], image_names)
        count += len(pre)
        if count % 25 == 0:
            print('evaluate', count, flush=True)
    if count != (2 if args.smoke else 450):
        raise RuntimeError('Expected smoke Val2 or complete Val450')
    if check_replay and replay is None:
        raise RuntimeError('Initial replay did not run')
    result = dict(image_count=count, metrics={n: metrics(cm) for n, cm in cms.items()},
                  routing={n: {k: v/count for k, v in row.items()} for n, row in stats.items()},
                  identity_max_abs=identity_error, region_replay_max_abs=region_error,
                  learned_base_q_mean_abs_change=q_drift_sum/count,
                  learned_base_q_max_abs_change=q_drift_max, test_accessed=False)
    if replay is not None:
        result['initial_replay'] = replay
    if diagnostic is not None:
        result['diagnostics'] = diagnostic.finalize()
    return result


def verify_baselines(bundle, initial, args):
    if args.smoke:
        return
    source_metrics = bundle['summary']['final_metrics']
    for name in ('old_pixel', 'old_region'):
        actual, expected = initial['metrics'][name]['valid_miou'], source_metrics[name]['valid_miou']
        if not math.isfinite(actual) or abs(actual-expected) > 1e-5:
            raise RuntimeError('Full-Val source baseline reproduction failed: '+name)
    if args.pool_mode == args.read_mode == args.source_head:
        actual = initial['metrics']['fixed_q']['valid_miou']
        expected = source_metrics[args.source_head]['valid_miou']
        if not math.isfinite(actual) or abs(actual-expected) > 1e-5:
            raise RuntimeError('Full-Val chosen saved Readout reproduction failed')


def checkpoint_payload(branch, name, step, metric, manifest):
    return dict(format='ROI_Q_FOLLOWUP_V1', name=name, step=step,
                decoder=branch.decoder.state_dict(), readout=branch.readout.state_dict(),
                metrics=metric, manifest=manifest)


def main(argv=None):
    wall_start = time.monotonic()
    parser = add_arguments(argparse.ArgumentParser(__doc__), training=True)
    args = parser.parse_args(argv)
    try:
        validate_training_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.smoke and (args.steps != 5000 or args.eval_every != 500):
        parser.error('Formal fixed-Q/learned-Q comparison requires 5000 steps and Full Val every 500')
    seed_everything(args.seed)
    bundle = load_source(args)
    front, device = bundle['front'], bundle['device']
    branches = make_branches(front, bundle['heads'][args.source_head])
    initial_hash = {n: {m: state_digest(getattr(b, m)) for m in ('decoder', 'readout')}
                    for n, b in branches.items()}
    opts = {n: torch.optim.AdamW((p for p in b.parameters() if p.requires_grad),
                                lr=args.lr, weight_decay=.005) for n, b in branches.items()}
    out = new_output(args.output)
    identity = source_identity(bundle, args)
    identity['gradient_policy'] = ('Frozen V2 front/B/ROI and first RGB pyramid; copied complete '
        'Readouts train in both branches; copied decoder trains only for learn_q; Q gradients preserved')
    manifest = dict(format='ROI_Q_FOLLOWUP_V1', arguments=vars(args), identity=identity,
        provenance=bundle['provenance'], initial_state_sha256=initial_hash,
        source_head=args.source_head, frozen_front=True, frozen_building=True,
        learned_partition=True, no_memory=True, new_backbone=False, new_gate=False,
        smoke=args.smoke, test_accessed=False, paired_crop_batches=True,
        parameter_count={n: dict(total=sum(p.numel() for p in b.parameters()),
            trainable=sum(p.numel() for p in b.parameters() if p.requires_grad),
            decoder=sum(p.numel() for p in b.decoder.parameters()),
            readout=sum(p.numel() for p in b.readout.parameters())) for n, b in branches.items()},
        loss='identical existing full-image region NLL+Lovasz; no auxiliary losses',
        initialization='two independent exact decoder and complete chosen same-step saved Readout copies',
        fixed_q='original captured Q; copied decoder retained frozen for matched state and replay',
        learn_q='copied decoder on detached FIRST RGB pyramid; original iterative_assignment with zeros_like(B)',
        content='exact original pixel-head input including SoftROI; no learned-Q feedback to content',
        primary_endpoint='same-step final learn_q minus fixed_q; best checkpoints secondary',
        checkpoint_policy='only decoder+readout states, metadata, best+last; no front or optimizer',
        schedule='AdamW weight_decay=.005; 100-step warmup (smoke adapts), cosine, clip=1',
        resource_scope='combined frozen front and paired branches; not separate per-method resource cost')
    save_json(out/'manifest.json', manifest)
    save_json(out/'summary.json', dict(complete=False, smoke=args.smoke, test_accessed=False))
    training = loader(args.dataset, bundle['train'], True, args.batch, args.workers, seed=args.seed+2)
    validation = loader(args.dataset, bundle['val'][:2] if args.smoke else bundle['val'],
                        workers=args.workers, seed=args.seed+3)
    initial = evaluate(bundle, branches, validation, args, check_replay=True)
    verify_baselines(bundle, initial, args)
    assert_source(bundle)
    save_json(out/'initial.json', initial)
    best, best_steps = {n: -1. for n in branches}, {n: 0 for n in branches}
    nonzero_steps = {n: {m: 0 for m in ('decoder', 'readout')} for n in branches}
    iterator, start = iter(training), time.monotonic()
    kwargs = branch_arguments(bundle, args)
    for step in range(1, args.steps+1):
        warm = min(100, max(1, args.steps//10))
        factor = (step/warm if step <= warm else
                  .5*(1+math.cos(math.pi*(step-warm)/max(1, args.steps-warm))))
        totals = {n: 0. for n in branches}
        for name, branch in branches.items():
            branch.train()
            opts[name].zero_grad(set_to_none=True)
            for group in opts[name].param_groups:
                group['lr'] = args.lr*factor
        for _ in range(args.accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(training)
                batch = next(iterator)
            pre, post, _, gt, _ = parse_batch(batch, device)
            reference, pyramid = capture_q_inputs(front, pre, post)
            for name, branch in branches.items():
                # Crucially OUTSIDE capture's no_grad and historical Readout.forward.
                output = branch(reference, pyramid, gt.shape[-2:], **kwargs)
                if name == 'learn_q' and not output['q'].requires_grad:
                    raise RuntimeError('Learned partition graph was detached')
                loss = objective(output['prob'], gt)/args.accum
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss: '+name)
                loss.backward()
                totals[name] += float(loss.detach())
                del output, loss
        gradients = gradient_report(branches, front)
        for name, branch in branches.items():
            for member in ('decoder', 'readout'):
                nonzero_steps[name][member] += int(gradients[name][member]['nonzero'])
            torch.nn.utils.clip_grad_norm_((p for p in branch.parameters() if p.requires_grad),
                                          1., error_if_nonfinite=True)
            opts[name].step()
        if step == 1 or step % 20 == 0:
            row = dict(step=step, crops=step*16, losses=totals, lr=args.lr*factor,
                       gradients=gradients, elapsed_seconds=time.monotonic()-start)
            append_json(out/'training.jsonl', row)
            print(row, flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(bundle, branches, validation, args, detailed=step == args.steps)
            assert_source(bundle)
            if state_digest(branches['fixed_q'].decoder) != initial_hash['fixed_q']['decoder']:
                raise RuntimeError('Fixed-Q decoder parameters or buffers changed')
            if 'diagnostics' in result:
                save_json(out/'audit_final.json', result.pop('diagnostics'))
                result['diagnostics_file'] = 'audit_final.json'
            append_json(out/'validation.jsonl', dict(step=step, **result))
            for name, branch in branches.items():
                value = result['metrics'][name]['valid_miou']
                if not math.isfinite(value):
                    raise RuntimeError('Nonfinite validation metric: '+name)
                state = checkpoint_payload(branch, name, step, result['metrics'][name], manifest)
                if value > best[name]:
                    best[name], best_steps[name] = value, step
                    save_pt(out/(name+'_best.pth'), state)
                if step == args.steps:
                    save_pt(out/(name+'_last.pth'), state)
            print('VAL', step, {n: m['valid_miou'] for n, m in result['metrics'].items()}, flush=True)
    assert_source(bundle)
    final_hash = {n: {m: state_digest(getattr(b, m)) for m in ('decoder', 'readout')}
                  for n, b in branches.items()}
    changed = {n: {m: final_hash[n][m] != initial_hash[n][m] for m in ('decoder', 'readout')}
               for n in branches}
    if changed['fixed_q']['decoder'] or not changed['learn_q']['decoder']:
        raise RuntimeError('Fixed/learned decoder parameter-change contract failed')
    if nonzero_steps['learn_q']['decoder'] == 0 or any(
            not changed[n]['readout'] or nonzero_steps[n]['readout'] == 0 for n in branches):
        raise RuntimeError('No observed nonzero gradients and actual trainable-weight changes')
    save_json(out/'summary.json', dict(complete=True, final_step=args.steps,
        final_metrics=result['metrics'], best=best, best_steps=best_steps,
        final_state_sha256=final_hash, weights_changed=changed,
        nonzero_gradient_steps=nonzero_steps, immutable_front_verified=True,
        learned_q_gradient_verified=True, smoke=args.smoke, test_accessed=False,
        learned_base_q_mean_abs_change=result['learned_base_q_mean_abs_change'],
        learned_base_q_max_abs_change=result['learned_base_q_max_abs_change'],
        final_learn_minus_fixed_miou=result['metrics']['learn_q']['valid_miou']-result['metrics']['fixed_q']['valid_miou'],
        diagnostics_file='audit_final.json', elapsed_seconds=time.monotonic()-wall_start,
        train_and_periodic_eval_seconds=time.monotonic()-start,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        resource_scope=manifest['resource_scope']))
    print('Q FOLLOWUP COMPLETE. Fixed-step primary; best secondary. Train/Val only.', flush=True)


if __name__ == '__main__':
    main()
