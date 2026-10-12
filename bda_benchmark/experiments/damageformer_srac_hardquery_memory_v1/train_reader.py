"""E2/E3: query-only/bank/hard x ungated/gated corrections of fixed J1."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import torch

from .io_utils import (add_source_arguments, load_verified_bundle, new_output, artifact_identity,
                       verify_unchanged, write_json, append_json, digest, fail_summary, finite_positive)
from .reader import GatedMemoryCorrection, retrieve_evidence, drop_support
from .prepare_queries import pool_base_probability, choose_smoke_names
from .runtime import evaluate, load_new_metrics, corrected_logits, query_only_evidence, TRAIN_ARMS


def parser():
    p = add_source_arguments(argparse.ArgumentParser(__doc__))
    p.add_argument('--protocol', required=True)
    p.add_argument('--metrics', required=True, help='Completed paired bank/hard metric output')
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--temperature', type=float, default=.1)
    p.add_argument('--bound', type=float, default=2.)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--support_dropout', type=float, default=.1)
    p.add_argument('--empty_support_dropout', type=float, default=.1)
    return p


def validate_args(args):
    for name in ('lr', 'temperature', 'bound'):
        finite_positive(getattr(args, name), name)
    if args.hidden < 1:
        raise ValueError('hidden must be positive')
    for name in ('support_dropout', 'empty_support_dropout'):
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= 1:
            raise ValueError('Invalid dropout probability: ' + name)
    if args.smoke:
        if args.steps != 20 or args.eval_every != 20:
            raise ValueError('Smoke requires --steps 20 --eval_every 20')
    elif args.steps != 5000 or args.eval_every != 500:
        raise ValueError('Matched formal comparison requires 5000 updates per head and eval_every=500')


def assert_initial_identity(result):
    for arm in TRAIN_ARMS:
        if result['metrics'][arm]['confusion'] != result['metrics']['j1']['confusion']:
            raise RuntimeError('Initial correction changed the fixed J1 predictions: ' + arm)
        if result['max_abs_probability_to_fixed_j1'][arm] != 0.:
            raise RuntimeError('Initial probability differs from fixed J1: ' + arm)


def run(args):
    from bda_benchmark.experiments.damageformer_srac_joint_context_v1.common import FrozenFrontCapture
    from bda_benchmark.experiments.damageformer_srac_joint123_v1.train import full_image_loader
    from bda_benchmark.experiments.damageformer_srac_joint123_v1.joint_runtime import decoder_forward, upsample_probability, module_fingerprint
    from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import parse_batch, event
    from bda_benchmark.experiments.damageformer_srac_fourstage_v2.losses import region_nll
    from bda_benchmark.script.standard_ML.train_S5_S8_GPU import lovasz_softmax
    from bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool import PoolGuard
    validate_args(args)
    output, capture = new_output(args.output), None
    try:
        bundle = load_verified_bundle(args)
        projectors, metric_hashes = load_new_metrics(args.metrics, bundle, smoke=args.smoke)
        metric_summary = Path(args.metrics) / 'summary.json'
        metric_summary_hash = digest(metric_summary)
        bundle['input_paths']['paired_metric_summary'] = metric_summary
        bundle['inputs_before']['paired_metric_summary'] = metric_summary_hash
        metric_states = {k: module_fingerprint(v) for k, v in projectors.items()}
        template = GatedMemoryCorrection(bundle['bank']['tokens'].shape[-1], args.hidden, args.bound).to(bundle['device'])
        heads = {arm: copy.deepcopy(template) for arm in TRAIN_ARMS}
        optimizers = {arm: torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=5e-3)
                      for arm, head in heads.items()}
        manifest = dict(format='SRAC_HARDQUERY_READER_V1', complete=False, smoke=args.smoke,
                        test_accessed=False, arguments=vars(args), **artifact_identity(bundle),
                        metric_checkpoint_sha256=metric_hashes,
                        metric_summary_sha256=metric_summary_hash,
                        arms=list(TRAIN_ARMS), trainable='New correction heads only; entire J1/front/Q/bank/metrics fixed',
                        output='fixed_J1_logits + SameQ(gated bounded region delta)',
                        query_control='Identical head/query/J1 inputs, zero historical posterior and quality; bank availability mask shared',
                        training='FP32 full images, batch=1, no augmentations; equal updates and image order',
                        empty_support='Hard identity fallback; zero-support cases do not teach a gate to reject nonempty bad evidence')
        write_json(output / 'manifest.json', manifest)
        capture = FrozenFrontCapture(bundle['front'])
        train_names = choose_smoke_names(bundle['train_names']) if args.smoke else bundle['train_names']
        val_names = bundle['val_names'][:2] if args.smoke else bundle['val_names']
        index_by_name = {name: i for i, name in enumerate(bundle['train_names'])}
        train_data = full_image_loader(args.dataset, train_names, True, args.workers, args.seed + 1)
        val_data = full_image_loader(args.dataset, val_names, False, args.workers, args.seed + 2)
        best = {arm: -1. for arm in TRAIN_ARMS}
        with PoolGuard(output / 'pool_debug', mode=args.pool_mode) as guard:
            initial = evaluate(bundle, capture, val_data, guard, projectors=projectors, heads=heads,
                               temperature=args.temperature, smoke=args.smoke)
            write_json(output / 'initial_identity.json', initial)
            if initial['reproduction']['passed'] is False:
                raise RuntimeError('Initial historical reproduction failed; no optimizer update')
            assert_initial_identity(initial)
            iterator = iter(train_data)
            for step in range(1, args.steps + 1):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_data)
                    batch = next(iterator)
                pre, post, _, target, names = parse_batch(batch, bundle['device'])
                name = str(names[0])
                if name not in index_by_name or name not in train_names:
                    raise ValueError('Unexpected training image')
                event_id = bundle['bank']['event_names'].index(event(name))
                image_id = index_by_name[name]
                guard.set_image(name)
                with torch.no_grad():
                    result, feature = capture(pre, post)
                    base = decoder_forward(bundle['decoders']['j1'], 'j1', feature, result, None, None)['logits']
                    base_probability = upsample_probability(base, target.shape[-2:])
                    base_region, _ = pool_base_probability(base_probability, result)
                    evidence = {key: retrieve_evidence(result['tokens'][0], bundle['bank'], event_id, image_id,
                                bundle['topk'], projector, args.temperature) for key, projector in projectors.items()}
                warmup = min(500, max(1, args.steps // 10))
                factor = step / warmup if step <= warmup else .5 * (1 + math.cos(
                    math.pi * (step - warmup) / max(1, args.steps - warmup)))
                rows = {}
                for arm, head in heads.items():
                    key, mode = arm.rsplit('_', 1)
                    head.train()
                    # Reusing the seed gives all arms the same slot and
                    # whole-query dropout draws despite different retrieved IDs.
                    generator = torch.Generator(device=bundle['device']).manual_seed(args.seed + 100000 + step)
                    item = drop_support(evidence['bank' if key == 'query' else key], args.support_dropout, generator,
                                        row_probability=args.empty_support_dropout)
                    if key == 'query':
                        item = query_only_evidence(item)
                    optimizer = optimizers[arm]
                    optimizer.zero_grad(set_to_none=True)
                    for group in optimizer.param_groups:
                        group['lr'] = args.lr * factor
                    delta, gate, _ = head(result['tokens'][0], base_region, item, mode=mode)
                    probability = upsample_probability(corrected_logits(base, delta, result), target.shape[-2:])
                    if (target != 255).any():
                        loss = region_nll(probability, target) + lovasz_softmax(probability, target, classes='present', ignore=255)
                    else:
                        loss = probability.sum() * 0
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    rows[arm] = dict(loss=float(loss.detach()), gradient_norm=float(gradient),
                                     mean_gate=float(gate.detach().mean()), supported_fraction=float(item['supported'].float().mean()))
                if step == 1 or step % 20 == 0:
                    row = dict(step=step, image=name, lr=args.lr * factor, arms=rows)
                    append_json(output / 'training.jsonl', row)
                    print(row, flush=True)
                if step % args.eval_every == 0 or step == args.steps:
                    final = evaluate(bundle, capture, val_data, guard, projectors=projectors, heads=heads,
                                     temperature=args.temperature, smoke=args.smoke,
                                     image_log=output / ('val_%05d_images.jsonl' % step))
                    append_json(output / 'validation.jsonl', dict(step=step, **final))
                    if final['reproduction']['passed'] is False:
                        raise RuntimeError('Frozen historical baseline drifted; checkpoints not promoted')
                    for arm, head in heads.items():
                        score = final['metrics'][arm]['valid_miou']
                        state = dict(**{k: v for k, v in manifest.items() if k != 'complete'}, complete=False,
                                     arm=arm, step=step, model=head.state_dict(), optimizer=optimizers[arm].state_dict(),
                                     valid_miou=score, dim=template.dim, hidden=args.hidden, bound=args.bound)
                        if score > best[arm]:
                            best[arm] = score
                            torch.save(state, output / (arm + '_best.pth'))
                        if step == args.steps:
                            torch.save(state, output / (arm + '_last.pth'))
                    print('VAL', step, {arm: final['metrics'][arm]['valid_miou'] for arm in ('j1', *TRAIN_ARMS)}, flush=True)
        checks = verify_unchanged(bundle)
        if any(module_fingerprint(projectors[k]) != v for k, v in metric_states.items()):
            raise RuntimeError('A frozen new metric changed')
        if any(digest(Path(args.metrics) / ('metric_' + k + '_last.pth')) != v for k, v in metric_hashes.items()):
            raise RuntimeError('New metric file changed')
        hashes = {}
        for arm in TRAIN_ARMS:
            hashes[arm] = {}
            for suffix in ('best', 'last'):
                path = output / (arm + '_' + suffix + '.pth')
                state = torch.load(path, map_location='cpu', weights_only=False)
                state.update(complete=True, completed_steps=args.steps)
                torch.save(state, path)
                hashes[arm][suffix] = digest(path)
        record = dict(**{k: v for k, v in manifest.items() if k != 'complete'}, complete=True,
                      full_validation_evidence=not args.smoke and final['image_count'] == 450,
                      steps=args.steps, updates_per_arm={arm: args.steps for arm in TRAIN_ARMS},
                      immutable=checks, best=best, final=final, checkpoint_sha256=hashes,
                      comparison='bank/hard vs query-only at same gate: historical evidence; hard-bank: retrieval training; gated-ungated: gate; all vs fixed J1')
        write_json(output / 'summary.json', record)
        return record
    except Exception as error:
        fail_summary(output, error)
        raise
    finally:
        if capture is not None:
            capture.close()


if __name__ == '__main__':
    run(parser().parse_args())
