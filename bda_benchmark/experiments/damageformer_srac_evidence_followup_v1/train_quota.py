"""Optional fresh J1/M0/M1 control: only event quota changes between M0 and M1."""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch

from bda_benchmark.experiments.damageformer_srac_fourstage_v2.label_audit import (
    transfer_checked_batch, validate_batch_labels,
)
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.losses import region_nll
from bda_benchmark.experiments.damageformer_srac_joint_context_v1.common import FrozenFrontCapture
from bda_benchmark.experiments.damageformer_srac_joint_context_v1.model import JointContextDecoder
from bda_benchmark.experiments.damageformer_srac_joint123_v1.train import full_image_loader
from bda_benchmark.experiments.damageformer_srac_joint123_v1.joint_runtime import (
    decoder_forward, upsample_probability, bank_fingerprint,
)
from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit import (
    load_bundle, input_fingerprints, diagnostic_code_fingerprint,
)
from bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_core import (
    reproduction_report, transition_counts, transition_report, region_retrieval_report, add_nested_counts,
    label_evidence, project_label_evidence, cohort_evidence, eligible_mask, distribution_inventory,
)
from bda_benchmark.experiments.damageformer_srac_cause_audit_v1.support import module_fingerprint
from bda_benchmark.experiments.damageformer_srac_partition_probe_v1.numeric_runtime import (
    configure_determinism, assert_determinism,
)
from bda_benchmark.experiments.damageformer_srac_predsplit_v1.audit import metrics
from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import select_ids, gather_support
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import (
    digest, device_bank, confusion_update, event, parse_batch, region_targets,
    save_json, save_pt, seed_everything,
)
from .build_banks import FORMAT as BANK_FORMAT
from .quota import add_arguments, evidence_gate, source_hash
from .cohorts import difficult_retrieval


ARMS = ('j1', 'm0', 'm1')
FORMAT = 'SRAC_EVENT_QUOTA_TRAIN_V1'
EVIDENCE_TEMPERATURE = 0.1
EVIDENCE_DEFINITION = (
    'Diagnostic cosine-softmax/0.1 weighted Train labels projected by Q; '
    'NOT actual Reader attention or calibrated posterior. Cohorts use this evaluation\'s '
    'same fresh J1 predictions for M0/M1; J1 changes across training checkpoints.'
)


def parser():
    p = add_arguments(argparse.ArgumentParser(__doc__))
    p.add_argument('--banks', required=True)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--pool_mode', choices=('native', 'grouped'), default='grouped')
    return p


def read_banks(args, bundle):
    root = Path(args.banks)
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('format') != BANK_FORMAT or manifest.get('complete') is not True:
        raise ValueError('Incomplete or unsupported quota banks')
    if manifest.get('test_accessed') is not False or manifest.get('train_only') is not True:
        raise ValueError('Banks must be Train-only')
    if not args.smoke and (manifest.get('smoke') is not False or manifest.get('evidence_gate', {}).get('passed') is not True):
        raise ValueError('Smoke/unapproved banks cannot enter formal training')
    if manifest.get('source_sha256') != source_hash():
        raise ValueError('Followup code changed since bank creation; preserve artifacts and create a new pair')
    if manifest.get('source_hashes') != bundle['hashes'] or manifest.get('source_head_hashes') != bundle['head_hashes']:
        raise ValueError('Quota banks and J parent differ')
    if manifest.get('reference_hashes') != bundle['reference_hashes']:
        raise ValueError('Quota source reference differs')
    paths = {key: Path(value) for key, value in manifest['original_input_paths'].items()}
    if input_fingerprints(paths) != manifest['original_input_sha256']:
        raise ValueError('Original artifacts or full Train shards changed')
    inputs = dict(quota_manifest=manifest_path, **{'original_' + key: path for key, path in paths.items()})
    banks = {}
    expected_classes = torch.bincount(bundle['bank_cpu']['distribution'].argmax(-1), minlength=4).tolist()
    if manifest['class_totals'] != expected_classes:
        raise ValueError('Quota pair changed the original per-class bank budget')
    for arm in ('m0', 'm1'):
        path = root / f'{arm}_bank.pt'
        row_path = root / f'{arm}_rows.json'
        if digest(path) != manifest['bank_sha256'][arm] or digest(row_path) != manifest['selected_rows_sha256'][arm]:
            raise ValueError('Quota bank/row-list hash mismatch')
        bank = torch.load(path, map_location='cpu', weights_only=False)
        if bank['checkpoint_sha256'] != bundle['hashes']['front_sha256'] or bank['event_names'] != bundle['bank_cpu']['event_names']:
            raise ValueError('Quota bank provenance mismatch')
        d = bank['distribution'].float()
        if not torch.isfinite(d).all() or (d < 0).any() or not torch.allclose(d.sum(-1), torch.ones(len(d)), atol=1e-5, rtol=0):
            raise ValueError('Invalid support label distribution')
        if torch.bincount(d.argmax(-1), minlength=4).tolist() != expected_classes:
            raise ValueError('Unequal class budget')
        if any(not torch.isfinite(bank[key]).all() for key in ('tokens', 'geometry')):
            raise ValueError('Nonfinite quota-bank content')
        banks[arm] = device_bank(bank, torch.device(args.device))
        inputs[arm + '_bank'] = path
        inputs[arm + '_rows'] = row_path
    return banks, manifest, inputs


def select_support(result, bank, event_id, image_id, topk):
    with torch.no_grad():
        ids = select_ids(result['tokens'][0], bank, event_id, image_id, topk, projector=None)
        return ids, gather_support(ids, bank)


def decode(decoder, arm, feature, result, support):
    return decoder_forward(decoder, 'j1' if arm == 'j1' else 'j2', feature, result, support, None)


@torch.no_grad()
def evaluate(capture, decoders, data, device, banks, bundle, guard, original=False):
    for decoder in decoders.values():
        decoder.eval()
    names = ('a1_pixel', 'direct_region', *ARMS)
    confusion = {name: torch.zeros(4, 4, dtype=torch.long) for name in names}
    per_event = {name: {} for name in names}
    pairs = ('m0_vs_j1', 'm1_vs_j1', 'm1_vs_m0')
    transitions = {key: torch.zeros(4, 4, 4, dtype=torch.long) for key in pairs}
    retrieval = {'m0': {}, 'm1': {}}
    difficult_cohorts, difficult_results = {}, {}
    event_cohorts, event_difficult, event_retrieval, eligible_by_event = {}, {}, {}, {}
    old_confusion = {name: torch.zeros(4, 4, dtype=torch.long) for name in ('a1_pixel', 'direct_region', 'j1', 'j2', 'j3')}
    maximum = {arm: 0.0 for arm in ARMS}
    count = 0
    for batch in data:
        pre, post, _, target, image_names = parse_batch(batch, device)
        name = str(image_names[0])
        guard.set_image(name)
        result, feature = capture(pre, post)
        event_name = event(name)
        event_names = bundle['bank']['event_names']
        event_id = event_names.index(event_name) if event_name in event_names else -1
        probabilities = dict(a1_pixel=result['old_pixel'].softmax(1), direct_region=result['prob'])
        supports, selected = {}, {}
        for arm, bank in banks.items():
            selected[arm], supports[arm] = select_support(result, bank, event_id, -1, bundle['topk'])
        for arm, decoder in decoders.items():
            output = decode(decoder, arm, feature, result, supports.get(arm))
            probabilities[arm] = upsample_probability(output['logits'], target.shape[-2:])
            maximum[arm] = max(maximum[arm], float((probabilities[arm] - probabilities['a1_pixel']).abs().max()))
        predicted = {}
        for arm, probability in probabilities.items():
            if not torch.isfinite(probability).all() or not torch.allclose(probability.sum(1), torch.ones_like(probability[:, 0]), atol=2e-5, rtol=0):
                raise ValueError('Non-normalized/nonfinite evaluation output: ' + arm)
            predicted[arm] = probability.argmax(1)
            confusion_update(confusion[arm], target, predicted[arm])
            per_event[arm].setdefault(event_name, torch.zeros(4, 4, dtype=torch.long))
            confusion_update(per_event[arm][event_name], target, predicted[arm])
        for first, second in (('j1', 'm0'), ('j1', 'm1'), ('m0', 'm1')):
            transitions[second + '_vs_' + first] += transition_counts(target, predicted[first], predicted[second])
        if original:
            old_prob = {key: probabilities[key] for key in ('a1_pixel', 'direct_region')}
            old_ids = select_ids(result['tokens'][0], bundle['bank'], event_id, -1, bundle['topk'], projector=None)
            metric_ids = select_ids(result['tokens'][0], bundle['bank'], event_id, -1, bundle['topk'], projector=bundle['metric'])
            cosine, projected = gather_support(old_ids, bundle['bank']), gather_support(metric_ids, bundle['bank'])
            for arm, decoder in bundle['decoders'].items():
                output = decoder_forward(decoder, arm, feature, result, cosine, projected)
                old_prob[arm] = upsample_probability(output['logits'], target.shape[-2:])
            for arm, probability in old_prob.items():
                confusion_update(old_confusion[arm], target, probability.argmax(1))
        # Reporting only, after all selections/classifications. Neither query GT
        # nor J1 error masks can change the selected rows, Reader inputs or loss.
        distribution, weight = region_targets(target, result)
        rows_cohort, rows_difficult, rows_retrieval = {}, {}, {}
        for arm, bank in banks.items():
            rows_retrieval[arm] = region_retrieval_report(selected[arm], bank, distribution[0], weight[0])
            evidence, supported, _diagnostic_weights = label_evidence(
                result['tokens'][0], selected[arm], bank, EVIDENCE_TEMPERATURE, projector=None)
            label_probability, support_mass, evidence_prediction = project_label_evidence(
                evidence, supported, result['q'], result['ids'], result['shape'], target.shape[-2:])
            rows_cohort[arm] = cohort_evidence(
                target, predicted['j1'], predicted[arm], evidence_prediction, support_mass, label_probability)
            rows_difficult[arm] = difficult_retrieval(
                target, predicted['j1'], selected[arm], bank, result['q'], result['ids'], result['shape'])
            if arm not in eligible_by_event.setdefault(event_name, {}):
                eligible_by_event[event_name][arm] = distribution_inventory(
                    bank['distribution'], eligible_mask(bank, event_id, -1))
        add_nested_counts(retrieval, rows_retrieval)
        add_nested_counts(difficult_cohorts, rows_cohort)
        add_nested_counts(difficult_results, rows_difficult)
        add_nested_counts(event_cohorts.setdefault(event_name, {}), rows_cohort)
        add_nested_counts(event_difficult.setdefault(event_name, {}), rows_difficult)
        add_nested_counts(event_retrieval.setdefault(event_name, {}), rows_retrieval)
        count += 1
        if count % 25 == 0:
            print(f'quota evaluation {count}', flush=True)
    return dict(metrics={name: metrics(c) for name, c in confusion.items()},
                per_event={arm: {name: metrics(c) for name, c in values.items()} for arm, values in per_event.items()},
                transitions={name: transition_report(c) for name, c in transitions.items()},
                retrieval=retrieval, max_abs_probability_to_a1=maximum, image_count=count,
                difficult_cohorts=difficult_cohorts, difficult_retrieval=difficult_results,
                per_event_difficult_cohorts=event_cohorts, per_event_difficult_retrieval=event_difficult,
                per_event_retrieval=event_retrieval, eligible_bank_by_query_event=eligible_by_event,
                label_evidence_temperature=EVIDENCE_TEMPERATURE, label_evidence_definition=EVIDENCE_DEFINITION,
                original_metrics={name: metrics(c) for name, c in old_confusion.items()} if original else None,
                test_accessed=False)


def append_json(path, value):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, allow_nan=False) + '\n')


def run(args):
    if args.smoke:
        if (args.steps, args.eval_every) != (20, 20):
            raise ValueError('Smoke requires --steps 20 --eval_every 20')
    elif (args.steps, args.eval_every) != (5000, 500):
        raise ValueError('Formal control requires --steps 5000 --eval_every 500')
    if args.workers < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('Invalid training arguments')
    evidence_gate(args)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    status = dict(complete=False, run_complete=False, smoke=args.smoke, test_accessed=False,
                  full_validation_evidence=False, gates_passed=False, training=True)
    save_json(out / 'summary.json', status)
    capture = None
    step, image_name = 0, None
    started = time.perf_counter()
    try:
        seed_everything(args.seed)
        configure_determinism()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        numerical = assert_determinism()
        device = torch.device(args.device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(device)
        bundle = load_bundle(args, device)
        gate = evidence_gate(args, bundle)
        banks, bank_manifest, extra_paths = read_banks(args, bundle)
        file_hashes = input_fingerprints(extra_paths)
        code_before = source_hash()
        seed_everything(args.seed)  # Frozen loaders must not determine fresh head RNG.
        template = JointContextDecoder(bundle['front'].base.clf_2, dim=bundle['cache']['config']['dim'],
                                       num_events=len(bundle['bank']['event_names']), slots=bundle['topk']).to(device)
        decoders = {arm: copy.deepcopy(template) for arm in ARMS}
        initial_head_hash = module_fingerprint(template)
        if any(module_fingerprint(d) != initial_head_hash for d in decoders.values()):
            raise ValueError('Fresh heads not identically initialized')
        optimizers = {arm: torch.optim.AdamW(d.parameters(), lr=args.lr, weight_decay=5e-3) for arm, d in decoders.items()}
        frozen = dict(front=module_fingerprint(bundle['front']), metric=module_fingerprint(bundle['metric']),
                      original_bank=bank_fingerprint(bundle['bank']),
                      **{arm: bank_fingerprint(bank) for arm, bank in banks.items()},
                      **{'old_' + arm: module_fingerprint(d) for arm, d in bundle['decoders'].items()})
        manifest = dict(format=FORMAT, arguments=vars(args), source_sha256=code_before,
                        source_hashes=bundle['hashes'], reference_hashes=bundle['reference_hashes'],
                        source_head_hashes=bundle['head_hashes'], quota_artifact_sha256=file_hashes,
                        quota_manifest=bank_manifest, numerical=numerical, evidence_gate=gate,
                        initial_head_sha256=initial_head_hash, head_parameters=sum(p.numel() for p in template.parameters()),
                        trainable='Fresh independent J1/M0/M1 JointContextDecoder; no front/Q/metric/bank training',
                        arms=dict(j1='Fresh learned-slot noMemory control', m0='Fresh cosine reader; reconstructed equal-cap quotas',
                                  m1='Fresh cosine reader; partial event balance, same per-class bank totals'),
                        schedule='Same full 1024 Train image/update for all three arms; batch=1, no augmentation',
                        loss='NLL plus Lovasz-Softmax; no class reweighting',
                        label_evidence_temperature=EVIDENCE_TEMPERATURE, label_evidence_definition=EVIDENCE_DEFINITION,
                        test_accessed=False, smoke=args.smoke, full_validation_evidence=False)
        save_json(out / 'manifest.json', manifest)
        used_train = bundle['train_names']
        if args.smoke:
            used_train, seen = [], set()
            for name in bundle['train_names']:
                if event(name) not in seen:
                    seen.add(event(name))
                    used_train.append(name)
                if len(used_train) == 4:
                    break
        names = bundle['val_names'][:2] if args.smoke else bundle['val_names']
        train_data = full_image_loader(args.dataset, used_train, True, args.workers, args.seed + 1)
        val_data = full_image_loader(args.dataset, names, False, args.workers, args.seed + 2)
        name_to_id = {name: i for i, name in enumerate(bundle['train_names'])}
        capture = FrozenFrontCapture(bundle['front'])
        from .native_pool import PoolGuard
        from bda_benchmark.script.standard_ML.train_S5_S8_GPU import lovasz_softmax
        with PoolGuard(out / 'native_pool', mode=args.pool_mode, capture='failure') as guard:
            initial = evaluate(capture, decoders, val_data, device, banks, bundle, guard, original=True)
            repro = reproduction_report(initial['original_metrics'], bundle['reference'], args.smoke, initial['image_count'])
            identity = all(initial['metrics'][arm]['confusion'] == initial['metrics']['a1_pixel']['confusion'] for arm in ARMS)
            identity = identity and max(initial['max_abs_probability_to_a1'].values()) <= 1e-6
            save_json(out / 'initial_identity.json', dict(**initial, reproduction=repro, identity_passed=identity))
            if not identity or (not args.smoke and repro['passed'] is not True):
                raise ValueError('Initial A1 identity or original J1/J2/J3 full-Val reproduction failed; zero updates')
            best = {arm: -1.0 for arm in ARMS}
            counts = {arm: 0 for arm in ARMS}
            iterator = iter(train_data)
            for step in range(1, args.steps + 1):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_data)
                    batch = next(iterator)
                check = validate_batch_labels(batch, step, 0)
                if not check['valid']:
                    save_json(out / 'invalid_labels.json', check)
                    raise ValueError('Invalid Train label')
                pre, post, _, target, image_names = transfer_checked_batch(batch, device)
                image_name = str(image_names[0])
                image_id = name_to_id[image_name]
                event_id = bundle['bank']['event_names'].index(event(image_name))
                guard.set_image(image_name)
                result, feature = capture(pre, post)
                supports = {arm: select_support(result, bank, event_id, image_id, bundle['topk'])[1] for arm, bank in banks.items()}
                warmup = min(500, max(1, args.steps // 10))
                factor = step / warmup if step <= warmup else .5 * (1 + math.cos(math.pi * (step-warmup) / max(1, args.steps-warmup)))
                losses = {}
                # Same stochastic seed per step/arm prevents arm order from deciding dropout.
                rng_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
                for arm, decoder in decoders.items():
                    decoder.train()
                    optimizer = optimizers[arm]
                    optimizer.zero_grad(set_to_none=True)
                    for group in optimizer.param_groups:
                        group['lr'] = args.lr * factor
                    with torch.random.fork_rng(devices=rng_devices):
                        torch.manual_seed(args.seed + step)
                        output = decode(decoder, arm, feature, result, supports.get(arm))
                        probability = upsample_probability(output['logits'], target.shape[-2:])
                        loss = (region_nll(probability, target) + lovasz_softmax(probability, target, classes='present', ignore=255)) if (target != 255).any() else probability.sum() * 0
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite loss')
                        loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    counts[arm] += 1
                    losses[arm] = dict(loss=float(loss.detach()), grad_norm=float(norm))
                if step == 1 or step % 20 == 0:
                    row = dict(step=step, image=image_name, lr=args.lr * factor, arms=losses)
                    append_json(out / 'training.jsonl', row)
                    print(json.dumps(row), flush=True)
                if step % args.eval_every == 0 or step == args.steps:
                    evaluation = evaluate(capture, decoders, val_data, device, banks, bundle, guard)
                    append_json(out / 'validation.jsonl', dict(step=step, **evaluation))
                    for arm in ARMS:
                        score = evaluation['metrics'][arm]['valid_miou']
                        state = dict(format=FORMAT, arm=arm, model=decoders[arm].state_dict(), optimizer=optimizers[arm].state_dict(),
                                     step=step, run_complete=False, smoke=args.smoke, test_accessed=False,
                                     manifest_sha256=digest(out / 'manifest.json'), valid_miou=score)
                        if score > best[arm]:
                            best[arm] = score
                            save_pt(out / f'{arm}_best.pth', state)
                        save_pt(out / f'{arm}_last.pth', state)
                    print('VAL ' + str(step) + ' ' + json.dumps({arm: evaluation['metrics'][arm]['valid_miou'] for arm in ARMS}), flush=True)
        pool_report = guard.report()
        if pool_report['restored'] is not True:
            raise ValueError('Pooling aliases were not restored')
        after = dict(front=module_fingerprint(bundle['front']), metric=module_fingerprint(bundle['metric']),
                     original_bank=bank_fingerprint(bundle['bank']),
                     **{arm: bank_fingerprint(bank) for arm, bank in banks.items()},
                     **{'old_' + arm: module_fingerprint(d) for arm, d in bundle['decoders'].items()})
        if frozen != after or input_fingerprints(extra_paths) != file_hashes or source_hash() != code_before:
            raise ValueError('Frozen evidence/bank/code changed')
        if input_fingerprints(bundle['input_paths']) != bundle['inputs_before'] or diagnostic_code_fingerprint(bundle['source_manifest']['format']) != bundle['code_before']:
            raise ValueError('Original artifact/code changed')
        if numerical != assert_determinism():
            raise ValueError('Numerical execution policy changed')
        if evidence_gate(args, bundle) != gate:
            raise ValueError('Evidence reports changed during training')
        for arm in ARMS:
            for suffix in ('best', 'last'):
                path = out / f'{arm}_{suffix}.pth'
                state = torch.load(path, map_location='cpu', weights_only=False)
                state.update(run_complete=not args.smoke, completed_steps=args.steps)
                save_pt(path, state)
        save_json(out / 'summary.json', dict(status, complete=True, run_complete=not args.smoke,
                  gates_passed=True, full_validation_evidence=not args.smoke,
                  best=best, final_metrics=evaluation['metrics'], train_image_counts=counts,
                  final_difficult_cohorts=evaluation['difficult_cohorts'],
                  final_difficult_retrieval=evaluation['difficult_retrieval'],
                  final_eligible_bank_by_query_event=evaluation['eligible_bank_by_query_event'],
                  label_evidence_temperature=EVIDENCE_TEMPERATURE, label_evidence_definition=EVIDENCE_DEFINITION,
                  initial_reproduction=repro, frozen_fingerprints_unchanged=True,
                  pool_guard=pool_report, wall_seconds=time.perf_counter()-started,
                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                  note='Only M1 minus reconstructed M0 tests quota policy; fresh J1 is the same-budget noMemory control'))
        print('QUOTA TRAINING COMPLETE. Independent J1/M0/M1; no Test. Smoke=' + str(args.smoke), flush=True)
    except Exception as error:
        save_json(out / 'summary.json', dict(status, failure=dict(type=type(error).__name__, message=str(error), step=step, image=image_name)))
        raise
    finally:
        if capture is not None:
            capture.close()


if __name__ == '__main__':
    run(parser().parse_args())
