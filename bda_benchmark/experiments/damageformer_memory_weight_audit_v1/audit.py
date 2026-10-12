"""Same trained reader/supports; only redistribute external attention. No training."""
import argparse
import hashlib
import time
from pathlib import Path

import torch

from .core import TEMPERATURE, trace_weight_cases
from .inputs import load_parent, verify_parent_state, assert_parent_unchanged
from bda_benchmark.experiments.damageformer_memory_readout_audit_v1.core import ReaderStats, hard_counts
from bda_benchmark.experiments.damageformer_memory_readout_audit_v1.audit import (
    differences, sum_differences, finish_differences)
from bda_benchmark.experiments.damageformer_memory_readout_audit_v1.provenance import (
    load_trained, assert_trained_unchanged, verify_trained_metrics)
from bda_benchmark.experiments.damageformer_roi_memory_balanced_v2 import common
from bda_benchmark.experiments.damageformer_roi_memory_balanced_v2.memory import (
    retrieve, target_distribution_and_quality)
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import encode_regions, decode_regions
from bda_benchmark.experiments.damageformer_region_memory_v2.common import (
    baseline_policy, initial_gate, OutputLock, require_hard)
from bda_benchmark.experiments.damageformer_roi_exchange_v1.diagnostics import DiagnosticAccumulator


FORMAT = 'MEMORY_WEIGHT_AUDIT_V1'
CASES = ('normal', 'identity', 'uniform', 'cosine')
PAIRS = [('nomemory', 'normal'), ('nomemory', 'uniform'), ('nomemory', 'cosine'),
         ('normal', 'uniform'), ('normal', 'cosine'), ('uniform', 'cosine'),
         ('normal', 'identity'), ('source_hard', 'normal')]


def implementation_hash():
    # Also guard the observational helpers inherited from the previous audit.
    from bda_benchmark.experiments import damageformer_memory_readout_audit_v1 as previous
    result = hashlib.sha256()
    for root in (Path(__file__).parent, Path(previous.__file__).parent):
        for path in sorted(root.iterdir()):
            if path.suffix in ('.py', '.sh'):
                result.update((root.name + '/' + path.name).encode())
                result.update(path.read_bytes())
    return result.hexdigest()


def empty_replay():
    return dict(max_abs_context=0., max_abs_logits=0.,
        identity_region_argmax_disagreements=0, identity_pixel_argmax_disagreements=0,
        max_self_weight_error=0., max_external_mass_error=0., max_invalid_weight=0.,
        source_probability_max_abs=0.)


def merge_replay(total, row):
    for key, value in row.items():
        if key.endswith('argmax_disagreements'):
            total[key] = total.get(key, 0) + value
        else:
            total[key] = max(total.get(key, 0.), value)


@torch.no_grad()
def classify_all(tokens, mass, base, state, name):
    """Single retrieval per query chunk, reused identically by every intervention.

    No GT, B, class prediction, purity threshold, or label-based query filter is
    an argument. All positive-mass slots participate, including soft-only ones.
    """
    if len(tokens) != 1:
        raise ValueError('Audit evaluates one full image per batch')
    settings, bank = state['settings'], state['banks']['hard']
    event_id = bank['event_lookup'].get(common.event(name), -1)
    image_id = bank['image_lookup'].get(name, -1)
    active = (mass[0] > 0).nonzero().flatten()
    if not len(active):
        raise RuntimeError('No positive-mass regions')
    logits = {case: base.clone() for case in (*CASES, 'nomemory')}
    chunks, replay = [], empty_replay()
    for begin in range(0, len(active), settings['query_chunk']):
        selected = active[begin:begin + settings['query_chunk']]
        query, original = tokens[0, selected], base[0, selected]
        support, labels, valid, meta = retrieve(query, bank, event_id, image_id,
            settings['topk'], not settings['allow_same_event'], balanced=True)
        output = trace_weight_cases(state['heads']['normal'], query, original,
                                    support, labels, valid, meta['cosine'])
        for case, trace in output['traces'].items():
            logits[case][0, selected] = trace['logits']
        logits['nomemory'][0, selected] = state['heads']['nomemory'](query, original, mode='nomemory')
        merge_replay(replay, output['replay'])
        chunks.append(dict(selected=selected, base=original, labels=labels, valid=valid,
                           cosine=meta['cosine'], traces=output['traces']))
    return logits, chunks, replay


@torch.no_grad()
def evaluate(bundle, state, data, args):
    front, source = bundle['front'], bundle['heads']['hard']
    diagnostic = DiagnosticAccumulator(bundle['manifest']['arguments']['threshold'], PAIRS)
    reader, replay = ReaderStats(), empty_replay()
    matrices, differences_total, seen = {}, {}, []
    routing = dict(active_slots=0, hard_slots=0, tail_slots=0, mass=0., tail_mass=0.)
    for batch in data:
        pre, post, _, gt, names = common.parse_batch(batch, bundle['device'])
        reference = common.capture(front, pre, post)
        pool_route, read_route = common.routes(reference, bundle['manifest'], 'hard', 'hard')
        route_copies = [(route, route['q'].clone(), route['ids'].clone())
                        for route in (pool_route, read_route)]
        encoded = encode_regions(source, reference['features']['roi'], pool_route)
        tokens, mass = encoded['tokens'], encoded['mass']
        base = source.classifier(tokens)
        counts = hard_counts(pool_route)
        logits, chunks, info = classify_all(tokens, mass, base, state, names[0])
        merge_replay(replay, info)
        output_size = pre.shape[-2:]
        probability = decode_regions(base, read_route, mass, output_size)
        legacy = source(reference['features']['roi'], pool_route, output_size)
        torch.testing.assert_close(probability, legacy, atol=2e-5, rtol=2e-4)
        replay['source_probability_max_abs'] = max(replay['source_probability_max_abs'],
                                                   float((probability-legacy).abs().max()))
        predictions = dict(old_pixel=reference['pixel'].argmax(1),
                           old_region=reference['prob'].argmax(1), source_hard=probability.argmax(1))
        for case, values in logits.items():
            result = decode_regions(values, read_route, mass, output_size)
            if not torch.isfinite(result).all():
                raise RuntimeError('Nonfinite output: ' + case)
            predictions[case] = result.argmax(1)
        replay['identity_pixel_argmax_disagreements'] += int(
            (predictions['identity'] != predictions['normal']).sum())
        for route, q, ids in route_copies:
            if not torch.equal(route['q'], q) or not torch.equal(route['ids'], ids):
                raise RuntimeError('Intervention changed Q or region IDs')
        # Query GT enters ONLY post-forward, descriptive statistics.
        target, labelled_mass, gt_counts, _, _ = target_distribution_and_quality(gt, pool_route)
        if not torch.equal(counts, gt_counts):
            raise RuntimeError('Diagnostic hard-occupancy mismatch')
        for chunk in chunks:
            selected = chunk['selected']
            for case, trace in chunk['traces'].items():
                reader.update(case, chunk['base'], trace, chunk['labels'], chunk['valid'], chunk['cosine'],
                    target[0, selected], labelled_mass[0, selected], mass[0, selected], counts[0, selected],
                    temperature=TEMPERATURE)
        for case in ('identity', 'uniform', 'cosine', 'nomemory'):
            sum_differences(differences_total.setdefault(case, {}),
                            differences(logits['normal'], logits[case], mass, counts))
        for case, pred in predictions.items():
            common.cm_update(matrices.setdefault(case, torch.zeros(4, 4, dtype=torch.long)), gt, pred)
        diagnostic.update(gt, predictions, reference['building'], names)
        active, tail = mass > 0, (mass > 0) & (counts == 0)
        routing['active_slots'] += int(active.sum())
        routing['hard_slots'] += int((active & (counts > 0)).sum())
        routing['tail_slots'] += int(tail.sum())
        routing['mass'] += float(mass.double().sum())
        routing['tail_mass'] += float(mass[tail].double().sum())
        seen.extend(names)
        if len(seen) <= 2 or len(seen) % 10 == 0:
            print(f'memory weight audit {len(seen)}/{2 if args.smoke else 450}', flush=True)
    expected = bundle['val'][:2] if args.smoke else bundle['val']
    if seen != list(expected):
        raise RuntimeError('Image list/order does not match declared Val')
    routing['tail_mass_fraction'] = routing['tail_mass']/routing['mass'] if routing['mass'] else None
    identity_passed = (replay['identity_region_argmax_disagreements'] == 0
                       and replay['identity_pixel_argmax_disagreements'] == 0)
    return dict(final_metrics={name: common.metrics(cm) for name, cm in matrices.items()},
        image_count=len(seen), replay=replay, identity_passed=identity_passed, routing=routing,
        reader_stats=reader.finish(), case_differences=finish_differences(differences_total),
        diagnostics=diagnostic.finalize(), temperature=TEMPERATURE)


def run(args, parent):
    require_hard(args)
    out = common.new_output(args.output)
    started = time.monotonic()
    common.save_json(out/'summary.json', dict(format=FORMAT, complete=False,
        phase='loading', smoke=args.smoke, test_accessed=False))
    result = {}
    try:
        own_hash = implementation_hash()
        bundle = common.load_source(args)
        policy = baseline_policy(bundle, args)
        state = load_trained(args, bundle)
        verify_parent_state(parent, state)
        common.seed_everything(state['settings']['seed'])
        common.save_json(out/'baseline_policy.json', policy)
        common.save_json(out/'manifest.json', dict(format=FORMAT, arguments=vars(args),
            settings=state['settings'], temperature=TEMPERATURE, audit_code_sha256=own_hash,
            parent_file_sha256=parent['files'], input_file_sha256=state['files'],
            source_identity=common.source_identity(bundle, args), smoke=args.smoke, test_accessed=False,
            comparison='same trained reader: cosine-normal, uniform-normal, cosine-uniform; versus nomemory',
            prediction='normal uses original need_weights=False; identity manually replays V/out-proj',
            intervention='per-head self weight/external total unchanged; redistribute only valid external weights',
            frozen='front,B,Q,ROI,embedding,source logits,bank,neighbor IDs,labels,all reader parameters',
            gt_policy='Train support labels unchanged; query GT only for post-forward diagnostic evaluation',
            saved='aggregate JSON only; no checkpoints, bank copies, dense attention or dense predictions'))
        validation = common.loader(args.dataset, bundle['val'][:2] if args.smoke else bundle['val'],
            workers=args.workers, seed=state['settings']['seed']+3)
        if bundle['device'].type == 'cuda':
            torch.cuda.synchronize(bundle['device'])
            torch.cuda.reset_peak_memory_stats(bundle['device'])
        result = evaluate(bundle, state, validation, args)
        common.save_json(out/'audit_final.json', result.pop('diagnostics'))
        common.save_json(out/'reader_evidence.json', result)
        # Fail closed: a numerical implementation change must not look like an intervention gain.
        if not result['identity_passed']:
            raise RuntimeError('Identity replay changed predictions. Evidence saved; do not interpret effects or relax the gate.')
        source_gate = initial_gate(bundle, dict(metrics=result['final_metrics']), args, out, policy)
        learned_gate = verify_trained_metrics(result['final_metrics'], state, args.smoke)
        passed = None if args.smoke else source_gate['passed'] and learned_gate['passed']
        reproduction = dict(passed=passed, status='smoke_not_full_val' if args.smoke else
                            'passed' if passed else 'failed', source=source_gate, learned=learned_gate)
        common.save_json(out/'reproduction.json', reproduction)
        if not args.smoke and passed is not True:
            raise RuntimeError('Original trained results not reproduced; do not interpret intervention efficacy')
        assert_trained_unchanged(state, bundle)
        assert_parent_unchanged(parent)
        if implementation_hash() != own_hash:
            raise RuntimeError('Audit implementation changed during evaluation')
        if bundle['device'].type == 'cuda':
            torch.cuda.synchronize(bundle['device'])
        common.save_json(out/'summary.json', dict(result, format=FORMAT, complete=True,
            phase='complete', smoke=args.smoke, test_accessed=False, reproduction=reproduction,
            source_verified=True, bank_verified=True, head_verified=True,
            front_checkpoint_step=bundle['manifest']['checkpoint_step'],
            source_readout_step=bundle['provenance']['source_head_step'],
            memory_head_step=state['summary']['final_step'],
            elapsed_seconds=time.monotonic()-started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(bundle['device'])
                if bundle['device'].type == 'cuda' else None))
        print({name: row['valid_miou'] for name, row in result['final_metrics'].items()}, flush=True)
        print('MEMORY WEIGHT AUDIT COMPLETE. No weights/buffers updated. No Test.', flush=True)
    except Exception as exc:
        # Keep completed diagnostic values visible without promoting a failed
        # identity/reproduction/integrity check into successful evidence.
        evidence = {key: result[key] for key in ('image_count', 'replay', 'identity_passed',
                    'temperature', 'final_metrics', 'routing', 'reader_stats', 'case_differences')
                    if key in result}
        common.save_json(out/'summary.json', dict(evidence, format=FORMAT, complete=False, phase='failed',
            smoke=args.smoke, test_accessed=False, error=type(exc).__name__+': '+str(exc),
            elapsed_seconds=time.monotonic()-started))
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--parent_audit', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    options = parser.parse_args()
    args, parent = load_parent(options.parent_audit, options.output, options.device,
                               options.workers, options.smoke)
    with OutputLock(args.output):
        run(args, parent)


if __name__ == '__main__':
    main()
