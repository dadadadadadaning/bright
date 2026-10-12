"""Read-only actual-reader evidence; no optimizer, new bank, or Test access."""
import argparse
import hashlib
import time
from pathlib import Path

import torch

from .core import (ReaderStats, hard_counts, read_tail_fraction, split_corrections,
                   trace_reader)
from .provenance import load_trained, assert_trained_unchanged, verify_trained_metrics
from bda_benchmark.experiments.damageformer_roi_memory_balanced_v2 import common
from bda_benchmark.experiments.damageformer_roi_memory_balanced_v2.memory import (
    retrieve, shuffled_bank_labels, target_distribution_and_quality)
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import encode_regions, decode_regions
from bda_benchmark.experiments.damageformer_region_memory_v2.common import (
    add_arguments, baseline_policy, initial_gate, OutputLock, require_hard)
from bda_benchmark.experiments.damageformer_roi_exchange_v1.diagnostics import (
    DiagnosticAccumulator, accumulate, new_bucket, finalize_bucket)

FORMAT = 'MEMORY_READOUT_AUDIT_V1'
PAIRS = [('nomemory', 'normal'), ('source_hard', 'normal'),
         ('normal', 'shuffle_labels'), ('normal', 'off'),
         ('source_hard', 'hard_only'), ('source_hard', 'tail_only'),
         ('hard_only', 'normal'), ('tail_only', 'normal')]


def implementation_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.iterdir()):
        if path.suffix in ('.py', '.sh'):
            digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


@torch.no_grad()
def classify_all(tokens, mass, base, counts, state, name, shuffle_generator):
    """GT-free interventions on every actually classified mass>0 region.

    All classification settings come from the completed training manifest.
    The shuffle is once per image, using the parent's seed+505 stream.
    """
    if len(tokens) != 1:
        raise ValueError('This read-only audit uses one full image per batch')
    settings = state['settings']; bank = state['banks']['hard']
    event_id = bank['event_lookup'].get(common.event(name), -1)
    image_id = bank['image_lookup'].get(name, -1)
    exclude = not settings['allow_same_event']
    shuffled = shuffled_bank_labels(bank, event_id, image_id, exclude, shuffle_generator)
    active = (mass[0] > 0).nonzero().flatten()
    if not len(active):
        raise RuntimeError('Image has no positive-mass region')
    cases = ('normal', 'shuffle_labels', 'off')
    logits = {case: base.clone() for case in (*cases, 'nomemory')}
    observed = {case: base.clone() for case in cases}
    traces = []
    replay = dict(max_abs_context=0., max_abs_logits=0., argmax_disagreements=0,
                  label_shuffle_attention_max_abs=0.)
    for begin in range(0, len(active), settings['query_chunk']):
        selected = active[begin:begin+settings['query_chunk']]
        query, original = tokens[0, selected], base[0, selected]
        support, labels, valid, meta = retrieve(query, bank, event_id, image_id,
            settings['topk'], exclude, balanced=True)
        shuffled_support, shuffled_labels, shuffled_valid, shuffled_meta = retrieve(
            query, shuffled, event_id, image_id, settings['topk'], exclude, balanced=True)
        # Label-only intervention must not change retrieval keys or masks.
        if not torch.equal(support, shuffled_support) or not torch.equal(valid, shuffled_valid):
            raise RuntimeError('Label shuffle changed selected support features/mask')
        torch.testing.assert_close(meta['cosine'], shuffled_meta['cosine'], atol=0, rtol=0)
        rows = {}
        for case, injected, mask in (
                ('normal', labels, valid), ('shuffle_labels', shuffled_labels, valid),
                ('off', labels, torch.zeros_like(valid))):
            trace = trace_reader(state['heads']['normal'], query, original, support, injected, mask)
            logits[case][0, selected] = trace['logits']
            observed[case][0, selected] = trace['observation_logits']
            replay['max_abs_context'] = max(replay['max_abs_context'], trace['context_error'])
            replay['max_abs_logits'] = max(replay['max_abs_logits'], trace['logit_error'])
            replay['argmax_disagreements'] += trace['argmax_disagreements']
            rows[case] = dict(trace=trace, labels=injected, valid=mask, cosine=meta['cosine'])
        att_error = float((rows['normal']['trace']['weights'] -
                           rows['shuffle_labels']['trace']['weights']).abs().max())
        replay['label_shuffle_attention_max_abs'] = max(replay['label_shuffle_attention_max_abs'], att_error)
        torch.testing.assert_close(rows['normal']['trace']['weights'],
            rows['shuffle_labels']['trace']['weights'], atol=1e-6, rtol=1e-5,
            msg='Labels enter V only; fixed K/Q should preserve attention weights')
        logits['nomemory'][0, selected] = state['heads']['nomemory'](query, original, mode='nomemory')
        traces.append(dict(selected=selected, base=original, rows=rows))
    logits.update(split_corrections(base, logits['normal'], counts))
    return logits, observed, traces, replay


@torch.no_grad()
def differences(original, candidate, mass, counts):
    """Region-level changes, separately for all/hard/tail positive-mass slots."""
    result = {}
    difference = candidate-original
    centered = difference-difference.mean(-1, keepdim=True)
    for name, mask in dict(all=mass>0, hard=(mass>0)&(counts>0), tail=(mass>0)&(counts==0)).items():
        result[name] = dict(queries=int(mask.sum()),
            prob_l1_sum=float((candidate.softmax(-1)-original.softmax(-1)).abs().sum(-1)[mask].double().sum()),
            centered_delta_l1_sum=float(centered.abs().sum(-1)[mask].double().sum()),
            changed_regions=int((candidate.argmax(-1) != original.argmax(-1))[mask].sum()))
    return result


def sum_differences(total, row):
    for group, values in row.items():
        target = total.setdefault(group, {key: 0 for key in values})
        for key, value in values.items(): target[key] += value


def finish_differences(rows):
    result = {}
    for case, groups in rows.items():
        final = {}
        for group, row in groups.items():
            n = row['queries']
            final[group] = dict(queries=n, prob_l1=row['prob_l1_sum']/n if n else None,
                centered_delta_l1=row['centered_delta_l1_sum']/n if n else None,
                changed_region_fraction=row['changed_regions']/n if n else None)
        result[case] = dict(final['all'], groups=final)
    return result


@torch.no_grad()
def evaluate(bundle, state, data, args):
    source = bundle['heads']['hard']; front = bundle['front']
    diagnostic = DiagnosticAccumulator(bundle['manifest']['arguments']['threshold'], PAIRS)
    reader = ReaderStats(); tail_buckets = {}; case_differences = {}
    matrix = {}; seen = []
    generator = torch.Generator(device=bundle['device']).manual_seed(state['settings']['seed']+505)
    replay = dict(max_abs_context=0., max_abs_logits=0., argmax_disagreements=0,
                  label_shuffle_attention_max_abs=0., source_probability_max_abs=0.,
                  observation_pixel_argmax_disagreements=0)
    routing = dict(active_slots=0, hard_slots=0, tail_slots=0, allocated_slots=0,
                   mass=0., tail_mass=0., pixel_tail_sum=0., pixels=0,
                   foreground_pixel_tail_sum=0., foreground_pixels=0)
    for batch in data:
        pre, post, _, gt, names = common.parse_batch(batch, bundle['device'])
        reference = common.capture(front, pre, post)
        pool_route, read_route = common.routes(reference, bundle['manifest'], 'hard', 'hard')
        encoded = encode_regions(source, reference['features']['roi'], pool_route)
        tokens, mass = encoded['tokens'], encoded['mass']
        base = source.classifier(tokens)
        counts = hard_counts(pool_route)
        logits, observations, traces, info = classify_all(tokens, mass, base, counts, state, names[0], generator)
        # GT values are never used for forward, retrieval, or intervention.
        output_size = pre.shape[-2:]
        base_probability = decode_regions(base, read_route, mass, output_size)
        legacy = source(reference['features']['roi'], pool_route, output_size)
        torch.testing.assert_close(base_probability, legacy, atol=2e-5, rtol=2e-4)
        replay['source_probability_max_abs'] = max(replay['source_probability_max_abs'],
                                                  float((base_probability-legacy).abs().max()))
        predictions = dict(old_pixel=reference['pixel'].argmax(1), old_region=reference['prob'].argmax(1),
                           source_hard=base_probability.argmax(1))
        for case, values in logits.items():
            probability = decode_regions(values, read_route, mass, output_size)
            if not torch.isfinite(probability).all(): raise RuntimeError('Nonfinite probability: '+case)
            predictions[case] = probability.argmax(1)
        for case, values in observations.items():
            observed_pred = decode_regions(values, read_route, mass, output_size).argmax(1)
            replay['observation_pixel_argmax_disagreements'] += int((observed_pred != predictions[case]).sum())
        for key, value in info.items():
            replay[key] = replay[key]+value if key == 'argmax_disagreements' else max(replay[key], value)
        tail_fraction = read_tail_fraction(read_route, mass, counts, output_size)
        # Only now can query GT construct descriptive statistics.
        target, labelled_mass, gt_counts, _, _ = target_distribution_and_quality(gt, pool_route)
        if not torch.equal(counts, gt_counts): raise RuntimeError('Hard-occupancy diagnostic mismatch')
        for chunk in traces:
            selected = chunk['selected']
            for case, row in chunk['rows'].items():
                reader.update(case, chunk['base'], row['trace'], row['labels'], row['valid'], row['cosine'],
                    target[0, selected], labelled_mass[0, selected], mass[0, selected], counts[0, selected])
        for case in ('shuffle_labels', 'off', 'nomemory', 'hard_only', 'tail_only'):
            sum_differences(case_differences.setdefault(case, {}), differences(logits['normal'], logits[case], mass, counts))
        for case, pred in predictions.items():
            common.cm_update(matrix.setdefault(case, torch.zeros(4, 4, dtype=torch.long)), gt, pred)
        diagnostic.update(gt, predictions, reference['building'], names)
        y = gt.cpu(); fraction = tail_fraction.cpu(); preds = {k:v.cpu() for k,v in predictions.items()}
        valid = (y>=0)&(y<4); foreground = valid&(y>0)
        bands = dict(zero=fraction==0, positive_le_01=(fraction>0)&(fraction<=.1),
                     gt_01_le_05=(fraction>.1)&(fraction<=.5), gt_05=fraction>.5)
        for name, mask in bands.items():
            for suffix, scope in (('', valid), ('__foreground', foreground)):
                bucket = tail_buckets.setdefault(name+suffix, new_bucket(preds, PAIRS))
                accumulate(bucket, y, preds, mask&scope, PAIRS)
        active, tail = mass>0, (mass>0)&(counts==0)
        routing['active_slots'] += int(active.sum()); routing['hard_slots'] += int((active&(counts>0)).sum())
        routing['tail_slots'] += int(tail.sum()); routing['allocated_slots'] += mass.numel()
        routing['mass'] += float(mass.double().sum()); routing['tail_mass'] += float(mass[tail].double().sum())
        routing['pixels'] += fraction.numel(); routing['pixel_tail_sum'] += float(fraction.double().sum())
        routing['foreground_pixels'] += int(foreground.sum())
        routing['foreground_pixel_tail_sum'] += float(fraction[foreground].double().sum())
        seen.extend(names)
        if len(seen)<=2 or len(seen)%10==0:
            print(f'actual reader audit {len(seen)}/{2 if args.smoke else 450}', flush=True)
    expected_names = bundle['val'][:2] if args.smoke else bundle['val']
    if seen != list(expected_names): raise RuntimeError('Val names/order do not match the declared list')
    routing.update(tail_mass_fraction=routing['tail_mass']/routing['mass'] if routing['mass'] else None,
        mean_pixel_tail_fraction=routing['pixel_tail_sum']/routing['pixels'] if routing['pixels'] else None,
        mean_foreground_pixel_tail_fraction=routing['foreground_pixel_tail_sum']/routing['foreground_pixels'] if routing['foreground_pixels'] else None)
    return dict(final_metrics={k:common.metrics(v) for k,v in matrix.items()}, image_count=len(seen),
        replay=replay, routing=routing, reader_stats=reader.finish(),
        case_differences=finish_differences(case_differences), diagnostics=diagnostic.finalize(),
        tail_pixel_bins={k:finalize_bucket(v) for k,v in tail_buckets.items()})


def run(args):
    require_hard(args)
    out = common.new_output(args.output); started = time.monotonic()
    common.save_json(out/'summary.json', dict(format=FORMAT, complete=False, phase='loading', smoke=args.smoke, test_accessed=False))
    try:
        own_hash = implementation_hash()
        bundle = common.load_source(args)
        policy = baseline_policy(bundle, args)
        state = load_trained(args, bundle)
        common.seed_everything(state['settings']['seed'])
        common.save_json(out/'baseline_policy.json', policy)
        common.save_json(out/'manifest.json', dict(format=FORMAT, arguments=vars(args), settings=state['settings'],
            audit_code_sha256=own_hash, source_identity=common.source_identity(bundle, args),
            front_checkpoint_step=bundle['manifest']['checkpoint_step'],
            source_readout_step=bundle['provenance']['source_head_step'], memory_head_step=state['summary']['final_step'],
            input_file_sha256=state['files'], smoke=args.smoke, test_accessed=False,
            comparison='normal versus independently trained same-capacity nomemory at SAME final step',
            prediction_path='original need_weights=False; separate observational attention replay only',
            learned_metric_tolerance=1e-5, attention_atol=2e-5, attention_rtol=2e-4,
            interventions='label permutation; external support off; hard-only/tail-only correction, NO Q change',
            gt_policy='query GT only for post-forward diagnostics; no selection using query GT',
            saved='aggregate JSON only; no checkpoints, banks, dense attention maps, or predictions copied'))
        validation = common.loader(args.dataset, bundle['val'][:2] if args.smoke else bundle['val'],
            workers=args.workers, seed=state['settings']['seed']+3)
        if bundle['device'].type=='cuda':
            torch.cuda.synchronize(bundle['device']); torch.cuda.reset_peak_memory_stats(bundle['device'])
        result = evaluate(bundle, state, validation, args)
        common.save_json(out/'audit_final.json', result.pop('diagnostics'))
        common.save_json(out/'tail_pixel_bins.json', result.pop('tail_pixel_bins'))
        common.save_json(out/'reader_evidence.json', result)
        gate = initial_gate(bundle, dict(metrics=result['final_metrics']), args, out, policy)
        learned_gate = verify_trained_metrics(result['final_metrics'], state, args.smoke)
        reproduction = dict(passed=None if args.smoke else gate['passed'] and learned_gate['passed'],
            status='smoke_not_full_val' if args.smoke else 'passed' if gate['passed'] and learned_gate['passed'] else 'failed',
            source=gate, learned=learned_gate)
        common.save_json(out/'reproduction.json', reproduction)
        if not args.smoke and reproduction['passed'] is not True:
            raise RuntimeError('Trained reader reproduction failed; evidence saved, do not interpret mechanism yet')
        assert_trained_unchanged(state, bundle)
        if implementation_hash()!=own_hash: raise RuntimeError('Audit code changed during evaluation')
        if bundle['device'].type=='cuda': torch.cuda.synchronize(bundle['device'])
        common.save_json(out/'summary.json', dict(result, format=FORMAT, complete=True, smoke=args.smoke,
            phase='complete', test_accessed=False, source_verified=True, bank_verified=True, head_verified=True,
            front_checkpoint_step=bundle['manifest']['checkpoint_step'],
            source_readout_step=bundle['provenance']['source_head_step'], memory_head_step=state['summary']['final_step'],
            reproduction=reproduction, elapsed_seconds=time.monotonic()-started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(bundle['device']) if bundle['device'].type=='cuda' else None))
        print({key:value['valid_miou'] for key,value in result['final_metrics'].items()}, flush=True)
        print('ACTUAL READER AUDIT COMPLETE. No weights/buffers updated. No Test.', flush=True)
    except Exception as exc:
        common.save_json(out/'summary.json', dict(format=FORMAT, complete=False, phase='failed', smoke=args.smoke,
            test_accessed=False, error=type(exc).__name__+': '+str(exc), elapsed_seconds=time.monotonic()-started))
        raise


def main():
    parser = add_arguments(argparse.ArgumentParser(__doc__))
    parser.add_argument('--memory_run', required=True)
    parser.add_argument('--bank', required=True)
    args = parser.parse_args()
    with OutputLock(args.output): run(args)


if __name__=='__main__': main()
