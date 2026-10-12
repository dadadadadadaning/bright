"""Read-only 2x2 gather/read interventions within EACH frozen source head."""
import argparse
import json
import time
import torch
from .common import (add_arguments, load_source, assert_source, capture, routes,
                     loader, parse_batch, new_output, save_json, digest)
from .routing import forward_separate, route_diagnostics
from bda_benchmark.experiments.damageformer_roi_exchange_v1.audit import verify_reproduction
from bda_benchmark.experiments.damageformer_roi_exchange_v1.diagnostics import DiagnosticAccumulator

COMBINATIONS = {'HH': ('hard', 'hard'), 'EH': ('exchange', 'hard'),
                'HE': ('hard', 'exchange'), 'EE': ('exchange', 'exchange')}


def reproduction_differences(actual, expected):
    """Return explicit metric deltas before applying the strict source check."""
    rows = {}
    for name, values in actual.items():
        if name not in expected:
            rows[name] = {'missing_reference': True}
            continue
        row = {}
        for key in ('valid_miou', 'iou', 'precision', 'recall'):
            observed, reference = values[key], expected[name][key]
            a = observed if isinstance(observed, list) else [observed]
            b = reference if isinstance(reference, list) else [reference]
            if len(a) != len(b):
                row[key] = {'shape_mismatch': [len(a), len(b)]}
                continue
            deltas = [None if x is None or y is None else float(x-y)
                      for x, y in zip(a, b)]
            finite = [abs(x) for x in deltas if x is not None]
            row[key] = dict(observed=a, reference=b, delta=deltas,
                            max_abs=max(finite, default=0.))
        rows[name] = row
    return rows


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(bundle, data, combinations):
    front, heads, device = bundle['front'], bundle['heads'], bundle['device']
    comparisons = []
    for name in heads:
        for a, b in (('EE','EH'), ('HH','EH'), ('HH','HE'), ('HH','EE')):
            if a in combinations and b in combinations:
                comparisons.append((name+'_'+a, name+'_'+b))
    diagnostic = DiagnosticAccumulator(bundle['manifest']['arguments']['threshold'], pairs=comparisons)
    route_rows, timings, max_error = {}, {}, {n: 0. for n in heads}
    count = 0
    for batch in data:
        pre, post, _, gt, names = parse_batch(batch, device)
        reference = capture(front, pre, post)
        predictions = {'old_pixel': reference['pixel'].argmax(1), 'old_region': reference['prob'].argmax(1)}
        for name, head in heads.items():
            a, b = routes(reference, bundle['manifest'], name, name)
            original = head(reference['features']['roi'], a, gt.shape[-2:])
            predictions[name+'_original'] = original.argmax(1)
            # Always verify each source diagonal, even for a minimal EH/EE audit.
            replay = forward_separate(head, reference['features']['roi'], a, b, gt.shape[-2:])['prob']
            torch.testing.assert_close(replay, original, atol=2e-5, rtol=2e-4)
            max_error[name] = max(max_error[name], float((replay-original).abs().max()))
            for combo in combinations:
                label = name+'_'+combo
                pool_route, read_route = routes(reference, bundle['manifest'], *COMBINATIONS[combo])
                synchronize(device); start = time.perf_counter()
                result = forward_separate(head, reference['features']['roi'], pool_route, read_route, gt.shape[-2:])
                synchronize(device)
                timings[label] = timings.get(label, 0.) + time.perf_counter()-start
                if not torch.isfinite(result['prob']).all():
                    raise RuntimeError('Nonfinite probability: '+label)
                predictions[label] = result['prob'].argmax(1)
                stats = route_diagnostics(pool_route, read_route, result['mass'], result['attention_mass'], reference['building'])
                row = route_rows.setdefault(label, {})
                for key, value in stats.items():
                    row[key] = row.get(key, 0.)+value*len(pre)
        diagnostic.update(gt, predictions, reference['building'], names)
        count += len(pre)
        if count % 25 == 0:
            print('routing audit', count, flush=True)
    result = diagnostic.finalize()
    result['routing'] = {n: {k: v/count for k,v in row.items()} for n,row in route_rows.items()}
    result['diagonal_probability_max_abs'] = max_error
    result['head_forward_seconds_per_image'] = {n: t/count for n,t in timings.items()}
    result['timing_scope'] = 'head encode+classify+read only, synchronized; excludes shared front and diagnostics'
    return result


def main():
    parser = add_arguments(argparse.ArgumentParser(__doc__))
    parser.add_argument('--heads', nargs='+', choices=('hard','exchange'), default=['hard','exchange'])
    parser.add_argument('--combinations', nargs='+', choices=tuple(COMBINATIONS), default=list(COMBINATIONS))
    args = parser.parse_args()
    if len(set(args.combinations)) != len(args.combinations):
        parser.error('Repeated combination')
    bundle = load_source(args, args.heads)
    out = new_output(args.output)
    manifest = dict(format='ROI_FOLLOWUP_ROUTING_AUDIT_V1', arguments=vars(args),
                    provenance=bundle['provenance'], combinations=COMBINATIONS,
                    weights_updated=False, buffers_updated=False, gt_used_for_routing=False,
                    empty_policy='strictly zero gather mass masked before native-grid read normalization',
                    test_accessed=False, smoke=args.smoke,
                    interpretation='fixed-weight input interventions, not independently retrained ablations')
    save_json(out/'manifest.json', manifest)
    save_json(out/'summary.json', dict(complete=False, smoke=args.smoke, test_accessed=False))
    data = loader(args.dataset, bundle['val'][:2] if args.smoke else bundle['val'], workers=args.workers)
    start = time.monotonic()
    result = evaluate(bundle, data, args.combinations)
    if result['image_count'] != (2 if args.smoke else 450):
        raise RuntimeError('Incomplete Val traversal')
    whole = result['groups']['all']['metrics']
    # The full traversal is expensive. Preserve its raw result and exact source
    # deltas even if the strict reproduction gate rejects the run afterward.
    # A candidate file is diagnostic evidence, never a completed audit.
    assert_source(bundle)
    result['provenance'] = manifest
    save_json(out/'audit_candidate.json', result)
    if not args.smoke:
        actual = {n: whole[n+'_original'] for n in args.heads}
        actual.update({n: whole[n] for n in ('old_pixel','old_region')})
        expected = bundle['summary']['final_metrics']
        reproduction = dict(source=reproduction_differences(actual, expected), diagonal={})
        for name in args.heads:
            combo = 'HH' if name == 'hard' else 'EE'
            key = name+'_'+combo
            if key in whole:
                reproduction['diagonal'][name] = reproduction_differences(
                    {name:whole[key]}, {name:whole[name+'_original']})[name]
        save_json(out/'reproduction_differences.json', reproduction)
        try:
            verify_reproduction(actual, expected)
            # Generalized diagonal must also preserve actual full-image scores.
            for name in args.heads:
                combo = 'HH' if name == 'hard' else 'EE'
                key = name+'_'+combo
                if key in whole:
                    verify_reproduction({name:whole[key]}, {name:whole[name+'_original']})
        except (ValueError, RuntimeError) as exc:
            save_json(out/'summary.json', dict(
                complete=False, smoke=False, test_accessed=False,
                image_count=result['image_count'], immutable_source_verified=True,
                final_metrics_reproduced=False,
                failure_type=type(exc).__name__, failure=str(exc),
                candidate_file='audit_candidate.json',
                candidate_sha256=digest(out/'audit_candidate.json'),
                reproduction_file='reproduction_differences.json'))
            raise
    save_json(out/'audit_final.json', result)
    save_json(out/'summary.json', dict(complete=True, smoke=args.smoke, test_accessed=False,
              image_count=result['image_count'], final_metrics=whole,
              immutable_source_verified=True, diagonal_probability_max_abs=result['diagonal_probability_max_abs'],
              final_metrics_reproduced=not args.smoke, elapsed_seconds=time.monotonic()-start,
              peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(bundle['device']) if bundle['device'].type=='cuda' else None,
              resource_scope='combined front + source heads + sequential interventions',
              audit_sha256=digest(out/'audit_final.json')))
    print(json.dumps({n:m['valid_miou'] for n,m in whole.items()}, indent=2), flush=True)
    print('ROUTING AUDIT COMPLETE. No weights/buffers updated. No Test.', flush=True)


if __name__ == '__main__':
    main()
