"""Read-only hard-path evidence; no model interventions, optimization or Test."""
import argparse
import hashlib
import time
from pathlib import Path
import torch
from bda_benchmark.experiments.damageformer_roi_followup_v1 import common as source
from bda_benchmark.experiments.damageformer_roi_followup_v1.routing import encode_regions
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import unpool
from bda_benchmark.experiments.damageformer_roi_exchange_v1.diagnostics import DiagnosticAccumulator
from bda_benchmark.experiments.damageformer_region_memory_v2.common import (
    add_arguments, baseline_policy, initial_gate, OutputLock,
)
from .core import region_evidence, BIN_DEFINITIONS
from .aggregate import PoolAccumulator


def implementation_hash():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*')):
        if path.suffix in ('.py', '.sh'):
            h.update(path.name.encode()); h.update(path.read_bytes())
    return h.hexdigest()


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(bundle, data):
    front, head = bundle['front'], bundle['heads']['hard']
    device = bundle['device']
    threshold = bundle['manifest']['arguments']['threshold']
    diagnostic = DiagnosticAccumulator(threshold, pairs=[('old_pixel', 'source_hard')])
    pooling = PoolAccumulator(threshold)
    inference_seconds = 0.
    max_replay_error = 0.
    for batch in data:
        pre, post, _, gt, names = source.parse_batch(batch, device)
        sync(device); started = time.perf_counter()
        reference = source.capture(front, pre, post)
        route, _ = source.routes(reference, bundle['manifest'], 'hard', 'hard')
        feature = reference['features']['roi']
        prob = head(feature, route, gt.shape[-2:])  # ORIGINAL output, never replaced.
        sync(device); inference_seconds += time.perf_counter()-started
        encoded = encode_regions(head, feature, route)
        replay = unpool(head.classifier(encoded['tokens']).float().softmax(-1),
                        route['q'], route['ids'], route['shape'], gt.shape[-2:])
        torch.testing.assert_close(replay, prob, atol=2e-5, rtol=2e-4)
        max_replay_error = max(max_replay_error, float((replay-prob).abs().max()))
        attention = head.embedding.score(feature.float()).float().sigmoid()
        # No GT entered any operation above this line.
        evidence = region_evidence(route, attention, gt)
        torch.testing.assert_close(evidence['stats']['mass'], encoded['mass'], atol=2e-4, rtol=2e-4, check_dtype=False)
        torch.testing.assert_close(evidence['attention']['mass'], encoded['attention_mass'], atol=2e-4, rtol=2e-4, check_dtype=False)
        predictions = dict(old_pixel=reference['pixel'].argmax(1),
                           old_region=reference['prob'].argmax(1), source_hard=prob.argmax(1))
        diagnostic.update(gt, predictions, reference['building'], names)
        pooling.update(evidence, route, gt, predictions['old_pixel'], predictions['source_hard'], reference['building'])
        count = len(diagnostic.images)
        if count % 10 == 0 or count <= 2:
            print('pool evidence', count, flush=True)
    result = diagnostic.finalize()
    result['pooling_evidence'] = pooling.finalize()
    result['pooling_evidence']['protocol']['bins'] = BIN_DEFINITIONS
    result['pooling_evidence']['protocol']['gt_mapping'] = 'integer-area fractions per native feature cell; not the encoder receptive field'
    result['replay_probability_max_abs'] = max_replay_error
    result['inference_seconds_per_image'] = inference_seconds/max(1, result['image_count'])
    result['timing_scope'] = 'synchronized front capture + original hard head; excludes loader/GT diagnostics/replay'
    result['stats_bypass_alpha_tanh'] = float(head.embedding.alpha.tanh())
    return result


def main():
    parser = add_arguments(argparse.ArgumentParser(__doc__))
    args = parser.parse_args()
    if (args.source_head, args.pool_mode, args.read_mode) != ('hard', 'hard', 'hard'):
        parser.error('Only immutable hard/hard/hard is supported')
    code_hash = implementation_hash()
    with OutputLock(args.output):
        out = source.new_output(args.output)
        source.save_json(out/'summary.json', dict(complete=False, smoke=args.smoke, test_accessed=False))
        try:
            bundle = source.load_source(args, ['hard'])
            # Freeze reproduction policy BEFORE current evaluation, reusing audited history policy.
            policy = baseline_policy(bundle, args)
            source.save_json(out/'reproduction_policy.json', policy)
            source.save_json(out/'manifest.json', dict(format='POOL_EVIDENCE_V1', arguments=vars(args),
                provenance=bundle['provenance'], code_sha256=code_hash, weights_updated=False,
                buffers_updated=False, gt_used_for_routing=False, test_accessed=False, smoke=args.smoke))
            device = bundle['device']
            if device.type == 'cuda':
                sync(device); torch.cuda.reset_peak_memory_stats(device)
            data = source.loader(args.dataset, bundle['val'][:2] if args.smoke else bundle['val'], workers=args.workers)
            start = time.monotonic()
            result = evaluate(bundle, data)
            if result['image_count'] != (2 if args.smoke else 450):
                raise RuntimeError('Incomplete Val traversal')
            # Persist potentially useful evidence even when baseline reproduction fails.
            source.save_json(out/'audit_candidate.json', result)
            reproduction = initial_gate(bundle, dict(metrics=result['groups']['all']['metrics'],
                                       image_count=result['image_count']), args, out, policy)
            if implementation_hash() != code_hash:
                raise RuntimeError('Audit code changed during evaluation')
            source.assert_source(bundle)
            source.save_json(out/'audit_final.json', result)
            source.save_json(out/'summary.json', dict(complete=True, smoke=args.smoke,
                test_accessed=False, weights_updated=False, buffers_updated=False,
                image_count=result['image_count'], metrics=result['groups']['all']['metrics'],
                source_integrity_pass=True, reproduction_pass=reproduction['passed'],
                elapsed_seconds=time.monotonic()-start,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                resource_scope='entire audit, including GT statistics; NOT training memory requirement',
                audit_sha256=source.digest(out/'audit_final.json')))
            print({n: m['valid_miou'] for n, m in result['groups']['all']['metrics'].items()}, flush=True)
            print('POOL EVIDENCE COMPLETE. No weights/buffers updated. No Test.', flush=True)
        except Exception as exc:
            source.save_json(out/'summary.json', dict(complete=False, smoke=args.smoke,
                test_accessed=False, error_type=type(exc).__name__, error=str(exc)))
            raise


if __name__ == '__main__':
    main()
