"""A/B/C fixed-F/Q diagnostics. No optimizer, training, new partition or Memory."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_srac_cause_audit_v1.core import (
    capture, gt_mixing_diagnostics, probability_roundtrip,
)
from bda_benchmark.experiments.damageformer_srac_cause_audit_v1.support import module_fingerprint
from bda_benchmark.experiments.damageformer_srac_partition_probe_v1.numeric_runtime import (
    configure_determinism, assert_determinism,
)
from bda_benchmark.experiments.damageformer_srac_selective_readout_v1.evidence import Evidence
from bda_benchmark.experiments.damageformer_srac_selective_readout_v1.probability_check import validate_probability
from bda_benchmark.experiments.damageformer_srac_selective_readout_v1.runtime import load_fixed, reproduction_gate
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import lists, loader, parse_batch, digest
from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import project_region_probabilities

from .feature_core import feature_branches, damaged_margin_sums, resize_probability
from .grouped_pool import POOL_IMPLEMENTATION
from .shared import (new_output, save_json, source_hash, tensor_hash, close_report,
                     transition_counts, transition_summary)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'a1_checkpoint', 'dataset', 'train_list', 'val_list', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--start_index', type=int, default=0)
    p.add_argument('--max_images', type=int, default=0)
    p.add_argument('--embedding_interventions', action='store_true',
                   help='Optional fixed-weight dependency probes; not retrained ablations')
    return p


def embedding_probability(model, native, feature, size, mode='normal'):
    handles = []
    module = model.embedding
    adapted = native['adapted_content']
    if mode == 'no_structure':
        adapted = native['content_only']
    elif mode in ('uniform_attention', 'no_stats', 'no_geometry'):
        target = {'uniform_attention': module.score, 'no_stats': module.stats,
                  'no_geometry': module.geometry}[mode]
        # Zero score -> sigmoid(score)=0.5, not zero total pooling mass.
        handles.append(target.register_forward_hook(lambda _m, _i, output: torch.zeros_like(output)))
    elif mode != 'normal':
        raise ValueError(mode)
    try:
        tokens, _, _, _ = module(adapted, feature, native['q'], native['ids'],
                                 native['shape'], native['count'])
        return project_region_probabilities(model.region_classifier(tokens), native['q'],
                                            native['ids'], native['shape'], size)
    finally:
        for handle in handles:
            handle.remove()


def enforce_checks(checks):
    failed = [name for name, value in checks.items() if isinstance(value, dict) and value.get('passed') is False]
    if failed:
        error = RuntimeError('Stage numerical/replay contract failed: '+', '.join(failed))
        error.diagnostics = {name: checks[name] for name in failed}
        raise error


def pool_failure_snapshot(feature, q, ids, count, head):
    """Own CPU evidence BEFORE the kernel; safe to serialize after a CUDA fault."""
    tensors = dict(feature=feature, q=q, ids=ids, head_weight=head.weight)
    if head.bias is not None:
        tensors['head_bias'] = head.bias
    cpu = {name: value.detach().to('cpu').contiguous().clone() for name, value in tensors.items()}
    layout = {name: dict(shape=list(value.shape), stride=list(value.stride()),
                         dtype=str(value.dtype), device=str(value.device),
                         contiguous=value.is_contiguous()) for name, value in tensors.items()}
    return dict(tensors=cpu, count=count, shape=list(feature.shape[-2:]), layout=layout,
                pool_implementation=POOL_IMPLEMENTATION)


def save_failed_pool_inputs(out, snapshot, image_name):
    if snapshot is None:
        return None
    path = out/'failed_pool_inputs.pt'
    # Exclusive output, CPU-only tensors. Never retry the failed GPU computation.
    with path.open('xb') as handle:
        torch.save(dict(image=image_name, **snapshot), handle)
    return dict(file=path.name, sha256=digest(path), bytes=path.stat().st_size,
                contains='F/Q/IDs/A1 affine weights on CPU; no Train update, no GT')


@torch.no_grad()
def run(args):
    if args.start_index < 0 or args.max_images < 0 or (args.smoke and (args.start_index or args.max_images)):
        raise ValueError('Choose a two-image smoke OR a nonnegative diagnostic subset')
    _, val = lists(args.train_list, args.val_list)
    if args.start_index >= len(val):
        raise ValueError('start_index is outside Val')
    selected = val[:2] if args.smoke else val[args.start_index:args.start_index+args.max_images if args.max_images else None]
    subset = len(selected) != len(val) or args.smoke
    out = new_output(args.output)
    status = dict(complete=False, gates_passed=False, smoke=args.smoke, diagnostic_subset=subset,
                  full_validation_evidence=False, weights_updated=False, test_accessed=False,
                  memory_accessed=False, audit_type='feature', pool_implementation=POOL_IMPLEMENTATION)
    save_json(out/'summary.json', status)
    evidence, image_name, failure_snapshot = None, None, None
    current_phase, last_completed_phase = 'initialization', None
    val_index, device = None, None
    start = time.perf_counter()

    def mark_phase(name):
        nonlocal current_phase, last_completed_phase
        previous = current_phase
        current_phase = 'synchronize_before/'+name
        progress = dict(image=image_name, val_index_one_based=val_index, phase=current_phase,
                        previous_submitted_phase=previous, last_completed_phase=last_completed_phase,
                        pool_implementation=POOL_IMPLEMENTATION, complete=False)
        save_json(out/'progress.json', progress)
        if device is not None and device.type == 'cuda':
            torch.cuda.synchronize(device)
        last_completed_phase, current_phase = previous, name
        progress.update(phase=current_phase, last_completed_phase=last_completed_phase)
        if failure_snapshot is not None:
            progress['pool_input_layout'] = failure_snapshot['layout']
        save_json(out/'progress.json', progress)
    try:
        configure_determinism()
        # Explicit diagnostic policy, not a mutation of the trained model/source.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        execution = assert_determinism()
        model, payload, manifest = load_fixed(args)
        snapshot_metadata_path = Path(args.checkpoint).parent/'snapshot.json'
        snapshot_metadata_hash = digest(snapshot_metadata_path)
        if execution != assert_determinism():
            raise RuntimeError('Loading changed numerical execution settings')
        state_before = module_fingerprint(model)
        source_before = source_hash()
        device = torch.device(args.device)
        if device.type == 'cuda':
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(device)
        modes = ['a1', 'a1_resize', 'region', 'native_probq',
                 'pixel_grid', 'feature_mean', 'region_mean', 'grid_probq']
        pairs = [('a1', 'pixel_grid'), ('a1', 'region'), ('pixel_grid', 'feature_mean'),
                 ('feature_mean', 'region_mean'), ('region_mean', 'region'),
                 ('pixel_grid', 'grid_probq'), ('a1_resize', 'native_probq')]
        optional = ['no_structure', 'uniform_attention', 'no_stats', 'no_geometry'] if args.embedding_interventions else []
        modes.extend(optional)
        pairs.extend(('region', name) for name in optional)
        evidence = Evidence(modes, pairs)
        manifest.update(stage_source_sha256=source_before, execution=execution,
                        snapshot_metadata_sha256=snapshot_metadata_hash,
                        selected_images=selected, diagnostic_subset=subset, audit_type='feature',
                        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                        mean_definition='Q-normalized arithmetic mean, exact positive mass; empty token zero',
                        pool_implementation=POOL_IMPLEMENTATION,
                        reduction_policy='CPU integer routing only; grouped gather/sum on input device, no CUDA scatter in diagnostic mean',
                        failure_capture='CPU pool input snapshot written only on failure; progress saved before synchronized phases',
                        output_definition='A/B/C/grid_probq: softmax on Q grid then bilinear probability resize, align_corners=False',
                        native_a1_definition='original A1: resize logits then softmax; a1 vs pixel_grid measures this order difference',
                        source_capture='one parent forward per image; no recomputed Q replaces native Q',
                        optional_warning='Fixed-weight inference sensitivity only, not causal proof or trained benefit',
                        prediction_equivalence_atol=2e-5, prediction_equivalence_rtol=2e-5)
        save_json(out/'manifest.json', manifest)
        margin_total, mix_total, mix_pairs = {}, {}, {}
        max_checks = {}
        data = loader(args.dataset, selected, False, 1, args.workers, seed=args.seed)
        with (out/'per_image.jsonl').open('x', encoding='utf-8') as journal:
            for index, batch in enumerate(data):
                failure_snapshot = None
                current_phase = 'batch_transfer'
                image_name = selected[index]
                val_index = val.index(image_name)+1
                pre, post, _, gt, names = parse_batch(batch, device)
                image_name = str(names[0])
                val_index = val.index(image_name)+1 if image_name in val else None
                if image_name != selected[index]:
                    raise RuntimeError('Val image identity/order changed')
                if execution != assert_determinism() or any(m.training for m in model.modules()):
                    raise RuntimeError('Frozen/numerical execution policy changed')
                with torch.autocast(device_type=device.type, enabled=False):
                    mark_phase('native_capture')
                    native, auxiliary = capture(model, pre.float(), post.float())
                    mark_phase('capture_validation')
                    feature = auxiliary['fused_roi']
                    if tuple(feature.shape[-2:]) != tuple(native['shape']):
                        raise ValueError('A1 F and Q have different grids; no silent resize allowed')
                    frozen = dict(F=feature, B=auxiliary['p_building'], **{
                        k: native[k] for k in ('q', 'ids', 'tokens', 'adapted_content', 'content_only',
                                               'entropy_map', 'soft_boundary', 'prob')})
                    hashes = {k: tensor_hash(v) for k, v in frozen.items()}
                    mark_phase('snapshot_pool_inputs')
                    failure_snapshot = pool_failure_snapshot(feature, native['q'], native['ids'],
                                                             native['count'], model.base.clf_2)
                    probs, checks, logits = feature_branches(feature, native['q'], native['ids'],
                                                            native['count'], model.base.clf_2, gt.shape[-2:],
                                                            progress=mark_phase)
                    mark_phase('native_replay_checks')
                    native_a1 = native['old_pixel'].float().softmax(1)
                    checks['native_a1_replay'] = close_report(
                        F.interpolate(logits[0], gt.shape[-2:], mode='bilinear', align_corners=False).softmax(1), native_a1)
                    checks['native_region_replay'] = close_report(
                        embedding_probability(model, native, feature, gt.shape[-2:]), native['prob'])
                    # Failed comparisons are attached to the exception and saved in summary.json.
                    enforce_checks(checks)
                    mark_phase('native_probability_roundtrip')
                    grid = F.interpolate(native_a1, native['shape'], mode='area')
                    probs.update(a1=native_a1, region=native['prob'],
                                 a1_resize=resize_probability(grid, gt.shape[-2:]),
                                 native_probq=probability_roundtrip(grid, native['q'], native['ids'], native['count'], gt.shape[-2:]))
                    for name in optional:
                        mark_phase('embedding_intervention/'+name)
                        probs[name] = embedding_probability(model, native, feature, gt.shape[-2:], name)
                    mark_phase('probability_and_immutability_checks')
                    probability_checks = {name: validate_probability(p, name) for name, p in probs.items()}
                    unchanged = {k: tensor_hash(v) == hashes[k] for k, v in frozen.items()}
                    if not all(unchanged.values()):
                        raise RuntimeError('Captured evidence changed: '+str(unchanged))
                    predictions = {name: p.argmax(1).cpu() for name, p in probs.items()}
                    bgrid = F.interpolate(auxiliary['p_building'].float(), native['shape'], mode='bilinear', align_corners=False)
                    evidence.update(gt.cpu(), predictions, bgrid, image_name)
                    margin = damaged_margin_sums(*logits, gt)
                    for key, value in margin.items():
                        margin_total[key] = margin_total.get(key, 0.) + value
                    # GT begins here for Q-purity associations, never in prediction.
                    mark_phase('GT_mixing_diagnostics')
                    mixing, purity_bucket = gt_mixing_diagnostics(gt, native['q'], native['ids'], native['count'], native['shape'])
                    for label, values in mixing['per_class'].items():
                        dest = mix_total.setdefault(label, {})
                        for key, value in values.items():
                            dest[key] = dest.get(key, 0.) + value
                    full_bucket = F.interpolate(purity_bucket.float(), gt.shape[-2:], mode='nearest')[:, 0].long().cpu()
                    for bucket, label in enumerate(('purity_lt080', 'purity_080_095', 'purity_ge095')):
                        dest = mix_pairs.setdefault(label, {})
                        for a, b in pairs[:5]:
                            key = a+'__'+b
                            value = transition_counts(gt.cpu(), predictions[a], predictions[b], full_bucket == bucket)
                            dest[key] = dest.get(key, torch.zeros_like(value)) + value
                    for key, value in checks.items():
                        if isinstance(value, dict) and value.get('max_abs') is not None:
                            max_checks[key] = max(max_checks.get(key, 0.), value['max_abs'])
                    journal.write(json.dumps(dict(image=image_name, checks=checks, immutable=unchanged,
                                                  probability_checks=probability_checks, margins=margin,
                                                  paired=evidence.images[-1]), allow_nan=False)+'\n')
                    journal.flush()
                    mark_phase('image_complete')
                    failure_snapshot = None
                if (index+1) % 10 == 0 or index+1 == len(selected):
                    print(f'feature stage audit {index+1}/{len(selected)}', flush=True)
                # A DataLoader failure while fetching the NEXT batch must not be
                # attributed to the previous, successfully completed image.
                image_name, val_index = None, None
                current_phase, last_completed_phase = 'await_batch', 'image_complete'
                save_json(out/'progress.json', dict(complete=False, image=None,
                          val_index_one_based=None, phase=current_phase,
                          last_completed_phase=last_completed_phase,
                          completed_images=index+1, pool_implementation=POOL_IMPLEMENTATION))
        mark_phase('final_gates')
        result = evidence.finalize()
        repro = reproduction_gate(result, payload, subset)
        immutable = module_fingerprint(model) == state_before
        stable_inputs = (digest(args.checkpoint) == manifest['checkpoint_sha256']
                         and digest(args.a1_checkpoint) == manifest['a1_sha256']
                         and digest(args.train_list) == manifest['train_sha256']
                         and digest(args.val_list) == manifest['val_sha256']
                         and digest(snapshot_metadata_path) == snapshot_metadata_hash
                         and source_hash() == source_before)
        stable_execution = execution == assert_determinism()
        passed = (immutable and stable_inputs and stable_execution and result['image_count'] == len(selected)
                  and (subset or repro['passed'] is True))
        status.update(complete=passed, gates_passed=passed, full_validation_evidence=passed and not subset,
                      image_count=result['image_count'], weights_buffers_unchanged=immutable,
                      input_files_unchanged=stable_inputs, reproduction=repro,
                      numerical_execution_unchanged=stable_execution,
                      maximum_numerical_differences=max_checks, execution=execution,
                      elapsed_seconds=time.perf_counter()-start,
                      peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None)
        result.update(status, damaged_margin_sums=margin_total, mixing_by_GT_class=mix_total,
                      purity_associations={label: {pair: transition_summary(value) for pair, value in rows.items()}
                                           for label, rows in mix_pairs.items()},
                      interpretation='Inference interventions only. Affine checks apply before softmax. GT purity is not an mIoU ceiling.')
        # Include exact FP/TP changes; existing Evidence keeps full per-image matrices too.
        result['class_transitions'] = {key: transition_summary(torch.from_numpy(value))
                                       for key, value in evidence.total['joint'].items()}
        save_json(out/'audit.json', result)
        save_json(out/'summary.json', status)
        if not passed:
            raise RuntimeError('Evidence saved, but reproduction/immutability gate FAILED')
        save_json(out/'progress.json', dict(complete=True, image_count=result['image_count'],
                  pool_implementation=POOL_IMPLEMENTATION, phase='complete'))
        print({name: row['valid_miou'] for name, row in result['metrics'].items()}, flush=True)
        print('FEATURE STAGE AUDIT COMPLETE. No training. No Memory. No Test.', flush=True)
    except Exception as error:
        captured, capture_error = None, None
        try:
            captured = save_failed_pool_inputs(out, failure_snapshot, image_name)
        except Exception as save_error:
            capture_error = str(save_error)
        status.update(complete=False, gates_passed=False, full_validation_evidence=False,
                      pool_implementation=POOL_IMPLEMENTATION,
                      failure=dict(image=image_name, val_index_one_based=val_index, phase=current_phase,
                                   last_completed_phase=last_completed_phase,
                                   type=type(error).__name__, message=str(error),
                                   pool_inputs=captured, pool_capture_error=capture_error,
                                   diagnostics=getattr(error, 'diagnostics', None)))
        save_json(out/'summary.json', status)
        if evidence is not None and not (out/'audit.json').exists():
            save_json(out/'audit_partial.json', {**evidence.finalize(), **status})
        raise


if __name__ == '__main__':
    run(parser().parse_args())
