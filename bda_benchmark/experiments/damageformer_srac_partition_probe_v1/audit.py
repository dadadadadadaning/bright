"""Isolated, read-only probes of B priors, iteration count/support, and edges.

No optimizer, Memory construction, live checkpoint modification, or Test access.
Partition interventions freeze the native adapted feature (including native E/H).
Boundary interventions freeze Q and recompute adapter/embedding with changed E.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from bda_benchmark.experiments.damageformer_srac_cause_audit_v1.core import capture, probability_read
from bda_benchmark.experiments.damageformer_srac_cause_audit_v1.support import module_fingerprint
from bda_benchmark.experiments.damageformer_srac_selective_readout_v1.boundary import boundary_maps
from bda_benchmark.experiments.damageformer_srac_selective_readout_v1.evidence import Evidence, LABELS
from bda_benchmark.experiments.damageformer_srac_selective_readout_v1.probability_check import validate_probability
from bda_benchmark.experiments.damageformer_srac_selective_readout_v1.runtime import load_fixed, reproduction_gate
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import loader, lists, new_output, save_json
from bda_benchmark.experiments.damageformer_srac_transformer_v1.common import parse_batch
from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import project_region_probabilities

from .core import run_partition, sparse_pool, embedding_with_partition
from .numeric_runtime import configure_determinism, assert_determinism


TOLERANCE = 2e-5  # Original per-image replay gate; not relaxed for this probe.


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'a1_checkpoint', 'dataset', 'train_list', 'val_list', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--suite', choices=('prior', 'iterations', 'boundary', 'scale', 'all'), default='prior')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--prior_strength', type=float, default=2.0)
    p.add_argument('--leakage', type=float, default=0.1)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--start_index', type=int, default=0)
    p.add_argument('--max_images', type=int, default=0)
    return p


def plans(args):
    """No compounded new mechanisms; native uses checkpoint settings."""
    result = {}
    if args.suite in ('prior', 'all'):
        result['b_strong'] = dict(prior_mode='legacy', prior_strength=args.prior_strength)
        result['b_bounded'] = dict(prior_mode='bounded', prior_strength=args.prior_strength, leakage=args.leakage)
    if args.suite in ('iterations', 'all'):
        result['iter1'] = dict(iterations=1)
        result['iter5'] = dict(iterations=5)
        result['radius2_iter3'] = dict(iterations=3, radius=2)
    if args.suite in ('scale', 'all'):
        result['stride4'] = dict(stride=4)
    if args.suite in ('boundary', 'all'):
        for name in ('zero', 'mean', 'change'):
            result['edge_' + name] = dict(boundary=name)
    return result


def mode_names(variants):
    modes = ['a1', 'a1_resize', 'region', 'native_probq']
    pairs = [('a1', 'region'), ('a1', 'a1_resize'), ('a1_resize', 'native_probq')]
    for name, spec in variants.items():
        modes.append(name)
        pairs.append(('region', name))
        if 'boundary' not in spec:
            modes.append(name + '_probq')
            pairs.append(('native_probq', name + '_probq'))
    return modes, pairs


def cpu_copy(value):
    return value.detach().to('cpu').contiguous().clone()


def byte_hash(value):
    return hashlib.sha256(cpu_copy(value).reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def compare_tensor(actual, expected):
    """CPU numeric/byte evidence survives a misleading GPU equality result."""
    actual = cpu_copy(actual)
    expected = cpu_copy(expected)
    shape_ok = actual.shape == expected.shape and actual.dtype == expected.dtype
    if not shape_ok:
        return dict(passed=False, shape=list(actual.shape), expected_shape=list(expected.shape))
    byte_equal = byte_hash(actual) == byte_hash(expected)
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    delta = (actual.double() - expected.double()).abs()
    changes = actual != expected
    indices = changes.nonzero()[:4]
    examples = []
    for index in indices:
        ix = tuple(index.tolist())
        av, ev = float(actual[ix]), float(expected[ix])
        examples.append(dict(index=list(ix), actual=av if math.isfinite(av) else str(av),
                             expected=ev if math.isfinite(ev) else str(ev)))
    return dict(passed=byte_equal and finite, byte_equal=byte_equal, finite=finite,
                changed_elements=int(changes.sum()), max_abs_difference=float(delta.max()) if finite else None,
                examples=examples)


def capture_inputs(model, pre, post):
    structural = {}
    def take_input(_module, values):
        structural['embedding'] = values[0].detach().clone()
    handle = model.partition.register_forward_pre_hook(take_input)
    try:
        native, aux = capture(model, pre, post)
    finally:
        handle.remove()
    if 'embedding' not in structural:
        raise RuntimeError('Partition input hook did not run')
    fixed = dict(F=aux['fused_roi'], B=aux['p_building'], structural=structural['embedding'],
                 Q=native['q'], ids=native['ids'], H=native['entropy_map'],
                 E=native['soft_boundary'], adapted=native['adapted_content'], native_prob=native['prob'])
    return native, fixed


def partition_prediction(model, fixed, part, size, adapted=None):
    tokens, _, _, _ = embedding_with_partition(
        model.embedding, fixed['adapted'] if adapted is None else adapted, fixed['F'], part,
    )
    logits = model.region_classifier(tokens)
    return project_region_probabilities(logits, part['q'], part['ids'], part['shape'], size)


def roundtrip(grid, part, size):
    pooled = sparse_pool(grid, part['q'], part['ids'], part['count'])
    return probability_read(pooled['tokens'], part['q'], part['ids'], part['shape'], size)


def partition_diagnostics(part, building, gt):
    """GT is used after predictions, never supplied to run_partition()."""
    q, ids, shape, count = (part[k] for k in ('q', 'ids', 'shape', 'count'))
    h, w = shape
    b = F.interpolate(building.float(), shape, mode='bilinear', align_corners=False)
    predicted_fg = (b >= .5).float()
    pred_pool = sparse_pool(predicted_fg, q, ids, count)
    mass = pred_pool['mass']
    pred_fg = pred_pool['tokens'][..., 0]
    valid = (gt >= 0) & (gt < 4)
    area = F.interpolate(F.one_hot(gt.clamp(0, 3).long(), 4).permute(0, 3, 1, 2).float()
                         * valid[:, None], shape, mode='area')
    gt_pool = sparse_pool(area, q, ids, count)
    class_mass = gt_pool['tokens'] * mass[..., None]
    labelled = class_mass.sum(-1)
    dist = class_mass / labelled.clamp_min(1e-8)[..., None]
    purity = dist.max(-1).values
    binary_mix = 2 * dist[..., 0] * dist[..., 1:].sum(-1)
    raw_entropy = -(q * q.clamp_min(1e-8).log()).sum(1)
    # Use one observable definition for captured native Q and every new Q.
    # Numerical positive supports differ from the core trace's allowed mask
    # if softmax underflows. Report both counts rather than mixing definitions.
    positive_count = (q > 0).sum(1)
    entropy_valid = torch.where(positive_count > 1, raw_entropy / positive_count.clamp_min(2).float().log(),
                                torch.zeros_like(raw_entropy))
    possible = part.get('candidate_valid')
    hard = ids[None].expand(q.shape[0], -1, -1).gather(1, q.argmax(1, keepdim=True))[:, 0]
    classes = {}
    for c, label in enumerate(LABELS):
        cmass = class_mass[..., c]
        weight = float(cmass.sum())
        classes[label] = dict(label_mass=weight, purity_sum=float((cmass * purity).sum()),
                             binary_mixing_sum=float((cmass * binary_mix).sum()),
                             own_class_fraction_sum=float((cmass * dist[..., c]).sum()),
                             entropy_sum=float((area[:, c].flatten(1) * entropy_valid).sum()))
    return dict(
        allocated=count, supports=q.shape[1], soft_occupied=int((mass > 1e-6).sum()),
        hard_occupied=sum(torch.unique(row).numel() for row in hard),
        tiny_nonempty_regions=int(((mass > 1e-6) & (mass < 1)).sum()),
        entropy_raw=float(raw_entropy.mean()), entropy_over_positive_candidates=float(entropy_valid.mean()),
        positive_candidates_mean=float(positive_count.float().mean()),
        geometric_candidates_mean=float(possible.sum(0).float().mean()) if possible is not None else float(q.shape[1]),
        predicted_binary_mixing=float((2 * pred_fg * (1-pred_fg) * mass).sum() / mass.sum()),
        feature_grid=[h, w], original_pixels_per_feature_cell=[gt.shape[-2]/h, gt.shape[-1]/w],
        per_class_sums=classes,
    )


def boundary_statistics(maps, q):
    uncertainty = 1 - q.square().sum(1)
    return dict(
        per_pixel_gini_mean=float(uncertainty.mean()),
        maps={name: dict(mean=float(value.mean()), maximum=float(value.max()),
                         fraction_above_half=float((value > .5).float().mean())) for name, value in maps.items()},
        note='Old edge=pairwise distribution change + average endpoint Gini before neighbor max; '
             'max(old)-max(change) is not in general max(uncertainty).',
    )


def source_digest():
    from bda_benchmark.experiments.damageformer_srac_selective_readout_v1.runtime import experiment_source_hash
    digest = hashlib.sha256(experiment_source_hash().encode())
    for path in sorted(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent/'run.sh']:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def run(args):
    if not math.isfinite(args.prior_strength) or args.prior_strength < 0 or not 0 < args.leakage < 1:
        raise ValueError('Need finite nonnegative prior_strength and leakage in (0,1)')
    if args.start_index < 0 or args.max_images < 0 or (args.smoke and (args.start_index or args.max_images)):
        raise ValueError('Choose smoke OR a diagnostic subset, with nonnegative indices')
    _, val = lists(args.train_list, args.val_list)
    if args.start_index >= len(val):
        raise ValueError('start_index outside Val')
    selected = val[:2] if args.smoke else val[args.start_index:args.start_index+args.max_images if args.max_images else None]
    subset = bool(args.smoke or args.start_index or args.max_images)
    out = new_output(args.output)
    status = dict(complete=False, gates_passed=False, smoke=args.smoke, diagnostic_subset=subset,
                  test_accessed=False, weights_updated=False, memory_accessed=False)
    save_json(out/'summary.json', status)
    model = None
    evidence = None
    current_image = None
    current_variant = None
    try:
        status['execution'] = configure_determinism()
        save_json(out/'summary.json', status)
        model, payload, manifest = load_fixed(args)
        execution = assert_determinism()
        if execution != status['execution']:
            raise RuntimeError('Numerical execution settings changed while loading the source')
        print('Strict numerical execution: ' + json.dumps(execution), flush=True)
        # These tests deliberately refer to the existing stride8 / three-iteration parent.
        if model.partition.stride != 8 or model.partition.iterations != 3:
            raise ValueError('This declared comparison requires parent stride=8 and iterations=3')
        model.requires_grad_(False).eval()
        before_state = module_fingerprint(model)
        device = torch.device(args.device)
        if device.type == 'cuda':
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(device)
        variants = plans(args)
        modes, pairs = mode_names(variants)
        evidence = Evidence(modes, pairs)
        manifest.update(probe_source_sha256=source_digest(), variants=variants,
                        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                        execution=execution, q_replay_tolerance=TOLERANCE,
                        inference_only=True, full_validation=not subset,
                        fixed='All weights, A1 features/B, original adapted content for Q probes; Q fixed for edge probes',
                        entropy_adapter='Parent H/log(9) preserved; trace H/log(allowed) and summary H/log(positive Q) are diagnostics only',
                        prediction_warning='Inference interventions, not retrained ablations or a deployment checkpoint',
                        selected_images=selected, checkpoint_duplicated=False)
        save_json(out/'manifest.json', manifest)
        data = loader(args.dataset, selected, False, 1, args.workers, seed=args.seed)
        routing_sums = {}
        max_q_replay = max_probability_replay = 0.
        start = time.perf_counter()
        with (out/'per_image.jsonl').open('x', encoding='utf-8') as journal:
            for image_index, batch in enumerate(data):
                pre, post, _, gt, names = parse_batch(batch, device)
                current_image = str(names[0])
                current_variant = 'native_capture'
                if assert_determinism() != execution:
                    raise RuntimeError('Numerical execution settings changed during the audit')
                with torch.autocast(device_type=device.type, enabled=False):
                    native, fixed = capture_inputs(model, pre.float(), post.float())
                    snapshots = {key: cpu_copy(value) for key, value in fixed.items()}
                    replay = run_partition(model.partition, fixed['structural'], fixed['B'])
                    qdelta = float((replay['q']-native['q']).abs().max())
                    if not torch.equal(replay['ids'], native['ids']) or not math.isfinite(qdelta) or qdelta > TOLERANCE:
                        raise RuntimeError('Native Q replay failed: max_abs_difference=' + str(qdelta))
                    rebuilt = partition_prediction(model, fixed, replay, gt.shape[-2:])
                    pdelta = float((rebuilt-native['prob']).abs().max())
                    if not math.isfinite(pdelta) or pdelta > TOLERANCE:
                        raise RuntimeError('Native embedding/projection replay failed: max_abs_difference=' + str(pdelta))
                    # The replay is ONLY a compatibility gate. In particular,
                    # boundary interventions must not replace the captured Q
                    # with a second CUDA scatter/softmax computation.
                    native_part = {key: native[key] for key in ('q', 'ids', 'shape', 'count', 'entropy_map')}
                    native_part['candidate_valid'] = replay['candidate_valid']
                    max_q_replay = max(max_q_replay, qdelta)
                    max_probability_replay = max(max_probability_replay, pdelta)
                    a1 = native['old_pixel'].float().softmax(1)
                    grid = F.interpolate(a1, replay['shape'], mode='area')
                    resized = F.interpolate(grid, gt.shape[-2:], mode='bilinear', align_corners=False)
                    resized = resized/resized.sum(1, keepdim=True).clamp_min(1e-8)
                    predictions = {}
                    checks = {}
                    def collect(name, probability):
                        checks[name] = validate_probability(probability, name)
                        predictions[name] = probability.argmax(1).cpu()
                    collect('a1', a1)
                    collect('a1_resize', resized)
                    collect('region', native['prob'])
                    collect('native_probq', roundtrip(grid, native_part, gt.shape[-2:]))
                    maps = boundary_maps(native['q'], native['ids'], native['shape']) if any('boundary' in s for s in variants.values()) else None
                    if maps is not None and float((maps['old']-native['soft_boundary']).abs().max()) > TOLERANCE:
                        raise RuntimeError('Native boundary replay failed')
                    image_routing = {'native': partition_diagnostics(native_part, fixed['B'], gt)}
                    traces = {'native': replay.get('trace', [])}
                    times = {}
                    for name, spec in variants.items():
                        current_variant = name
                        sync(device)
                        timer = time.perf_counter()
                        if 'boundary' in spec:
                            adapted, _ = model.adapter(fixed['F'], fixed['B'], maps[spec['boundary']], fixed['H'])
                            collect(name, partition_prediction(model, fixed, native_part, gt.shape[-2:], adapted=adapted))
                        else:
                            part = run_partition(model.partition, fixed['structural'], fixed['B'], **spec)
                            collect(name, partition_prediction(model, fixed, part, gt.shape[-2:]))
                            collect(name+'_probq', roundtrip(grid, part, gt.shape[-2:]))
                            image_routing[name] = partition_diagnostics(part, fixed['B'], gt)
                            traces[name] = part.get('trace', [])
                        sync(device)
                        times[name] = time.perf_counter()-timer
                    immutable = {key: compare_tensor(value, snapshots[key]) for key, value in fixed.items()}
                    if not all(item['passed'] for item in immutable.values()):
                        save_json(out/'tensor_failure.json', dict(image=current_image, variant=current_variant, checks=immutable))
                        raise RuntimeError('Frozen evidence mutated; CPU byte diagnostics saved')
                    bgrid = F.interpolate(fixed['B'], replay['shape'], mode='bilinear', align_corners=False)
                    evidence.update(gt.cpu(), predictions, bgrid, current_image)
                    for name, row in image_routing.items():
                        dest = routing_sums.setdefault(name, dict(images=0, per_class_sums={}))
                        dest['images'] += 1
                        for key in ('allocated', 'supports', 'soft_occupied', 'hard_occupied', 'tiny_nonempty_regions',
                                    'entropy_raw', 'entropy_over_positive_candidates', 'positive_candidates_mean',
                                    'geometric_candidates_mean', 'predicted_binary_mixing'):
                            dest[key] = dest.get(key, 0.) + row[key]
                        for label, values in row['per_class_sums'].items():
                            sums = dest['per_class_sums'].setdefault(label, {k: 0. for k in values})
                            for key, value in values.items():
                                sums[key] += value
                    record = dict(image=current_image, index=args.start_index+image_index,
                                  q_replay_max_abs=qdelta, probability_replay_max_abs=pdelta,
                                  fixed_unchanged=True, probability_checks=checks, routing=image_routing,
                                  iteration_trace=traces, variant_seconds=times,
                                  boundary=boundary_statistics(maps, native['q']) if maps is not None else None)
                    journal.write(json.dumps(record, allow_nan=False)+'\n')
                    journal.flush()
                if (image_index+1) % 10 == 0 or image_index+1 == len(selected):
                    print(f'partition probe {image_index+1}/{len(selected)} [{args.suite}]', flush=True)
        result = evidence.finalize()
        for row in routing_sums.values():
            for label, values in row['per_class_sums'].items():
                mass = values['label_mass']
                values['mean_purity'] = values['purity_sum']/mass if mass else None
                values['mean_binary_mixing'] = values['binary_mixing_sum']/mass if mass else None
                values['mean_own_class_fraction'] = values['own_class_fraction_sum']/mass if mass else None
                values['mean_entropy_positive_candidates'] = values['entropy_sum']/mass if mass else None
            row['image_means'] = {k: row[k]/row['images'] for k in row if k not in ('images', 'per_class_sums')}
        unchanged = module_fingerprint(model) == before_state
        if assert_determinism() != execution:
            raise RuntimeError('Numerical execution settings changed before final checks')
        repro = reproduction_gate(result, payload, subset)
        passed = unchanged and (subset or repro['passed'] is True)
        status.update(complete=passed, gates_passed=passed, image_count=result['image_count'],
                      full_validation_evidence=passed and not subset, weights_buffers_unchanged=unchanged,
                      reproduction=repro, max_q_replay_abs=max_q_replay, max_probability_replay_abs=max_probability_replay,
                      elapsed_seconds=time.perf_counter()-start,
                      peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None,
                      warning='No interventions establish retrained benefit; no change to old failed audits')
        result.update(**status, routing=routing_sums)
        save_json(out/'audit.json', result)
        save_json(out/'summary.json', status)
        if not passed:
            raise RuntimeError('Saved evidence, but full-Val reproduction/immutability gate FAILED')
        print({name: item['valid_miou'] for name, item in result['metrics'].items()}, flush=True)
        print('PARTITION PROBE COMPLETE. No training. No Memory. No Test.', flush=True)
    except Exception as error:
        status.update(complete=False, gates_passed=False, failure=dict(image=current_image, variant=current_variant,
                      type=type(error).__name__, message=str(error), diagnostics=getattr(error, 'diagnostics', None)))
        if evidence is not None and not (out/'audit.json').is_file():
            save_json(out/'audit_partial.json', {**evidence.finalize(), **status})
        save_json(out/'summary.json', status)
        raise


if __name__ == '__main__':
    run(parser().parse_args())
