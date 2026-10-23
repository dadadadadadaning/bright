"""Read-only evidence and restoration helpers retained from the verified plain replay.

This module contains no diagnostic worker or training entry point.
"""
from __future__ import annotations
import importlib.util
import json
import math
import hashlib
from pathlib import Path

FORMAT = 'BRIGHT_BACKWARD158_PLAIN_V1'
BASELINE_FORMAT = 'BRIGHT_BACKWARD158_DIAGNOSTIC_V1'
PARENT_FORMAT = 'BRIGHT_SEVERITY_MEMORY_V1'
ROOT = Path('/root/shared-nvme/BRIGHT-checkpoints/SRAC_SEVERITY_MEMORY_V1')
PARENT = ROOT / 'night_3407_zi_sjcjz/train_original'
BASELINE = ROOT / 'backward158_btuleshk'
BASELINE_RUNNER_SHA256 = 'd988cf60db94131bcbc5bafeb69ea94fd7d12a8b024a1394f4060fc03e56df51'
PACKAGE = Path('/root/shared-nvme/BRIGHT-master/New/bright_severity_memory_20261023')
TARGET_IMAGE = 'mexico-hurricane_00000077'
RESTORED_STEP = 150
END_STEP = 158
IDENTITY = {
    'experiment.py': '79d1deb96ee4ce6ec743b0d63d7aa76662d9ffe7de24e8e0dc724894dbff409b',
    'old_model_adapter.py': '9a81006d0191a3379145286d74a74f7d236d230534a980b0671ff3da4b1e8ba4',
    'severity_adapter.py': '3867e126f4bf5a66cb3dba0144cb2c514d5965c84059db7b65fe102191435ae3',
}


def require(value, message):
    if not value:
        raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str)+'\n')
    temporary.replace(path)

def restore_model_optimizer(torch, state, model, optimizer, saved, step):
    require(saved.get('step') == step, 'Checkpoint step mismatch')
    require(all(k in saved for k in ('model', 'optimizer', 'rng')), 'Checkpoint missing model/optimizer/RNG')
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    for key, value in (('model', model.state_dict()), ('optimizer', optimizer.state_dict())):
        require(state.state_fingerprint(value) == state.state_fingerprint(saved[key]), 'Restored '+key+' differs')
    counts = [int(v['step']) for v in optimizer.state.values() if 'step' in v]
    require(counts and all(v == step for v in counts), 'Restored optimizer step differs')
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite checkpoint parameters')

def reconstruct_iterator(exp, runtime, manifest, rows, restored_step):
    prefix = validate_prefix(rows, manifest, restored_step)
    iterator = exp.batches(runtime, manifest['actual_training_names'], manifest['seed'])
    for index, row in enumerate(prefix, 1):
        epoch, batch = next(iterator)
        require(epoch == row['epoch'] and str(batch[-1][0]) == row['image'],
                'Reconstructed data order differs at step '+str(index))
        del batch
        if index % 200 == 0:
            print('RECOVERY DATA ORDER verified '+str(index)+'/'+str(restored_step), flush=True)
    return iterator

def assert_unchanged(fingerprints):
    for path, digest in fingerprints.items():
        require(sha(path) == digest, 'Parent evidence changed: '+path)


def validate_prefix(rows, manifest, restored_step):
    require(len(rows) >= restored_step, 'Journal shorter than checkpoint')
    names = set(manifest['actual_training_names'])
    for expected, row in enumerate(rows[:restored_step], 1):
        require(row.get('step') == expected and row.get('optimizer_step') == expected,
                'Nonconsecutive journal/optimizer step')
        require(isinstance(row.get('epoch'), int) and row['epoch'] >= 0 and
                row.get('epoch_seed') == manifest['seed'] + row['epoch'], 'Journal epoch mismatch')
        require(row.get('variant') == 'original' and row.get('image') in names,
                'Journal variant or training membership mismatch')
    return rows[:restored_step]


def read_parent(package, parent):
    """Fail closed before any Torch import; no remote checkpoint SHA is invented."""
    package, parent = Path(package).resolve(), Path(parent).resolve()
    for name, digest in IDENTITY.items():
        require(sha(package / name) == digest, 'Unsupported installed version: '+name)
    plan, manifest, status = [read(parent / n) for n in ('plan.json', 'manifest.json', 'summary.json')]
    require(manifest['plan'] == plan, 'Parent manifest/plan mismatch')
    expected = dict(format=PARENT_FORMAT, variant='original', effective_mode='original', seed=3407,
                    endpoint_step=2500, read_weight=0, extension_files=IDENTITY,
                    scheduler_horizon=5000, maximum_lr=.0001, weight_decay=.005,
                    warmup_steps=500, grad_clip=1., optimizer='AdamW', support_arm='d',
                    post500_lr_multiplier=.2, test_accessed=False)
    for key, value in expected.items():
        require(plan.get(key) == value, 'Unexpected parent recipe: '+key)
    require(status.get('phase') == 'failed' and status.get('last_complete_step') == 157
            and status.get('failure_phase') == 'update_158'
            and status.get('current_image') == TARGET_IMAGE, 'Requires the known original step158 failure')
    first = read(parent / 'first_failure.json')
    require(first.get('phase') == 'update_158' and first.get('image') == TARGET_IMAGE
            and first.get('last_complete_step') == 157, 'First failure does not match')
    require(manifest['seed'] == 3407 and manifest['actual_training_names'] == manifest['train_names'],
            'Not the full Train scope')
    require(len(manifest['train_names']) == len(set(manifest['train_names'])) == 2918 and
            len(manifest['val_names']) == len(set(manifest['val_names'])) == 450,
            'Train/Val scope differs')
    require(not set(manifest['train_names']) & set(manifest['val_names']), 'Train/Val overlap')
    checkpoint = parent / 'checkpoints/checkpoint_000150.pt'
    sidecar = read(checkpoint.with_suffix('.pt.json'))
    digest = sha(checkpoint)
    require(sidecar.get('step') == RESTORED_STEP and sidecar.get('sha256') == digest
            and sidecar.get('roundtrip_verified') is True and sidecar.get('variant') == 'original',
            'Checkpoint150 sidecar/hash mismatch')
    last = status.get('last_checkpoint') or {}
    require(last.get('step') == RESTORED_STEP and last.get('sha256') == digest
            and last.get('roundtrip_verified') is True, 'Summary last checkpoint does not match checkpoint150')
    rows = [json.loads(s) for s in (parent / 'train.jsonl').read_text().splitlines() if s.strip()]
    require(len(rows) == 157, 'Expected exactly157 completed journal rows')
    validate_prefix(rows, manifest, len(rows))
    paths = [parent / n for n in ('plan.json', 'manifest.json', 'summary.json', 'train.jsonl', 'first_failure.json')]
    paths += [checkpoint, checkpoint.with_suffix('.pt.json')]
    fingerprints = {str(p): sha(p) for p in paths}
    spec = importlib.util.spec_from_file_location('_verified_severity_experiment', package / 'experiment.py')
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    return exp, plan, manifest, rows, checkpoint, fingerprints


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_pair(package, parent, baseline=BASELINE):
    """Stdlib-only validation of the successful instrumented comparison window.

    Returns read_parent's six values followed by a bounded baseline-info dict.
    No checkpoint deserialization, Torch import, GPU use or evidence promotion.
    """
    parent_result = read_parent(package, parent)
    _, _, parent_manifest, parent_rows, checkpoint, parent_fingerprints = parent_result
    parent, baseline = Path(parent).resolve(), Path(baseline).resolve()
    runner = baseline / 'runner.pyz'
    require(sha(runner) == BASELINE_RUNNER_SHA256, 'Baseline diagnostic runner SHA differs')
    summary, manifest, lineage = [read(baseline / name) for name in
                                  ('summary.json', 'manifest.json', 'lineage.json')]
    required_summary = dict(format=BASELINE_FORMAT, phase='diagnostic_window_complete',
                            diagnostic_complete=True, complete=False, training_complete=False,
                            full_validation_evidence=False, full_test_evidence=False,
                            test_accessed=False, restored_step=150, last_complete_step=158,
                            bounded_endpoint=158, new_complete_updates=8,
                            optimizer_updates_returned=8, original_failed_runs_unchanged=True,
                            historical_failure_resolved=False, exact_historical_replay=False)
    for key, expected in required_summary.items():
        require(summary.get(key) == expected, 'Incomplete or mismatched baseline summary: '+key)
    require(summary.get('error') is None and Path(summary['parent']).resolve() == parent,
            'Baseline parent/error mismatch')
    require(manifest.get('format') == BASELINE_FORMAT and
            manifest.get('parent_manifest') == parent_manifest and
            manifest.get('protocol_info') == parent_manifest['protocol_info'],
            'Baseline manifest/protocol differs')
    for key in ('rng_restored', 'optimizer_state_restored', 'data_order_prefix_verified'):
        require(manifest.get(key) is True, 'Baseline restore evidence missing: '+key)
    before_config, after_config = dict(parent_manifest['runtime_config']), dict(manifest['runtime_config'])
    before_config.pop('output', None)
    after_config.pop('output', None)
    require(before_config == after_config, 'Baseline runtime config differs beyond output')
    require(Path(lineage['parent']).resolve() == parent and
            Path(lineage['checkpoint']).resolve() == checkpoint.resolve() and
            lineage.get('checkpoint_sha256') == sha(checkpoint) and
            lineage.get('parent_fingerprints') == parent_fingerprints,
            'Baseline checkpoint or parent fingerprints differ')
    require(lineage.get('steps') == list(range(151, 159)) and
            lineage.get('target_image') == TARGET_IMAGE and
            lineage.get('formal_checkpoint_written') is False and
            lineage.get('test_accessed') is False and
            lineage.get('historical_failure_resolved') is False and
            lineage.get('exact_historical_replay') is False,
            'Baseline lineage scope differs')
    rows = [json.loads(line) for line in (baseline / 'diagnostic_updates.jsonl').read_text().splitlines()
            if line.strip()]
    require(len(rows) == 8, 'Baseline must contain all eight completed updates')
    paths = [runner] + [baseline / name for name in
             ('summary.json', 'manifest.json', 'lineage.json', 'diagnostic_updates.jsonl')]
    for step, row in zip(range(151, 159), rows):
        expected_image = parent_rows[step-1]['image'] if step <= 157 else TARGET_IMAGE
        require(row.get('step') == step and row.get('optimizer_step') == step and
                row.get('epoch') == 0 and row.get('image') == expected_image and
                row.get('variant') == 'original', 'Baseline update order differs at '+str(step))
        for key in ('original_loss', 'training_loss', 'grad_norm_preclip', 'lr'):
            require(_finite_number(row.get(key)), 'Baseline nonfinite/missing '+key)
        require(row['training_loss'] == row['original_loss'], 'Baseline objective differs')
        path = baseline / f'graph_summary_{step:06d}.json'
        graph = read(path)
        require(graph.get('format') == 'BRIGHT_BACKWARD_SAVED_INDEX_OBSERVER_V1' and
                graph.get('step') == step and graph.get('phase') == 'observer_step_complete' and
                graph.get('complete') is True and graph.get('coverage_complete') is True,
                'Baseline graph coverage incomplete at '+str(step))
        nodes = graph.get('nodes')
        require(isinstance(nodes, list) and len(nodes) > 0 and
                graph.get('observed_index_nodes') == len(nodes) and
                all(node.get('prehook_calls', 0) > 0 and
                    (node.get('statistics') or {}).get('in_bounds') is True for node in nodes) and
                graph.get('prehook_calls') == sum(node['prehook_calls'] for node in nodes) and
                not graph.get('unexecuted_index_nodes'),
                'Baseline index-node coverage differs at '+str(step))
        compact = row.get('observation_summary') or {}
        require(all(compact.get(key) == graph.get(key) for key in
                    ('complete', 'observed_index_nodes', 'prehook_calls', 'snapshot_bytes', 'coverage_complete')),
                'Baseline journal/graph summary mismatch at '+str(step))
        require(not (baseline / f'graph_failure_{step:06d}.json').exists(),
                'Baseline has contradictory graph failure at '+str(step))
        paths.append(path)
    require(not (baseline / 'first_failure.json').exists(), 'Baseline has a recorded first failure')
    info = dict(root=str(baseline), rows=rows, runner_sha256=BASELINE_RUNNER_SHA256,
                fingerprints={str(path): sha(path) for path in paths})
    return (*parent_result, info)


def compare_rows(baseline_rows, plain_rows):
    """Describe matched scalar outcomes; this does not compare full gradients/weights."""
    require(len(baseline_rows) == 8 and len(plain_rows) <= 8, 'Comparison window must be151–158')
    pairs = []
    for before, after in zip(baseline_rows, plain_rows):
        require(all(before.get(key) == after.get(key) for key in ('step', 'image', 'epoch', 'variant')),
                'Comparison sample/step mismatch')
        pair = dict(step=after['step'], image=after['image'], epoch=after['epoch'])
        for key in ('original_loss', 'training_loss', 'grad_norm_preclip'):
            require(_finite_number(before.get(key)) and _finite_number(after.get(key)),
                    'Comparison requires finite scalar '+key)
            pair[key] = dict(observed=before[key], plain=after[key],
                             signed_difference=after[key]-before[key],
                             absolute_difference=abs(after[key]-before[key]))
        pairs.append(pair)
    return dict(format='BRIGHT_BACKWARD158_PAIRED_SCALARS_V1', pairs=pairs,
                paired_updates=len(pairs), requested_updates=8, comparison_complete=len(pairs) == 8,
                numeric_success_threshold=None, full_gradients_compared=False,
                all_parameters_identical_verified=False, historical_failure_resolved=False,
                exact_historical_replay=False, full_validation_evidence=False, full_test_evidence=False,
                interpretation='Descriptive scalar comparison only. A pass does not establish identical parameters, '
                               'resolve the historical fault, or establish its software/hardware cause.')
