"""Bounded original-model replay: checkpoint150 to step158, never a formal result."""
from __future__ import annotations
import fcntl
import importlib.util
import json
import math
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback

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


def mark(output, status, phase, **extra):
    status.update(phase=phase, **extra)
    write(output / 'summary.json', status)


def plain_update(exp, model, optimizer, example, runtime, factory, state, slots,
                 *, step, output, status, multiplier):
    """Call the pinned original update verbatim; no extra backward instrumentation."""
    mark(output, status, f'update_{step}')
    try:
        row = exp.update(model, optimizer, example, runtime, factory, state, slots,
                         step=step, read_weight=0, multiplier=multiplier)
    except BaseException:
        # exp.update also checks parameters after optimizer.step. With no wrapper
        # instrumentation, an exception cannot establish whether that call ran.
        status['failed_step_optimizer_execution_unknown'] = True
        status['failed_update_step'] = step
        raise
    status['optimizer_updates_returned'] += 1
    status['completed_update_calls'] += 1
    require(row.get('optimizer_step') == step, 'Original update returned wrong optimizer step')
    return row


def initial_status(output):
    return dict(format=FORMAT, phase='preflight', complete=False, diagnostic_complete=False,
                training_complete=False, full_validation_evidence=False, full_test_evidence=False,
                test_accessed=False, training_performed=False, optimizer_updates_returned=0,
                completed_update_calls=0,
                optimizer_counter_semantics='Only fully returned original update calls; a failed call may have executed optimizer.step',
                failed_step_optimizer_execution_unknown=False,
                restored_step=RESTORED_STEP, last_complete_step=RESTORED_STEP,
                bounded_endpoint=END_STEP, original_endpoint=2500, new_complete_updates=0,
                historical_failure_resolved=False, exact_historical_replay=False,
                observation_changes_timing=True, observation_synchronizes=True,
                observation_timing_reason='New process, phase logging and inherited CUDA_LAUNCH_BLOCKING=1; original update synchronizations retained',
                backward_index_observer=False, anomaly_detection=False,
                cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING'),
                saved_tensors_hooks_used=False, tensor_values_repaired=False,
                failed_samples_skipped=False, automatic_retry=False,
                parent=str(PARENT), baseline=str(BASELINE), output=str(output), pid=os.getpid())


def run_worker(args):
    output = args.output.resolve()
    status = initial_status(output)
    status['parent'] = str(args.parent.resolve())
    baseline = Path(getattr(args, 'baseline', BASELINE)).resolve()
    status['baseline'] = str(baseline)
    write(output / 'summary.json', status)
    started = time.monotonic()
    fingerprints = None
    baseline_info = None
    plain_rows = []
    try:
        require(os.environ.get('CUDA_LAUNCH_BLOCKING') == '1', 'CUDA_LAUNCH_BLOCKING=1 required before Torch import')
        require(not (output / 'manifest.json').exists(), 'Diagnostic output already used')
        with (args.parent.parent / '.severity_active.lock').open('a+') as active:
            fcntl.flock(active, fcntl.LOCK_EX | fcntl.LOCK_NB)
            exp, plan, parent_manifest, rows, checkpoint, fingerprints, baseline_info = check_pair(
                args.package, args.parent, baseline)
            bridge = exp.load_file(Path(plan['addon_dir']) / 'runtime_bridge.py', 'backward158_bridge')
            print('PLAIN PREFLIGHT: checkpoint150 and completed observed151–158 verified; original source verification', flush=True)
            exp.verify_plan(plan, bridge)
            write(output / 'lineage.json', dict(parent=str(args.parent.resolve()), checkpoint=str(checkpoint),
                 checkpoint_sha256=sha(checkpoint), checkpoint_hash_source='parent sidecar and summary; not an external pin',
                 parent_fingerprints=fingerprints, steps=list(range(151,159)), target_image=TARGET_IMAGE,
                 baseline=str(baseline), baseline_runner_sha256=BASELINE_RUNNER_SHA256,
                 baseline_fingerprints=baseline_info['fingerprints'],
                 formal_checkpoint_written=False, full_validation_evidence=False, test_accessed=False,
                 historical_failure_resolved=False, exact_historical_replay=False))
            bindings = bridge.load_runtime(plan['source'])
            torch, original, state = bindings.torch, bindings.original_train, bindings.legacy.state
            require(torch.cuda.is_available(), 'This diagnostic requires CUDA; CPU tests are not server evidence')
            require(not torch.is_anomaly_enabled(), 'Plain comparison requires anomaly detection disabled')
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            require(saved.get('format') == PARENT_FORMAT+'_CHECKPOINT' and saved.get('plan') == plan,
                    'Checkpoint recipe differs')
            for key in ('variant', 'effective_mode', 'read_weight', 'test_accessed'):
                require(saved.get(key) == plan[key], 'Checkpoint field mismatch: '+key)
            require(saved.get('protocol_info') == parent_manifest['protocol_info'], 'Checkpoint protocol differs')
            runtime_args = bridge.runtime_arguments(plan['source'], seed=3407, output=output)
            old_config, new_config = dict(parent_manifest['runtime_config']), dict(vars(runtime_args))
            old_config.pop('output', None)
            new_config.pop('output', None)
            require(old_config == new_config, 'Runtime config differs beyond output')
            old_adapter = exp.load_file(args.package / 'old_model_adapter.py', 'backward158_old_adapter')
            adapter = exp.load_file(args.package / 'severity_adapter.py', 'backward158_adapter')
            addon = Path(plan['addon_dir'])
            with exp.addon_imports(addon, bridge):
                support_module = bridge.named(addon / 'supports.py', 'readout_supports')
                import slots
                require(Path(slots.__file__).resolve() == (Path(plan['source']['package'])/'deps/original/slots.py').resolve(),
                        'Unexpected slots module')
                classifier = adapter.make_classifier(original.JointRegionMemoryClassifier, old_adapter, slots, mode='original')
                mark(output, status, 'load_runtime')
                print('DIAGNOSTIC LOAD: frozen model and Memory; checking inherited inputs', flush=True)
                with bindings.Runtime(runtime_args, output) as runtime:
                    original._formal_runtime(runtime, False)
                    full_train, full_val = original._names(runtime, False)
                    require(list(full_train) == parent_manifest['train_names'] and list(full_val) == parent_manifest['val_names'],
                            'Train/Val membership differs')
                    require(runtime.protocol_info == saved['protocol_info'], 'Runtime protocol differs')
                    original._checked_verify(runtime)
                    factory = support_module.SupportFactory(runtime, seed=3407)
                    original._seed(3407)
                    model = classifier(chunk_size=int(runtime_args.chunk_size)).to(runtime.device)
                    require(state.state_fingerprint(model.state_dict()) == parent_manifest['initial_head_sha256'],
                            'Initial original-model identity differs')
                    optimizer = torch.optim.AdamW(model.parameters(), lr=float(runtime_args.lr),
                                                 weight_decay=float(runtime_args.weight_decay))
                    restore_model_optimizer(torch, state, model, optimizer, saved, RESTORED_STEP)
                    mark(output, status, 'reconstruct_data_order', optimizer_state_restored=True)
                    print('DIAGNOSTIC RESTORED checkpoint150: reconstructing raw-batch order1–150', flush=True)
                    iterator = reconstruct_iterator(exp, runtime, parent_manifest, rows, RESTORED_STEP)
                    state.restore_rng(saved['rng'], require_cuda=True)
                    require(state.state_fingerprint(state.capture_rng()) == state.state_fingerprint(saved['rng']), 'RNG restore differs')
                    write(output / 'manifest.json', dict(format=FORMAT, parent_manifest=parent_manifest,
                          runtime_config=vars(runtime_args), protocol_info=runtime.protocol_info,
                          rng_restored=True, optimizer_state_restored=True, data_order_prefix_verified=True,
                          baseline=str(baseline), backward_index_observer=False, anomaly_detection=False))
                    del saved
                    with exp.capture_training_failure(output, status):
                        with (output / 'diagnostic_updates.jsonl').open('x') as journal:
                            for step in range(RESTORED_STEP+1, END_STEP+1):
                                mark(output, status, f'prepare_{step}', current_image=None)
                                epoch, batch = next(iterator)
                                example = runtime.prepare(batch)
                                original._assert_example(example)
                                require(example.image in full_train, 'Diagnostic image outside Train')
                                if step <= len(rows):
                                    require(example.image == rows[step-1]['image'] and epoch == rows[step-1]['epoch'],
                                            'Continuation order mismatch at '+str(step))
                                else:
                                    require(step == END_STEP and example.image == TARGET_IMAGE and epoch == 0,
                                            'Target step158 image differs')
                                mark(output, status, f'update_{step}', current_image=example.image, training_performed=True)
                                expected = baseline_info['rows'][step-RESTORED_STEP-1]
                                require(example.image == expected['image'] and epoch == expected['epoch'],
                                        'Plain/observed sample order differs at '+str(step))
                                row = plain_update(exp, model, optimizer, example, runtime, factory, state, slots,
                                    step=step, output=output, status=status, multiplier=plan['post500_lr_multiplier'])
                                row.update(step=step, epoch=epoch, image=example.image, variant='original')
                                if step <= len(rows):
                                    row['parent_loss_absolute_difference'] = abs(row['original_loss']-rows[step-1]['original_loss'])
                                journal.write(json.dumps(row, allow_nan=False)+'\n')
                                journal.flush()
                                plain_rows.append(row)
                                write(output / 'comparison.json', compare_rows(baseline_info['rows'], plain_rows))
                                mark(output, status, f'update_complete_{step}', last_complete_step=step,
                                     new_complete_updates=step-RESTORED_STEP)
                                print('PLAIN UPDATE '+json.dumps(row), flush=True)
                                del row, example, batch
                    mark(output, status, 'final_integrity')
                    status['runtime_integrity'] = original._checked_verify(runtime)
                    exp.verify_plan(plan, bridge)
                    bindings.legacy_bridge.verify_dependencies(bindings.legacy)
                guard = runtime.guard_report
                require(guard.get('restored') is True and not guard.get('failures') and not guard.get('restore_conflicts'),
                        'PoolGuard final verification failed')
                status['pool_guard'] = guard
            assert_unchanged(fingerprints)
            assert_unchanged(baseline_info['fingerprints'])
        mark(output, status, 'plain_window_complete', diagnostic_complete=True,
             original_failed_runs_unchanged=True, observed_baseline_unchanged=True,
             paired_updates=len(plain_rows), comparison=str(output / 'comparison.json'),
             error=None, elapsed_seconds=time.monotonic()-started)
        print('PLAIN WINDOW COMPLETE:151–158 only; descriptive paired scalars; historical failure unresolved; no automatic continuation', flush=True)
        return status
    except BaseException as error:
        failure = read(output / 'first_failure.json') if (output / 'first_failure.json').is_file() else {}
        status.update(failure_phase=failure.get('phase', status['phase']), phase='failed', diagnostic_complete=False,
                      error=failure.get('error', type(error).__name__+': '+str(error)), elapsed_seconds=time.monotonic()-started)
        if fingerprints:
            try:
                assert_unchanged(fingerprints)
                status['original_failed_runs_unchanged'] = True
            except Exception as integrity_error:
                status['original_failed_runs_unchanged'] = False
                status['integrity_error'] = str(integrity_error)
        if baseline_info:
            try:
                assert_unchanged(baseline_info['fingerprints'])
                status['observed_baseline_unchanged'] = True
            except Exception as integrity_error:
                status['observed_baseline_unchanged'] = False
                status['baseline_integrity_error'] = str(integrity_error)
            try:
                write(output / 'comparison.json', compare_rows(baseline_info['rows'], plain_rows))
            except Exception as comparison_error:
                status['comparison_error'] = str(comparison_error)
        write(output / 'summary.json', status)
        current_traceback = traceback.format_exc()
        primary_traceback = failure.get('traceback') or current_traceback
        (output / 'traceback.txt').write_text(primary_traceback)
        if current_traceback != primary_traceback:
            (output / 'outer_exception_traceback.txt').write_text(current_traceback)
        raise
