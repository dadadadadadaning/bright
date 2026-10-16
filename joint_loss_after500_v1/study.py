#!/usr/bin/env python3
"""Fork completed D checkpoint 500 into matched original/no-background-Lovasz studies.

This is a NEW experiment, not a modification/resume of the old workflow.
The original unmodified runtime supplies frozen features and supports.
The candidate removes only the background Lovasz term with the original denominator.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
FORMAT = 'BRIGHT_JOINT_LOSS_AFTER500_V1'
EXPECTED_SOURCE = 'b84e2867a4117911b8279e2f97ccf9c5fbd021a7dc539f59e6bbb06396546142'
DEFAULT_ROOT = '/root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_V1/run_20261014_202057_full_fEYBZv7D'
DEFAULT_TOOL = '/root/shared-nvme/BRIGHT-master/New/soft_memory_q_rebased_v1_fix2'
LATEST = Path('/root/shared-nvme/BRIGHT-master/New/joint_loss_after500_latest.txt')


@contextmanager
def exclusive_pipeline_lock(path):
    """Prevent a second copy of this study from duplicating an active run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+', encoding='utf-8') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            raise RuntimeError('Another loss study holds the run lock: ' + handle.read().strip()) from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({'pid': os.getpid(), 'tool': str(HERE)}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=str) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def package_fingerprint():
    h = hashlib.sha256()
    for path in sorted(HERE.glob('*.py')):
        h.update(path.name.encode()); h.update(path.read_bytes())
    return h.hexdigest()


def verify_release():
    record = read_json(HERE / 'SHA256SUMS.json')
    for name, expected in record.items():
        path = HERE / name
        if (Path(name).name != name or path.is_symlink()
                or not path.is_file() or digest(path) != expected):
            raise ValueError('Delivery package checksum mismatch: ' + name)
    if not {p.name for p in HERE.glob('*.py')} <= set(record):
        raise ValueError('Unbound Python file in delivery package')
    return digest(HERE / 'SHA256SUMS.json')


def historical_rows(path, train_names):
    rows = []
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('step') != len(rows) + 1 or row.get('image') not in train_names:
                raise ValueError('Historical journal is not contiguous or crosses Train')
            rows.append(row)
    if len(rows) != 5000:
        raise ValueError('Expected the completed original 5000-row journal')
    prefix = rows[:1500]
    if len({r['image'] for r in prefix}) != 1500 or any(r.get('epoch') != 0 for r in prefix):
        raise ValueError('This study requires the original epoch-zero 1..1500 sequence')
    return rows


def validate_static(args):
    root = Path(args.train_root).expanduser().resolve()
    tool = Path(args.tool_dir).expanduser().resolve()
    train = root / 'train'
    required = [train/'summary.json', root/'workflow_state.json', train/'manifest.json',
                train/'checkpoints/checkpoint_000500.pt', train/'checkpoints/checkpoint_000500.pt.json',
                train/'val_step_000500/result.json', train/'train.jsonl', tool/'SHA256SUMS.json']
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Required original inputs are missing; restore their originals, do not fabricate manifests:\n' + '\n'.join(missing))
    summary, workflow = read_json(train / 'summary.json'), read_json(root / 'workflow_state.json')
    for record in (summary, workflow):
        if record.get('complete') is not True or record.get('full_validation_evidence') is not True:
            raise ValueError('Original formal run must be complete; no completion fields are changed')
    if summary.get('smoke') is not False:
        raise ValueError('Source training is a smoke run')
    manifest = read_json(train / 'manifest.json')
    names, val = manifest.get('train_names', []), manifest.get('val_names', [])
    if (len(names) != 2918 or len(set(names)) != 2918 or len(val) != 450
            or len(set(val)) != 450 or set(names) & set(val)):
        raise ValueError('Source must bind disjoint Train2918 and Val450')
    checkpoint = train / 'checkpoints/checkpoint_000500.pt'
    meta = read_json(checkpoint.with_suffix('.pt.json'))
    if (meta.get('step') != 500 or meta.get('mode') != 'train' or meta.get('smoke') is not False
            or meta.get('training_complete') is not False or digest(checkpoint) != meta.get('sha256')
            or meta.get('source_hash') != EXPECTED_SOURCE or manifest.get('source_hash') != EXPECTED_SOURCE):
        raise ValueError('Original step-500 checkpoint / manifest / source binding failed')
    reference = read_json(train / 'val_step_000500/result.json')
    if (reference.get('image_count') != 450 or reference.get('execution_complete') is not True
            or not {'grid_probq', 'd'} <= set(reference.get('metrics', {}))):
        raise ValueError('Original step-500 full Val450 report is missing')
    hashes = read_json(tool / 'SHA256SUMS.json')
    for name, expected in hashes.items():
        path = tool / name
        if Path(name).name != name or path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError('Historical package file is missing/changed: ' + str(path))
    if not {p.name for p in tool.glob('*.py')} <= set(hashes):
        raise ValueError('Unexpected Python file in original tool directory')
    source = hashlib.sha256()
    for path in sorted(tool.glob('*.py')):
        if not path.name.startswith('test_'):
            source.update(path.name.encode()); source.update(path.read_bytes())
    if source.hexdigest() != EXPECTED_SOURCE:
        raise ValueError('Use the exact original fix2 package, unchanged')
    rows = historical_rows(train / 'train.jsonl', set(names))
    fingerprint_paths = [checkpoint, checkpoint.with_suffix('.pt.json'), train / 'manifest.json',
                         train / 'summary.json', root / 'workflow_state.json', train / 'train.jsonl',
                         train / 'val_step_000500/result.json', tool / 'SHA256SUMS.json']
    return dict(root=str(root), tool=str(tool), checkpoint=str(checkpoint), meta=meta,
                manifest=manifest, reference=reference, rows=rows,
                old_files={str(p): digest(p) for p in fingerprint_paths})


def import_original(static):
    execution = static['meta']['protocol_info']['execution_settings']
    blocking = execution.get('cuda_launch_blocking')
    if blocking is None:
        os.environ.pop('CUDA_LAUNCH_BLOCKING', None)
    else:
        os.environ['CUDA_LAUNCH_BLOCKING'] = str(blocking)
    sys.path.insert(0, static['tool'])
    import torch
    import common
    import train
    import run
    from server_runtime import Runtime
    release = run.verify_release()
    if release['code_sha256'] != EXPECTED_SOURCE:
        raise ValueError('Imported the wrong original runtime')
    return torch, common, train, run, Runtime


def load_saved(static, torch):
    from continuation import validate_checkpoint_payload
    saved = torch.load(static['checkpoint'], map_location='cpu', weights_only=False)
    validate_checkpoint_payload(saved, require_cuda=True)
    if (saved.get('source_hash') != EXPECTED_SOURCE
            or saved.get('protocol_info') != static['meta']['protocol_info']
            or saved.get('format') != 'SRAC_SOFT_MEMORY_Q_REBASED_V1'):
        raise ValueError('Checkpoint payload differs from its bound manifest')
    if saved['updates_per_arm'] != {a: 500 for a in ('a', 'b', 'c', 'd')}:
        raise ValueError('Original four heads were not all updated 500 times')
    if not math.isclose(float(saved['config']['lr']), 1e-4, rel_tol=1e-12):
        raise ValueError('This fixed study expects original maximum LR=1e-4')
    return saved


def check_old_files(static):
    for name, expected in static['old_files'].items():
        if digest(name) != expected:
            raise ValueError('An original artifact changed during the new study: ' + name)


def check_inputs_against_checkpoint(runtime, saved):
    ledger = saved.get('input_file_sha256', {})
    if not ledger:
        raise ValueError('Source checkpoint has no input ledger')
    for path, actual in runtime.input_ledger.items():
        if ledger.get(str(path)) != actual:
            raise ValueError('Runtime input differs from checkpoint-500 provenance: ' + str(path))


def validate_example(example, runtime, original_train):
    original_train._assert_example(example)
    if example.diagnostics.get('query_GT_used_for_supports') is not False:
        raise ValueError('Query ground-truth separation is unverified')
    selected = example.supports['d']['support_ids']
    chosen = selected[selected >= 0]
    eid, iid = example.diagnostics['event_id'], example.diagnostics['image_id']
    for key, query in (('events', eid), ('images', iid)):
        if query >= 0 and bool((runtime.bank[key][chosen] == query).any()):
            raise ValueError('Memory includes the query image or event')
    return original_train._coverage(example)


def save_checkpoint(folder, heads, optimizers, step, saved, static, runtime, torch, common):
    from continuation import capture_rng
    path = folder / ('checkpoint_%06d.pt' % step)
    payload = dict(format=FORMAT, step=step, inherited_updates=500, new_updates_per_branch=step-500,
                   primary_endpoint_step=1500, suffix_training_complete=step==1500,
                   historical_checkpoint_sha256=static['meta']['sha256'], historical_source_hash=EXPECTED_SOURCE,
                   study_code_sha256=package_fingerprint(), protocol_info=runtime.protocol_info,
                   config=saved['config'], heads={k:v.state_dict() for k,v in heads.items()},
                   optimizers={k:v.state_dict() for k,v in optimizers.items()}, rng=capture_rng(),
                   input_file_sha256={str(p):h for p,h in runtime.input_ledger.items()},
                   test_accessed=False, native_historical_resume=False,
                   schedule_horizon=5000, branch_step=500, lr_multipliers={'d_control':1., 'd_no_bg_lovasz':1.},
                   objective_by_branch={'d_control':'original_nll_plus_present_lovasz',
                                        'd_no_bg_lovasz':'original_nll_plus_present_foreground_lovasz_original_denominator'})
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary); os.replace(temporary, path)
    record = dict(format=FORMAT, step=step, sha256=digest(path),
                  head_hashes={k:common.module_hash(v) for k,v in heads.items()})
    write_json(path.with_suffix('.pt.json'), record)
    return path, record


def worker(args):
    verify_release()
    static = validate_static(args)
    torch, common, original_train, original_run, Runtime = import_original(static)
    from continuation import (BINDINGS, fork_heads_and_optimizers, restore_rng,
                              raw_batch_identity, train_matched_step, lr_at_step)
    from evaluation import evaluate, compare_initial_to_old, metriccontrast, print_table
    saved = load_saved(static, torch)
    smoke = args.mode == 'smoke'
    last_step = 520 if smoke else 1500
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'checkpoints').mkdir()
    config = copy.deepcopy(saved['config'])
    config.update(command='train', mode='train', smoke=False, output=str(output),
                  baseline_policy='independent', baseline_root=None, baseline_evidence=None,
                  allow_smoke_values=False)
    runtime_args = SimpleNamespace(**config)
    state = dict(format=FORMAT, complete=False, execution_complete=False, training_complete=False,
                 full_validation_evidence=False, final_integrity_pending=True,
                 phase='loading', smoke=smoke, pid=os.getpid(), test_accessed=False,
                 original_root=static['root'], source_checkpoint=static['checkpoint'],
                 source_checkpoint_sha256=static['meta']['sha256'], study_code_sha256=package_fingerprint(),
                 original_source_hash=EXPECTED_SOURCE, inherited_updates_per_branch=500,
                 suffix_updates_per_branch={k:0 for k in BINDINGS}, endpoint_step=last_step,
                 schedule_horizon=5000, branch_step=500, bindings=BINDINGS, evaluations={},
                 claims=dict(new_controlled_fork=True, exact_historical_trajectory=False,
                             independent_seed_replication=False, all_Train2918_visited=False))
    state['experiment_definition'] = dict(
        changed='Only the candidate omits the background Lovasz term; original present-class denominator retained',
        retained=['both soft slots', 'parent region', 'pixels', 'Memory', 'four output classes',
                  'all valid pixels including background', 'original NLL', 'original LR schedule'],
        primary_comparison='d_no_bg_lovasz minus d_control at step1500',
        lower_candidate_loss_is_not_evidence_of_improvement=True)
    write_json(output / 'summary.json', state)
    started = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('Real BRIGHT smoke/training requires CUDA; CPU tests are separate')
        original_run.check_inputs(runtime_args)
        with Runtime(runtime_args, output) as runtime:
            original_train._formal_runtime(runtime, False)
            if (runtime.protocol_info != saved['protocol_info']
                    or runtime.train_names != static['manifest']['train_names']
                    or runtime.val_names != static['manifest']['val_names']):
                raise ValueError('Current frozen source, bank, runtime or data split differs from checkpoint500')
            check_inputs_against_checkpoint(runtime, saved)
            original_train._checked_verify(runtime)
            heads, optimizers, bindings = fork_heads_and_optimizers(saved, runtime_args, runtime.device)
            for name, (base, _) in BINDINGS.items():
                if common.module_hash(heads[name]) != static['meta']['head_hashes'][base]:
                    raise ValueError('Restored head hash differs from original manifest: ' + name)
            # Verify the copied schedule against the actual inherited function.
            for step in (1, 500, 501, 1000, 1500, 5000):
                if lr_at_step(step) != original_train._lr(step, 5000, float(runtime_args.lr)):
                    raise ValueError('Original schedule differs from the declared study')
            state['initial_forks'] = bindings
            # Original loader has a private seeded generator and inference-only transforms.
            # Re-read and skip 500 CPU batches: no frontend and no optimization here.
            state['phase'] = 'reconstruct_loader'; write_json(output / 'summary.json', state)
            iterator = iter(runtime.batches(runtime.train_names, train=True, seed=int(runtime_args.seed)))
            for index in range(500):
                batch = next(iterator)
                if raw_batch_identity(batch) != static['rows'][index]['image']:
                    raise ValueError('Historical Train order mismatch at step %d' % (index+1))
                if (index+1) % 100 == 0:
                    print('Loader position verified %d/500; no frontend/training' % (index+1), flush=True)
            del batch
            check_inputs_against_checkpoint(runtime, saved)
            restore_rng(saved['rng'], require_cuda=True)
            val_names = runtime.val_names[:2] if smoke else runtime.val_names
            state['phase'] = 'initial_val'; write_json(output / 'summary.json', state)
            initial = evaluate(runtime, heads, val_names, output/'val_step_000500', step=500,
                               split='val', seed=int(runtime_args.seed), original_train=original_train)
            if not smoke:
                reproduction = compare_initial_to_old(initial, static['reference'])
                write_json(output/'initial_reproduction.json', reproduction)
                if reproduction.get('passed') is not True:
                    raise ValueError('Initial Val450 fails checkpoint500 replay; see initial_reproduction.json')
                state['initial_reproduction'] = reproduction
            state['evaluations']['500'] = str(output/'val_step_000500/result.json')
            state['initial_metrics'] = initial['metrics']
            print_table(initial)
            fixed_names = [row['image'] for row in static['rows'][:(2 if smoke else 450)]]
            state['phase'] = 'fixed_train_initial'; write_json(output/'summary.json', state)
            fixed_initial = evaluate(runtime, heads, fixed_names, output/'fixed_train_initial', step=500,
                                     split='fixed_train', seed=int(runtime_args.seed), original_train=original_train)
            state['fixed_train_initial_report'] = str(output/'fixed_train_initial/result.json')
            print_table(fixed_initial)
            # Forking and initial evaluation must not advance the training RNG.
            restore_rng(saved['rng'], require_cuda=True)
            state['phase'] = 'training'; write_json(output/'summary.json', state)
            seen, support_counts = [], {k:0 for k in BINDINGS}
            gradient_counts = {k:0 for k in BINDINGS}
            clipped_counts = {k:0 for k in BINDINGS}
            with (output/'train.jsonl').open('x', encoding='utf-8') as journal:
                for step in range(501, last_step+1):
                    batch = next(iterator)
                    expected = static['rows'][step-1]
                    if raw_batch_identity(batch) != expected['image']:
                        raise ValueError('Train loader differs from historical order at step %d' % step)
                    example = runtime.prepare(batch)
                    if example.image != expected['image'] or example.event != expected['event']:
                        raise ValueError('Prepared image/event differs from the historical journal')
                    coverage = validate_example(example, runtime, original_train)
                    rows = train_matched_step(heads, optimizers, example, runtime, step, float(runtime_args.lr))
                    for name, (base, _) in BINDINGS.items():
                        support_counts[name] += coverage[base]['supported_slot_queries']
                        gradient_counts[name] += int(rows[name]['reader_grad_abs_sum'] > 0)
                        clipped_counts[name] += int(rows[name]['grad_norm'] > 1.0)
                        state['suffix_updates_per_branch'][name] = step-500
                    seen.append(example.image)
                    row = dict(step=step, suffix_step=step-500, epoch=0, image=example.image,
                               event=example.event, heads=rows, coverage=coverage,
                               elapsed_seconds=time.monotonic()-started)
                    journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); journal.flush()
                    if step % 20 == 0 or step == 501:
                        print('LOSS STUDY '+json.dumps(row, ensure_ascii=False), flush=True)
                        state['last_step'] = step; write_json(output/'summary.json', state)
                    del example, batch
                    if step == 1000 or step == last_step:
                        path, record = save_checkpoint(output/'checkpoints', heads, optimizers,
                                                       step, saved, static, runtime, torch, common)
                        # Evaluate serialized checkpoint contents, not only live weights.
                        reopened = torch.load(path, map_location='cpu', weights_only=False)
                        if digest(path) != record['sha256']:
                            raise ValueError('New checkpoint changed before validation')
                        for name in BINDINGS:
                            heads[name].load_state_dict(reopened['heads'][name], strict=True)
                            if common.module_hash(heads[name]) != record['head_hashes'][name]:
                                raise ValueError('Serialized head differs before validation')
                        del reopened
                        state['phase'] = 'val_step_%06d' % step; write_json(output/'summary.json', state)
                        result = evaluate(runtime, heads, val_names, output/('val_step_%06d'%step),
                                          step=step, split='val', seed=int(runtime_args.seed),
                                          original_train=original_train, save_predictions=(step==last_step))
                        state['evaluations'][str(step)] = str(output/('val_step_%06d'%step)/'result.json')
                        state['last_checkpoint'] = str(path)
                        state['comparisons_pp'] = metriccontrast(result)
                        state['change_from_500_pp'] = {
                            name: 100*(result['metrics'][name]['foreground_miou']-
                                       initial['metrics'][name]['foreground_miou'])
                            for name in ('grid_probq', *BINDINGS)}
                        print_table(result)
                        state['phase'] = 'training'; write_json(output/'summary.json', state)
            state['training_complete'] = True
            if any(support_counts[k] <= 0 or gradient_counts[k] <= 0 for k in BINDINGS):
                raise ValueError('A branch has no legal support or no Reader gradient')
            if len(seen) != last_step-500 or len(set(seen)) != len(seen):
                raise ValueError('Suffix query coverage is not the intended epoch-zero sequence')
            state['phase'] = 'fixed_train_endpoint'; write_json(output/'summary.json', state)
            fixed = evaluate(runtime, heads, fixed_names, output/'fixed_train_endpoint', step=last_step,
                             split='fixed_train', seed=int(runtime_args.seed), original_train=original_train)
            print_table(fixed)
            state['fixed_train_report'] = str(output/'fixed_train_endpoint/result.json')
            if (fixed['names_sha256'] != fixed_initial['names_sha256']
                    or fixed['order_sha256'] != fixed_initial['order_sha256']):
                raise ValueError('Fixed Train initial/endpoint identities or order changed')
            state['fixed_train_change_from_500_pp'] = {
                name: 100*(fixed['metrics'][name]['foreground_miou']-
                           fixed_initial['metrics'][name]['foreground_miou'])
                for name in ('grid_probq', *BINDINGS)}
            state['fixed_train_scope'] = 'first 450 previously seen Train queries; not all Train2918' if not smoke else '2-image smoke'
            state['phase'] = 'final_integrity'; write_json(output/'summary.json', state)
            check_inputs_against_checkpoint(runtime, saved)
            integrity = original_train._checked_verify(runtime)
            check_old_files(static)
            if package_fingerprint() != state['study_code_sha256'] or common.source_hash() != EXPECTED_SOURCE:
                raise ValueError('New driver or original runtime code changed during the study')
            state.update(integrity=integrity, supported_slot_queries=support_counts,
                         reader_positive_gradient_updates=gradient_counts,
                         gradient_clipped_updates=clipped_counts,
                         suffix_unique_train_images=len(set(seen)), combined_unique_train_images=last_step,
                         train_order_verified_through_step=last_step)
            final_head_hashes = {k:common.module_hash(v) for k,v in heads.items()}
            if final_head_hashes != record['head_hashes']:
                raise ValueError('Validation changed final head parameters/buffers')
        state['guard_report'] = original_train._restored_guard(runtime)
        for path in [*state['evaluations'].values(), state['fixed_train_initial_report'], state['fixed_train_report']]:
            report = read_json(path)
            report.update(complete=True, smoke=smoke, phase='complete_smoke' if smoke else 'complete',
                          final_integrity_pending=False, final_integrity=integrity,
                          primary_endpoint=(not smoke and report['step']==1500 and report['split']=='val'),
                          full_validation_evidence=(not smoke and report['split']=='val'),
                          guard_report=state['guard_report'])
            write_json(path, report)
        state.update(complete=True, execution_complete=True, full_validation_evidence=not smoke,
                     final_integrity_pending=False, phase='complete_smoke' if smoke else 'complete',
                     elapsed_seconds=time.monotonic()-started)
        write_json(output/'input_fingerprints.json', {str(p):h for p,h in runtime.input_ledger.items()})
        write_json(output/'summary.json', state)
        if smoke:
            print('LOSS SMOKE COMPLETE. 20 new updates only; not formal Val450 evidence. No Test.', flush=True)
        else:
            print('LOSS TRAIN/VAL450 COMPLETE. Step1500 endpoint; no Test; no source/bank changes.', flush=True)
        return state
    except BaseException as error:
        state.update(complete=False, full_validation_evidence=False, execution_complete=False,
                     error=type(error).__name__+': '+str(error))
        (output/'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        write_json(output/'summary.json', state)
        raise


def pipeline(args):
    root = Path(args.output).expanduser().resolve()
    with exclusive_pipeline_lock(root.parent/'active_pipeline.lock'):
        return _pipeline(args)


def _pipeline(args):
    verify_release(); static = validate_static(args)
    root = Path(args.output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = dict(format=FORMAT, complete=False, execution_complete=False, test_accessed=False,
                 full_validation_evidence=False, training_complete=False,
                 current_stage='preflight', stages={}, pid=os.getpid(), arguments=vars(args),
                 source_checkpoint_sha256=static['meta']['sha256'])
    write_json(root/'workflow_state.json', state)
    (root/'logs').mkdir()
    stages = [('cpu_contracts', [sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE), '-p', 'test_*.py', '-q'])]
    for mode in (('smoke',) if args.command=='check' else ('smoke','train')):
        stages.append((mode, [sys.executable, '-u', str(HERE/'study.py'), 'worker', '--mode', mode,
                             '--train-root', args.train_root, '--tool-dir', args.tool_dir,
                             '--output', str(root/mode)]))
    try:
        for name, command in stages:
            log = root/'logs'/(name+'.log')
            state['current_stage'] = name
            state['stages'][name] = dict(status='running', command=command, log=str(log))
            write_json(root/'workflow_state.json', state)
            print('START '+name+' LOG='+str(log), flush=True)
            with log.open('x', encoding='utf-8') as handle:
                child_env = os.environ.copy()
                if name == 'cpu_contracts':
                    child_env['CUDA_VISIBLE_DEVICES'] = ''
                    # AST parity checks read the existing server checkout without
                    # importing its CUDA/model dependencies.
                    source_config = static['manifest'].get('arguments', {})
                    child_env['BRIGHT_REPO'] = source_config.get('repo', '/root/shared-nvme/BRIGHT-master')
                child = subprocess.Popen(command, cwd=HERE, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1, env=child_env)
                state['stages'][name]['pid'] = child.pid; write_json(root/'workflow_state.json', state)
                for line in child.stdout:
                    handle.write(line); handle.flush(); print(line, end='', flush=True)
                code = child.wait()
            state['stages'][name].update(returncode=code, status='complete' if code==0 else 'failed')
            if code != 0:
                raise RuntimeError(name+' failed; later stages stopped; see '+str(log))
            if name != 'cpu_contracts':
                report = read_json(root/name/'summary.json')
                formal = name == 'train'
                if (report.get('complete') is not True
                        or report.get('execution_complete') is not True
                        or report.get('training_complete') is not True
                        or report.get('final_integrity_pending') is not False
                        or report.get('smoke') is not (not formal)
                        or report.get('endpoint_step') != (1500 if formal else 520)
                        or report.get('full_validation_evidence') is not formal):
                    raise ValueError('Child exit did not have complete evidence: '+name)
            write_json(root/'workflow_state.json', state)
        state.update(complete=True, execution_complete=True, current_stage='complete',
                     training_complete=args.command=='full',
                     full_validation_evidence=args.command=='full')
        write_json(root/'workflow_state.json', state)
        print('OUTPUT_ROOT='+str(root), flush=True)
        print_report(root)
    except BaseException as error:
        state.update(complete=False, execution_complete=False, error=type(error).__name__+': '+str(error))
        write_json(root/'workflow_state.json', state)
        (root/'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def print_report(root):
    root = Path(root).expanduser().resolve()
    print('结果目录：', root)
    wf = root/'workflow_state.json'
    if wf.is_file():
        state = read_json(wf)
        print('阶段：', state.get('current_stage'), '完成：', state.get('complete'), '错误：', state.get('error'))
    train = root/'train' if (root/'train').is_dir() else root/'smoke' if (root/'smoke').is_dir() else root
    path = train/'summary.json'
    if not path.is_file():
        print('尚无训练子阶段报告；查看 logs/。'); return
    summary = read_json(path)
    print('子阶段：', summary.get('phase'), '新增更新：', summary.get('suffix_updates_per_branch'),
          '完整证据：', summary.get('full_validation_evidence'), '错误：', summary.get('error'))
    names = ('grid_probq','d_control','d_no_bg_lovasz')
    print('step path fg_mIoU% Damaged_IoU% Damaged_precision% Damaged_recall% Damaged_to_BG')
    for step, report_path in sorted(summary.get('evaluations', {}).items(), key=lambda x:int(x[0])):
        if not Path(report_path).is_file():
            print(step, '报告尚不存在：', report_path)
            continue
        report = read_json(report_path)
        for name in names:
            m = report['metrics'][name]
            vals = [m['foreground_miou'],m['iou'][2],m['precision'][2],m['recall'][2]]
            print(step, name, *('NA' if v is None else '%.4f'%(100*v) for v in vals), m['confusion'][2][0])
        print('两支都按共同原目标和候选目标评分；列：原总loss 候选总loss NLL 原Lovasz 候选Lovasz')
        for name in names:
            m = report['metrics'][name]
            keys = ('mean_image_loss','mean_image_candidate_loss','mean_image_nll',
                    'mean_image_lovasz','mean_image_candidate_lovasz')
            print(step, name, *('NA' if m.get(k) is None else '%.6f'%m[k] for k in keys))
            print('  四类NLL贡献：', m.get('mean_image_class_nll_contribution'))
            print('  四类Lovasz贡献：', m.get('mean_image_class_lovasz_contribution'))
            print('  Intact→Damaged / Intact→Destroyed：', m['confusion'][1][2], m['confusion'][1][3])
    print('最新比较（百分点）：', json.dumps(summary.get('comparisons_pp', {}), ensure_ascii=False))
    print('相对各自500步的变化（百分点）：', json.dumps(summary.get('change_from_500_pp', {}), ensure_ascii=False))
    print('固定Train相对500步的变化（百分点）：', json.dumps(summary.get('fixed_train_change_from_500_pp', {}), ensure_ascii=False))
    fixed_path = summary.get('fixed_train_report')
    if fixed_path and Path(fixed_path).is_file():
        fixed = read_json(fixed_path)
        print('固定 Train 子集：', fixed['image_count'], '张；并非完整 Train2918。')
        print('path fg_mIoU% Damaged_IoU% mean_loss mean_NLL mean_Lovasz')
        for name in names:
            m = fixed['metrics'][name]
            print(name, '%.4f'%(100*m['foreground_miou']),
                  'NA' if m['iou'][2] is None else '%.4f'%(100*m['iou'][2]),
                  *(('%.6f'%m[k]) for k in ('mean_image_loss','mean_image_nll','mean_image_lovasz')))
    print('主比较为同次1500步 no_bg_Lovasz−control；两组学习率相同，两个槽和Memory均保留。无Test；单种子探索。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan','check','full','worker','report'))
    parser.add_argument('--train-root', default=DEFAULT_ROOT)
    parser.add_argument('--tool-dir', default=DEFAULT_TOOL)
    parser.add_argument('--output')
    parser.add_argument('--root')
    parser.add_argument('--mode', choices=('smoke','train'), default='train')
    args = parser.parse_args()
    if args.command=='report':
        root = args.root or (LATEST.read_text().strip() if LATEST.is_file() else None)
        if not root:
            parser.error('Pass --root or launch start_server.sh first')
        return print_report(root)
    if args.command=='plan':
        verify_release(); static=validate_static(args)
        print(json.dumps(dict(format=FORMAT, file_checks_passed=True, CUDA_checked=False,
                              source_checkpoint=static['checkpoint'], old_files_unchanged=True,
                              inherited_step=500, endpoint=1500, horizon=5000,
                              new_branches=['D 原NLL＋原Lovasz','D 原NLL＋去背景项Lovasz（保留原分母）'],
                              same_original_learning_rate=True, two_slots_retained=True, memory_retained=True,
                              bank_rebuild=False, test_accessed=False), ensure_ascii=False, indent=2))
        return
    if not args.output:
        parser.error('--output is required; use start_server.sh for automatic new paths')
    return worker(args) if args.command=='worker' else pipeline(args)


if __name__=='__main__':
    main()
