#!/usr/bin/env python3
"""Independent, fail-closed recovery of the saved joint Memory experiment.

The parent imports no torch, never edits the original tool, and gives the two
declared SAR runs priority before it starts isolated CUDA workers.
"""
import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
FAILED = '/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_MEMORY_JOINT_SYNC_V1/run_20261013_073724_212223_3841256'
TOOL = '/root/shared-nvme/BRIGHT-master/New/joint_train_ZAtNI2Wi/region_memory_joint_train_v1'
AFTER = (
    '/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_Q_MATCHED_V1/run_20261013_082551_3863093',
    '/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_SOFT_CROSS_V1/run_20261013_092358_3887540',
)
ASSOCIATION_SHA = 'fb73fe6472c5cca3c1756ec127383243bc79c762b80bf556fa3690511f825463'
FORMAT = 'SRAC_JOINT_POOL_RECOVERY_V1'
RESULT_BASE = Path('/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_POOL_RECOVERY_V1')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def source_hash(folder):
    result = hashlib.sha256()
    for path in sorted(Path(folder).glob('*.py')):
        if not path.name.startswith('test_'):
            result.update(path.name.encode())
            result.update(path.read_bytes())
    return result.hexdigest()


def verify_release(folder):
    folder = Path(folder)
    hashes = read(folder / 'SHA256SUMS.json')
    if not hashes:
        raise ValueError('Empty release manifest')
    for name, expected in hashes.items():
        if Path(name).name != name or digest(folder / name) != expected:
            raise ValueError('Release checksum mismatch: ' + str(folder / name))
    if not {p.name for p in folder.glob('*.py')} <= set(hashes):
        raise ValueError('Unbound Python file in release: ' + str(folder))
    return digest(folder / 'SHA256SUMS.json')


def beneath(path, folder):
    path, folder = Path(path).resolve(), Path(folder).resolve()
    if folder not in path.parents:
        raise ValueError('Evidence file outside selected run: ' + str(path))
    return path


def inspect_failed(root, tool):
    root, tool = Path(root).resolve(), Path(tool).resolve()
    release = verify_release(tool)
    manifest = read(root / 'train/manifest.json')
    summary = read(root / 'train/summary.json')
    failure = read(root / 'train/pool_debug/failure.json')
    guard = read(root / 'train/pool_debug/guard.json')
    pool_manifest = read(root / 'train/pool_debug/manifest.json')
    if manifest.get('source_hash') != source_hash(tool):
        raise ValueError('Original training source changed')
    config = manifest['arguments']
    if config.get('mode') != 'train' or config.get('steps') != 5000:
        raise ValueError('Requires the original matched 5000-update formal run')
    if summary.get('complete') is True or not summary.get('error'):
        raise ValueError('Select the failed training run, not a completed/live run')
    updates = summary['updates_per_arm']
    if set(updates) != {'r0', 'rt', 'rf'} or len(set(updates.values())) != 1:
        raise ValueError('Three arms are not at the same saved update')
    if guard.get('backend') != 'native' or failure.get('phase') != 'pool_kernel':
        raise ValueError('This repair only targets the captured native pool kernel failure')
    if pool_manifest.get('original_sha256') != ASSOCIATION_SHA:
        raise ValueError('Captured operator differs from the audited arithmetic')
    association = Path(pool_manifest['original_source']).resolve()
    if digest(association) != ASSOCIATION_SHA:
        raise ValueError('Original pool source changed after failure')
    guard_source = Path(config['repo']) / 'bda_benchmark/experiments/damageformer_srac_evidence_followup_v1/native_pool.py'
    if digest(guard_source) != pool_manifest.get('audit_source_sha256'):
        raise ValueError('Original pre-kernel guard source changed after failure')
    snapshot = beneath(root / 'train/pool_debug' / Path(failure['input']).name, root)
    checkpoint = beneath(summary['last_checkpoint'], root / 'train/checkpoints')
    sidecar = read(checkpoint.with_suffix(checkpoint.suffix + '.json'))
    step = sidecar.get('step')
    if type(step) is not int or not 0 < step < 5000 or set(updates.values()) != {step}:
        raise ValueError('Checkpoint step does not match all three failed-run updates')
    if digest(checkpoint) != sidecar.get('sha256'):
        raise ValueError('Checkpoint checksum mismatch')
    if (sidecar.get('source_hash') != manifest['source_hash']
            or sidecar.get('protocol_info') != manifest['protocol_info']):
        raise ValueError('Checkpoint sidecar/source binding mismatch')
    history = root / 'train/train.jsonl'
    rows = []
    with history.open(encoding='utf-8') as stream:
        for index, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get('step') != index:
                raise ValueError('Training journal is not an exact contiguous prefix')
            rows.append(row)
    if len(rows) != step:
        raise ValueError('Training journal contains missing/unsaved updates; manual audit needed')
    bound_paths = [root / 'train/manifest.json', root / 'train/summary.json',
                   root / 'train/pool_debug/failure.json', root / 'train/pool_debug/guard.json',
                   root / 'train/pool_debug/manifest.json', checkpoint,
                   checkpoint.with_suffix(checkpoint.suffix + '.json'), snapshot, history,
                   association, guard_source, tool / 'SHA256SUMS.json']
    return dict(format=FORMAT, failed_run=str(root), tool=str(tool), repo=config['repo'],
                arguments=config, protocol_info=manifest['protocol_info'],
                initial_head_hashes=manifest['initial_head_hashes'],
                original_source_hash=manifest['source_hash'], original_release_sha256=release,
                checkpoint=str(checkpoint), checkpoint_step=step, sidecar=sidecar,
                snapshot=str(snapshot), history=str(history), failed_image=failure.get('image'),
                failed_call=failure.get('call'),
                runtime=manifest['protocol_info']['source_runtime_settings'],
                preserved_sha256={str(p): digest(p) for p in bound_paths},
                historical_reproduction_claim=False)


def check_preserved(evidence):
    for name, expected in evidence['preserved_sha256'].items():
        if digest(name) != expected:
            raise ValueError('Preserved source/evidence changed: ' + name)


def terminal_source(root):
    path = Path(root) / 'workflow_state.json'
    if not path.is_file():
        return False, 'state_missing'
    state = read(path)
    if state.get('complete') is True and state.get('failure'):
        raise ValueError('Inconsistent complete and failure SAR state: ' + str(root))
    if state.get('failure'):
        stamp = state.get('finished_at')
        if type(stamp) not in (float, int) or not math.isfinite(stamp) or stamp <= 0:
            raise ValueError('SAR failure lacks a terminal timestamp: ' + str(root))
        return True, 'failed'
    if state.get('complete') is True:
        if state.get('full_validation_evidence') is not True:
            raise ValueError('SAR completion without full evidence: ' + str(root))
        stamp = state.get('finished_at')
        if type(stamp) not in (float, int) or not math.isfinite(stamp) or stamp <= 0:
            raise ValueError('SAR completion lacks a terminal timestamp: ' + str(root))
        return True, 'complete'
    return False, state.get('current_stage') or 'pending'


def wait_sources(roots, timeout=0, pause=time.sleep, now=time.monotonic, emit=print):
    if len(roots) != 2:
        raise ValueError('Specify exactly the SAR-Q and SAR-soft result roots, in that order')
    start = now()
    while True:
        for root, expected in zip(roots, ('BRIGHT_SAR_Q_MATCHED_V1', 'SAR_SOFT_CROSS_PIPELINE_V1')):
            state_path = Path(root) / 'workflow_state.json'
            if state_path.is_file() and read(state_path).get('format') != expected:
                raise ValueError('Unexpected declared SAR format: ' + str(root))
        soft_path = Path(roots[1]) / 'workflow_state.json'
        if soft_path.is_file() and Path(read(soft_path).get('source_root', '')).resolve() != Path(roots[0]).resolve():
            raise ValueError('Declared SAR-soft task is bound to a different SAR-Q run')
        observations = [(str(root), *terminal_source(root)) for root in roots]
        emit('WAITING_DECLARED_SAR ' + json.dumps(observations, ensure_ascii=False), flush=True)
        if all(terminal for _, terminal, _ in observations):
            return observations
        if timeout and now() - start >= timeout:
            raise TimeoutError('Declared SAR runs have not terminated; Memory recovery not started')
        pause(30)


def child_environment(runtime):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_LAUNCH_BLOCKING='1')
    source_environment = runtime.get('environment', {})
    if not isinstance(source_environment, dict):
        raise TypeError('Invalid saved runtime environment')
    for key in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_TF32_OVERRIDE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        if key in source_environment:
            value = source_environment[key]
            if value is None:
                env.pop(key, None)
            elif isinstance(value, str):
                env[key] = value
            else:
                raise TypeError('Invalid saved environment value: ' + key)
    workspace = runtime.get('cublas_workspace_config')
    if workspace is None:
        env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    elif isinstance(workspace, str):
        env['CUBLAS_WORKSPACE_CONFIG'] = workspace
    else:
        raise TypeError('Invalid saved CUBLAS configuration')
    return env


def run_child(command, log, env):
    with Path(log).open('x', encoding='utf-8') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=HERE)
    return result.returncode


@contextmanager
def recovery_lock(base=RESULT_BASE):
    """Prevent duplicate recovery launchers; does not reserve the GPU globally."""
    import fcntl
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    with (base / '.recovery.lock').open('a+', encoding='utf-8') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another Memory recovery launcher holds the lock; use report, do not launch twice')
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()) + '\n')
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def pipeline(args):
    verify_release(HERE)
    evidence = inspect_failed(args.failed_run, args.tool)
    if args.stage == 'inspect':
        print(json.dumps({k: evidence[k] for k in ('failed_run', 'checkpoint', 'checkpoint_step',
                         'failed_image', 'snapshot', 'runtime')}, ensure_ascii=False, indent=2))
        return
    root = Path(args.output).resolve() if args.output else Path(
        str(RESULT_BASE)) / (
            'run_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '_' + str(os.getpid()))
    if root.exists() or root == HERE or HERE in root.parents:
        raise ValueError('A new recovery result directory is required')
    for forbidden in (Path(evidence['failed_run']), Path(evidence['tool']), Path(evidence['repo'])):
        if root == forbidden or forbidden in root.parents:
            raise ValueError('Recovery results must be outside preserved run/tool/repository')
    root.mkdir(parents=True)
    (root / 'logs').mkdir()
    evidence.update(repair_source_hash=source_hash(HERE), repair_release_sha256=digest(HERE / 'SHA256SUMS.json'))
    write(root / 'evidence.json', evidence)
    write(root / 'runtime.json', evidence['runtime'])
    state = dict(format=FORMAT, complete=False, full_validation_evidence=False,
                 exact_historical_resume_claim=False, current_stage='cpu_replay', stages={},
                 prefix_updates=evidence['checkpoint_step'], target_updates=5000,
                 failure=None, training_started=False, test_accessed=False)
    write(root / 'workflow_state.json', state)
    print('OUTPUT_ROOT=' + str(root), flush=True)
    env = child_environment(evidence['runtime'])

    def stage(name, command, output, allow_failure=False):
        state['current_stage'] = name
        log = root / 'logs' / (name + '.log')
        row = dict(command=command, log=str(log), status='running')
        state['stages'][name] = row
        write(root / 'workflow_state.json', state)
        print('START ' + name + ' log=' + str(log), flush=True)
        code = run_child(command, log, env)
        record = read(output / 'summary.json') if (output / 'summary.json').is_file() else {}
        row.update(returncode=code, status='complete' if code == 0 else 'failed',
                   result=str(output), gates_passed=record.get('gates_passed'))
        if allow_failure:
            row['diagnostic_only'] = True
            row['required_repair_gate'] = False
        write(root / 'workflow_state.json', state)
        if not allow_failure and (code or record.get('gates_passed') is not True):
            raise RuntimeError('Stage failed: ' + name + '; read ' + str(log))
        return record

    try:
        cpu = root / 'replay_cpu'
        cpu_record = stage('replay_cpu', [sys.executable, '-u', str(HERE / 'replay_worker.py'),
              '--input', evidence['snapshot'], '--output', str(cpu), '--backend', 'cpu', '--repeats', '1'], cpu)
        if (cpu_record.get('image') != evidence['failed_image']
                or cpu_record.get('call') != evidence['failed_call']
                or cpu_record.get('source_layout_recorded') is not True):
            raise ValueError('Saved input does not match the failure or lacks original layout metadata')
        if args.stage == 'cpu':
            check_preserved(evidence)
            state.update(complete=True, current_stage='cpu_complete', training_started=False)
            write(root / 'workflow_state.json', state)
            print('CPU INPUT CHECK COMPLETE. No CUDA worker or training started.', flush=True)
            return
        state['current_stage'] = 'waiting_declared_sar'
        write(root / 'workflow_state.json', state)
        state['sar_terminal'] = wait_sources(args.after_sar, args.wait_timeout)
        state['current_stage'] = 'waiting_gpu'
        write(root / 'workflow_state.json', state)
        from gpu_queue import wait_for_idle
        device = evidence['arguments']['device']
        if not device.startswith('cuda:'):
            raise ValueError('The saved formal device must be explicit cuda:N')
        previous_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        try:
            if 'CUDA_VISIBLE_DEVICES' in env:
                os.environ['CUDA_VISIBLE_DEVICES'] = env['CUDA_VISIBLE_DEVICES']
            else:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            state['gpu_idle_observation'] = wait_for_idle(int(device[5:]), log=lambda x: print(x, flush=True))
        finally:
            if previous_visible is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = previous_visible
        if not all(terminal_source(path)[0] for path in args.after_sar):
            raise ValueError('Declared SAR state became nonterminal; recovery stopped')
        check_preserved(evidence)
        for backend in ('native', 'contiguous'):
            out = root / ('replay_' + backend)
            record = stage('replay_' + backend, [sys.executable, '-u', str(HERE / 'replay_worker.py'),
                           '--input', evidence['snapshot'], '--output', str(out), '--backend', backend,
                           '--device', device, '--runtime-json', str(root / 'runtime.json'),
                           '--repeats', '3'], out, allow_failure=backend == 'native')
            if backend == 'contiguous' and record.get('gates_passed') is not True:
                raise RuntimeError('Candidate index-layout repair has not passed; training stopped')
        state['current_stage'] = 'checkpoint_reval_and_continuation'
        state['training_worker_started'] = True
        state['training_started'] = None  # The child first revalidates; report derives actual updates.
        write(root / 'workflow_state.json', state)
        out = root / 'train'
        command = [sys.executable, '-u', str(HERE / 'server_worker.py'),
                   '--evidence', str(root / 'evidence.json'), '--output', str(out)]
        code = run_child(command, root / 'logs/train.log', env)
        record = read(out / 'summary.json') if (out / 'summary.json').is_file() else {}
        state['stages']['continuation'] = dict(returncode=code, result=str(out),
                                            log=str(root / 'logs/train.log'))
        if code or record.get('complete') is not True or record.get('full_validation_evidence') is not True:
            raise RuntimeError('Repaired checkpoint evaluation/continuation failed; preserve new checkpoints and read train.log')
        check_preserved(evidence)
        if source_hash(HERE) != evidence['repair_source_hash']:
            raise ValueError('Recovery code changed during execution')
        state.update(complete=True, current_stage='complete', full_validation_evidence=True,
                     training_started=True,
                     updates_per_arm=record['updates_per_arm'], prefix_operator='original_native',
                     continuation_operator='native_contiguous_indices_candidate')
        write(root / 'workflow_state.json', state)
        print('REPAIRED CONTINUATION COMPLETE. New evidence; no exact historical resume claim.', flush=True)
    except Exception as error:
        continuation_summary = root / 'train/summary.json'
        if continuation_summary.is_file():
            latest = read(continuation_summary)
            state['training_started'] = any(v > 0 for v in latest.get('new_updates_per_arm', {}).values())
            state['updates_per_arm'] = latest.get('updates_per_arm')
        state['failure'] = type(error).__name__ + ': ' + str(error)
        write(root / 'workflow_state.json', state)
        (root / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('inspect', 'cpu', 'recover', 'report'), default='inspect')
    parser.add_argument('--failed-run', default=FAILED)
    parser.add_argument('--tool', default=TOOL)
    parser.add_argument('--output')
    parser.add_argument('--root', help='Existing recovery root for report')
    parser.add_argument('--after-sar', nargs='+', default=list(AFTER))
    parser.add_argument('--wait-timeout', type=int, default=0, help='SAR wait deadline in seconds; 0 keeps waiting')
    args = parser.parse_args()
    if args.wait_timeout < 0:
        parser.error('--wait-timeout must be nonnegative')
    if args.stage == 'report':
        if not args.root:
            parser.error('--root is required for report')
        state = read(Path(args.root) / 'workflow_state.json')
        summary = Path(args.root) / 'train/summary.json'
        if summary.is_file():
            data = read(summary)
            state['training_started'] = any(v > 0 for v in data.get('new_updates_per_arm', {}).values())
            state['current_training_phase'] = data.get('phase')
            state['updates_per_arm'] = data.get('updates_per_arm')
            print(json.dumps(state, ensure_ascii=False, indent=2))
            print(json.dumps({k: data[k] for k in ('phase', 'complete', 'updates_per_arm',
                             'new_updates_per_arm', 'prefix_updates_per_arm',
                             'full_validation_evidence', 'error', 'last_checkpoint') if k in data},
                             ensure_ascii=False, indent=2))
        else:
            print(json.dumps(state, ensure_ascii=False, indent=2))
        return
    if args.stage == 'recover':
        with recovery_lock():
            pipeline(args)
    else:
        pipeline(args)


if __name__ == '__main__':
    main()
