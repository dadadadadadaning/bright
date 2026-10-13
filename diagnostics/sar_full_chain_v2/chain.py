#!/usr/bin/env python3
"""Background SAR Q four-arm -> SAR/soft eight-head serial workflow.

Frozen-state witnessing is diagnostic instrumentation, not a proven repair.
Never certifies old failed weights, retries failures, or accesses Test.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = '/root/shared-nvme/BRIGHT-master'
DEFAULT_PROTOCOL = '/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34/fresh_e0/protocol.json'
DEFAULT_PARENT = '/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_FULL_CHAIN_V1'
PROBE_PARENT = '/root/shared-nvme/BRIGHT-checkpoints/SRAC_FROZEN_WITNESS_V1'
Q_MODULE = 'bda_benchmark.experiments.damageformer_srac_q_factorial_v1.train_q'
FORMAT = 'SAR_FULL_CHAIN_V2'


def read(path):
    result = json.loads(Path(path).read_text())
    if not isinstance(result, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return result


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def instrument_plan(plan, repo, witness, q_root, python):
    result = copy.deepcopy(plan)
    for stage in result:
        if stage['kind'] != 'train':
            continue
        old = stage['command']
        if len(old) < 5 or old[:4] != [str(python), '-u', '-m', Q_MODULE]:
            raise ValueError('Unexpected original Q training command')
        evidence = Path(q_root) / 'frozen_witness' / stage['name']
        if evidence.resolve().is_relative_to(Path(stage['output']).resolve()):
            raise ValueError('Witness output must be outside original fresh training output')
        stage['command'] = [str(python), '-u', str(witness), 'child', '--repo', str(repo),
                            '--evidence', str(evidence), '--', *old[4:]]
        stage['frozen_witness_script'] = str(witness)
        stage['frozen_witness_output'] = str(evidence)
    return result


def validate_witness(stage, q_root):
    output = Path(q_root) / 'frozen_witness' / stage['name']
    summary_path, declaration_path = output / 'summary.json', output / 'declaration.json'
    journal_path = output / 'witness.jsonl'
    data, declaration = read(summary_path), read(declaration_path)
    smoke = stage['smoke']
    expected = 2 if smoke else 2000
    original = data.get('original_summary', {})
    immutable = original.get('immutable')
    if (data.get('complete') is not True or data.get('exact_frozen_checks') is not True
            or data.get('updates') != expected or data.get('smoke') is not smoke
            or data.get('diagnostic_only') is not smoke or data.get('test_accessed') is not False
            or original.get('complete') is not True or original.get('steps') != expected
            or original.get('arm') != stage['arm'] or original.get('smoke') is not smoke
            or original.get('test_accessed') is not False
            or original.get('a1_unchanged') is not True or original.get('downstream_unchanged') is not True
            or original.get('full_validation_evidence') is not (not smoke)
            or not isinstance(immutable, dict) or not immutable
            or any(value is not True for value in immutable.values())):
        raise ValueError('Frozen witness did not complete its exact training budget: ' + stage['name'])
    arguments = declaration.get('arguments', {})
    if (declaration.get('format') != 'SAR_FROZEN_WITNESS_V1'
            or declaration.get('original_checks_enabled') is not True
            or declaration.get('changes_training_math') is not False
            or declaration.get('old_checkpoint_reused') is not False
            or declaration.get('test_accessed') is not False
            or declaration.get('diagnostic_only') is not smoke
            or arguments.get('arm') != stage['arm'] or arguments.get('steps') != expected
            or arguments.get('smoke') is not smoke):
        raise ValueError('Frozen witness declaration differs from stage')
    witness_script = stage.get('frozen_witness_script')
    if witness_script and file_sha(witness_script) != declaration.get('script_sha256'):
        raise ValueError('Frozen witness implementation changed')
    lines = [json.loads(line) for line in journal_path.read_text().splitlines() if line.strip()]
    if (not lines or lines[-1].get('event') != 'all_frozen_states_exact'
            or lines[-1].get('phase') != 'original_training_checks_passed'
            or lines[-1].get('step') != expected):
        raise ValueError('Frozen witness final exact check is missing')
    return dict(frozen_exact=True, frozen_witness_summary=str(summary_path),
                frozen_witness_summary_sha256=file_sha(summary_path),
                frozen_witness_journal_sha256=file_sha(journal_path),
                frozen_witness_declaration_sha256=file_sha(declaration_path))


def restore_environment(protocol, inherited):
    env = dict(inherited, CUDA_LAUNCH_BLOCKING='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    runtime = protocol['runtime_settings']
    fields = dict(runtime['environment'])
    fields['CUBLAS_WORKSPACE_CONFIG'] = runtime.get('cublas_workspace_config')
    for name in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_TF32_OVERRIDE', 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG'):
        value = fields[name]
        if value is None:
            env.pop(name, None)
        elif isinstance(value, str):
            env[name] = value
        else:
            raise ValueError('Invalid protocol environment: ' + name)
    return env


def run_serial(q_call, soft_call, record):
    record['phase'] = 'sar_q'
    q = q_call()
    if (q.get('complete') is not True or q.get('full_validation_evidence') is not True
            or q.get('immutable') is not True or q.get('test_accessed') is not False):
        raise ValueError('SAR Q source is not complete and verified; SAR+soft will not start')
    soft = None
    if soft_call is not None:
        record['phase'] = 'sar_soft'
        soft = soft_call()
        if (soft.get('complete') is not True or soft.get('full_validation_evidence') is not True
                or soft.get('test_accessed') is not False):
            raise ValueError('SAR+soft did not complete its formal evaluation')
    record['phase'] = 'complete'
    return dict(q=q, soft=soft)


def load_q_pipeline():
    location = HERE / 'sar_q_matched_v1' / 'pipeline.py'
    spec = importlib.util.spec_from_file_location('_sar_full_original_pipeline', location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package_binding():
    return {str(p): file_sha(p) for p in sorted(HERE.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.sh', '.json', '.md')}


def verify_binding(bindings):
    for name, sha in bindings.items():
        if not Path(name).is_file() or file_sha(name) != sha:
            raise ValueError('Bound code/protocol changed: ' + name)


def concurrent_queue(queue_module, reason):
    """Observe the actual GPU once and proceed under an explicit shared policy."""
    def proceed(device_index, log=None, poll_seconds=30, consecutive=2):
        uuid = queue_module.resolve_gpu(device_index)
        sample = queue_module.observation(uuid)
        result = dict(sample, wait_skipped=True, execution_policy='concurrent',
                      reason=reason, waited_seconds=0.0, cuda_context_created=False,
                      exclusive_reservation=False)
        (log or print)('GPU_SHARED_RUN ' + json.dumps(result, ensure_ascii=False))
        return result
    return proceed


def configure_gpu_queue(directory, run_now):
    # Q and Soft import this alias inside their run functions. Each driver
    # gets the exact bundled module; an override in the parent cannot cross exec.
    path = Path(directory) / 'gpu_queue.py'
    spec = importlib.util.spec_from_file_location('gpu_queue', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if run_now:
        module.wait_for_idle = concurrent_queue(module, 'user_selected_keep_memory_share_gpu')
    sys.modules['gpu_queue'] = module
    return module


def load_soft_pipeline():
    location = HERE / 'sar_soft_cross_v1' / 'run.py'
    spec = importlib.util.spec_from_file_location('_sar_full_original_soft', location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_soft(config):
    verify_binding(config['bindings'])
    pipeline = load_soft_pipeline()
    original_files = pipeline.source_files
    extra_files = package_binding()
    def files(repo):
        verify_binding(extra_files)
        return {**original_files(repo), **extra_files}
    pipeline.source_files = files
    args = pipeline.parser().parse_args([
        'run', '--repo-root', config['repo'], '--sar-root', config['q_root'],
        '--output', config['soft_root'], '--python', config['python']])
    args.execution_policy = 'concurrent' if config.get('run_now') else 'wait_idle'
    args.resource_policy_driver = str(HERE / 'chain.py')
    args.wait_gpu_idle = not config.get('run_now', False)
    sys.path.insert(0, str(HERE / 'sar_soft_cross_v1'))
    configure_gpu_queue(HERE / 'sar_soft_cross_v1', config.get('run_now', False))
    result = pipeline.run(args)
    verify_binding(config['bindings'])
    return result


def run_q(config):
    pipeline = load_q_pipeline()
    original_plan = pipeline.build_plan
    original_snapshot = pipeline.snapshot_sources
    original_gate = pipeline.validate_stage
    q_root = Path(config['q_root'])
    witness = HERE / 'SAR_FREEZE_WITNESS_V1.py'
    # Bind external instrumentation in the original immutable source manifest.
    extra_files = package_binding()
    pipeline.build_plan = lambda args: instrument_plan(
        original_plan(args), args.repo_root, witness, q_root, args.python)

    def snapshot(args, **kwargs):
        verify_binding(extra_files)
        return {**original_snapshot(args, **kwargs), **extra_files}

    def gate(stage):
        result = original_gate(stage)
        if stage['kind'] == 'train':
            result.update(validate_witness(stage, q_root))
        return result

    pipeline.snapshot_sources = snapshot
    pipeline.validate_stage = gate
    args = argparse.Namespace(output=str(q_root), repo_root=config['repo'], protocol=config['protocol'],
                              python=config['python'], workers=4, amp='bf16',
                              wait_gpu_idle=not config.get('run_now', False),
                              execution_policy='concurrent' if config.get('run_now') else 'wait_idle',
                              frozen_witness=str(witness), frozen_witness_sha256=file_sha(witness),
                              serial_launcher=str(HERE / 'chain.py'),
                              old_failed_checkpoint_reused=False)
    sys.path.insert(0, str(HERE / 'sar_q_matched_v1'))
    configure_gpu_queue(HERE / 'sar_q_matched_v1', config.get('run_now', False))
    pipeline.run(args)
    state = read(q_root / 'workflow_state.json')
    # Recheck the added evidence before releasing the Q source to Soft.
    for stage in read(q_root / 'manifest.json')['stages']:
        if stage['kind'] == 'train':
            validate_witness(stage, q_root)
    return state


def run_child(command, repo, log, env):
    with Path(log).open('x') as stream:
        process = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code:
        raise RuntimeError('Child failed with exit %d; see %s' % (code, log))


def process_identity(pid, proc_root='/proc'):
    folder = Path(proc_root) / str(pid)
    try:
        stat = (folder / 'stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        if fields[0] == 'Z':
            return None
        argv = [v.decode('utf-8', errors='replace') for v in (folder / 'cmdline').read_bytes().split(b'\0') if v]
        return dict(pid=int(pid), start_ticks=fields[19], argv=argv)
    except (OSError, ValueError, IndexError):
        return None


def discover_memory_processes(proc_root='/proc'):
    result = []
    for folder in Path(proc_root).iterdir():
        if not folder.name.isdigit():
            continue
        item = process_identity(folder.name, proc_root)
        if item and any(token.endswith('.py') and any(part in token for part in (
                '/joint_live_recovery_v2/', '/joint_pool_recovery_v1/')) for token in item['argv']):
            result.append(item)
    return sorted(result, key=lambda row: row['pid'])


@contextmanager
def resource_policy(config, record, root):
    declared = discover_memory_processes()
    record['declared_memory_processes'] = declared
    record['execution_policy'] = 'concurrent' if config.get('run_now') else 'wait_idle'
    if config.get('run_now'):
        record['phase'] = 'concurrent_start'
        write(root / 'chain_state.json', record)
        print('SHARED_GPU_POLICY: keeping Memory processes; skipping Memory/probe/GPU idle waits. '
              'Required source and training checks remain enabled.', flush=True)
        yield
        return
    record['phase'] = 'waiting_declared_memory'
    write(root / 'chain_state.json', record)
    while True:
        alive = []
        for item in declared:
            current = process_identity(item['pid'])
            if current and current['start_ticks'] == item['start_ticks']:
                alive.append(item['pid'])
        if not alive:
            break
        print('WAIT_DECLARED_MEMORY ' + str(alive), flush=True)
        time.sleep(30)
    record['phase'] = 'waiting_existing_probe'
    write(root / 'chain_state.json', record)
    probe_parent = Path(config['probe_parent'])
    probe_parent.mkdir(parents=True, exist_ok=True)
    with (probe_parent / '.probe.lock').open('a+') as probe_lock:
        while True:
            try:
                fcntl.flock(probe_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                print('WAITING_EXISTING_SAR_PROBE: no GPU context created', flush=True)
                time.sleep(30)
        yield


def worker(root, lock_fd):
    # Keep inherited flock open for this worker's entire lifetime.
    os.fstat(lock_fd)
    root = Path(root)
    config = read(root / 'launch.json')
    record = dict(format=FORMAT, complete=False, full_validation_evidence=False,
                  phase='initializing', pid=os.getpid(), target=config['target'],
                  q_root=config['q_root'], soft_root=config['soft_root'],
                  started_at=time.time(), test_accessed=False)
    identity = process_identity(os.getpid())
    record['pid_start_ticks'] = identity['start_ticks'] if identity else None
    write(root / 'chain_state.json', record)
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt('SAR launcher received a termination signal')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        verify_binding(config['bindings'])
        protocol = read(config['protocol'])
        env = restore_environment(protocol, os.environ)
        os.environ.clear()
        os.environ.update(env)
        with resource_policy(config, record, root):
            def q_call():
                record['phase'] = 'checking_existing_q' if config['target'] == 'sar-soft' else 'sar_q'
                write(root / 'chain_state.json', record)
                verify_binding(config['bindings'])
                if config['target'] == 'sar-soft':
                    return read(Path(config['q_root']) / 'workflow_state.json')
                return run_q(config)

            def soft_call():
                record['phase'] = 'sar_soft'
                write(root / 'chain_state.json', record)
                verify_binding(config['bindings'])
                command = [config['python'], '-u', str(HERE / 'chain.py'),
                           '--soft-worker-root', str(root)]
                print('START SAR_PLUS_SOFT; source=' + config['q_root'], flush=True)
                run_child(command, config['repo'], root / 'sar_soft_driver.log', env)
                return read(Path(config['soft_root']) / 'workflow_state.json')

            run_serial(q_call, None if config['target'] == 'sar-q' else soft_call, record)
        verify_binding(config['bindings'])
        record.update(complete=True, full_validation_evidence=True, finished_at=time.time())
        write(root / 'chain_state.json', record)
        print('SERIAL_WORKFLOW_COMPLETE ' + str(root), flush=True)
    except BaseException as exc:
        record.update(complete=False, full_validation_evidence=False, failed_phase=record['phase'],
                      phase='failed', failure=repr(exc), finished_at=time.time())
        write(root / 'chain_state.json', record)
        (root / 'failure_traceback.txt').write_text(traceback.format_exc())
        print('FAILED. Later stages stopped; preserved all outputs. ' + str(root), flush=True)
        raise
    finally:
        os.close(lock_fd)


def checked_config(args):
    repo, protocol = Path(args.repo).expanduser().resolve(), Path(args.protocol).expanduser().resolve()
    if protocol.is_dir():
        protocol = protocol / 'protocol.json'
    if not (repo / 'bda_benchmark').is_dir():
        raise ValueError('Missing repository: ' + str(repo))
    q = load_q_pipeline()
    record, _device = q._validate_protocol(protocol)
    # CPU file checks; does not import Torch/create a CUDA context.
    for name in ('front', 'a1', 'train_list', 'val_list'):
        path = Path(record['input_paths'][name])
        if file_sha(path) != record['input_fingerprints'][name]:
            raise ValueError('E0 source file differs: ' + name)
    parent = Path(args.output_parent).expanduser().resolve()
    if parent.is_relative_to(repo) or parent.is_relative_to(HERE):
        raise ValueError('Use checkpoint output outside repository and tool package')
    if args.target == 'sar-soft':
        if not args.sar_root:
            raise ValueError('--target sar-soft requires --sar-root with a fully passed source')
        source = Path(args.sar_root).expanduser().resolve()
        state = read(source / 'workflow_state.json')
        run_serial(lambda: state, None, {})
        manifest = read(source / 'manifest.json')
        source_protocol = Path(manifest['arguments']['protocol']).resolve()
        if source_protocol.is_dir():
            source_protocol /= 'protocol.json'
        if source_protocol != protocol or Path(manifest['arguments']['repo_root']).resolve() != repo:
            raise ValueError('Existing SAR source uses a different repository or E0')
    elif args.sar_root:
        raise ValueError('--sar-root is only for --target sar-soft; full/Q runs use fresh E0 training')
    bindings = package_binding()
    bindings[str(protocol)] = file_sha(protocol)
    return dict(format=FORMAT, repo=str(repo), protocol=str(protocol),
                python=os.path.abspath(sys.executable), target=args.target,
                run_now=bool(getattr(args, 'run_now', False)),
                execution_policy='concurrent' if getattr(args, 'run_now', False) else 'wait_idle',
                probe_parent=str(Path(args.probe_parent).expanduser().resolve()),
                bindings=bindings, test_accessed=False, old_failed_checkpoint_reused=False)


def start(args):
    config = checked_config(args)
    parent = Path(args.output_parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    if getattr(args, 'replace_queued', False):
        from queue_control import cancel_queued
        result = cancel_queued(parent)
        print('QUEUE_REPLACEMENT=' + json.dumps(result, ensure_ascii=False), flush=True)
    with (parent / '.serial.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A SAR serial workflow is already queued/running; use --status')
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        root = Path(tempfile.mkdtemp(prefix='run_' + stamp + '_', dir=str(parent)))
        config.update(root=str(root), q_root=str(Path(args.sar_root).expanduser().resolve()) if args.target == 'sar-soft'
                      else str(root / 'sar_q'), soft_root=str(root / 'sar_soft'))
        write(root / 'launch.json', config)
        write(root / 'chain_state.json', dict(format=FORMAT, phase='starting', complete=False,
            full_validation_evidence=False, target=args.target, test_accessed=False))
        command = [config['python'], '-u', str(HERE / 'chain.py'), '--worker-root', str(root),
                   '--inherited-lock-fd', str(lock.fileno())]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        with (root / 'launcher.log').open('xb') as stream:
            child = subprocess.Popen(command, cwd=config['repo'], stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                     env=env, pass_fds=(lock.fileno(),))
        launch = dict(root=str(root), pid=child.pid, q_root=config['q_root'],
                      soft_root=config['soft_root'], log=str(root / 'launcher.log'),
                      execution_policy=config.get('execution_policy', 'wait_idle'),
                      started_at=time.time(), target=args.target, started_is_not_passed=True)
        write(root / 'launcher.json', launch)
        write(parent / 'latest.json', launch)
    print(json.dumps(launch, ensure_ascii=False, indent=2), flush=True)
    descriptions = {'all': 'SAR四组通过后自动进入SAR+soft',
                    'sar-q': '只运行完整SAR四组', 'sar-soft': '核验指定Q来源后运行SAR+soft'}
    policy = '后台已启动，可断开SSH；共享GPU，完成必要检查后直接运行；' if config.get('run_now') else \
             '后台已排队；可断开SSH。等待现有Memory/probe和GPU；'
    print(policy + descriptions[args.target] + '。', flush=True)
    print('不是从旧失败权重续训；不会自动放宽校验或重试失败。', flush=True)


def status(args):
    root = Path(args.root).expanduser().resolve() if args.root else Path(read(
        Path(args.output_parent).expanduser().resolve() / 'latest.json')['root'])
    config, state = read(root / 'launch.json'), read(root / 'chain_state.json')
    print('CHAIN_ROOT=' + str(root))
    print(json.dumps(state, ensure_ascii=False, indent=2))
    current = process_identity(state.get('pid', -1))
    alive = bool(current and current['start_ticks'] == state.get('pid_start_ticks'))
    print('WORKER_PROCESS_ALIVE=' + str(alive))
    if not alive and state.get('complete') is not True and state.get('phase') not in ('failed', 'starting'):
        print('状态未完成且未发现原进程；请读取launcher.log，不要仅凭旧phase判断仍在运行。')
    for label, path in (('SAR_Q', Path(config['q_root'])), ('SAR_SOFT', Path(config['soft_root']))):
        p = path / 'workflow_state.json'
        if p.is_file():
            d = read(p)
            print(label, json.dumps({k: d.get(k) for k in
                  ('current_stage', 'complete', 'full_validation_evidence', 'failure')}, ensure_ascii=False))
        else:
            print(label, '尚未开始')
    for path in sorted((Path(config['q_root']) / 'frozen_witness').glob('*/failure.json')):
        print('FROZEN_FAILURE=' + str(path))
        print(path.read_text())
    print('LOG=' + str(root / 'launcher.log'))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--repo', default=DEFAULT_REPO)
    p.add_argument('--protocol', default=DEFAULT_PROTOCOL)
    p.add_argument('--output-parent', default=DEFAULT_PARENT)
    p.add_argument('--probe-parent', default=PROBE_PARENT)
    p.add_argument('--target', choices=('all', 'sar-q', 'sar-soft'), default='all')
    p.add_argument('--sar-root')
    p.add_argument('--status', action='store_true')
    p.add_argument('--root')
    p.add_argument('--plan', action='store_true')
    p.add_argument('--run-now', action='store_true', help='Keep Memory running; share GPU without resource waits')
    p.add_argument('--replace-queued', action='store_true', help='Cancel only a verified waiting V1/V2 SAR chain before starting')
    p.add_argument('--cancel-queued', action='store_true', help='Cancel a verified waiting V1/V2 SAR chain only; no new training')
    p.add_argument('--worker-root', help=argparse.SUPPRESS)
    p.add_argument('--soft-worker-root', help=argparse.SUPPRESS)
    p.add_argument('--inherited-lock-fd', type=int, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    actions = sum(bool(v) for v in (args.status, args.plan, args.cancel_queued,
                                    args.worker_root, args.soft_worker_root))
    if actions > 1:
        p.error('Choose exactly one action')
    if args.replace_queued and (not args.run_now or actions):
        p.error('--replace-queued is only for a new --run-now launch')
    if args.root and not (args.status or args.cancel_queued):
        p.error('--root is only for --status or --cancel-queued')
    if args.soft_worker_root:
        config = read(Path(args.soft_worker_root) / 'launch.json')
        def terminate_soft(_signum, _frame):
            raise KeyboardInterrupt('Soft driver received a termination signal')
        signal.signal(signal.SIGTERM, terminate_soft)
        run_soft(config)
    elif args.worker_root:
        if args.inherited_lock_fd is None:
            p.error('Worker requires an inherited lock')
        worker(args.worker_root, args.inherited_lock_fd)
    elif args.status:
        status(args)
    elif args.cancel_queued:
        from queue_control import cancel_queued
        print(json.dumps(cancel_queued(args.output_parent, args.root), ensure_ascii=False, indent=2))
    elif args.plan:
        print(json.dumps(dict(target=args.target, no_training_started=True,
            q_arms=['baseline', 'rgb_capacity', 'shifted_sar', 'sar_cue'],
            q_updates_per_arm=2000, soft_q_sources=['baseline', 'sar_cue'],
            soft_routes=['unsplit', 'capacity_control', 'hard', 'soft'],
            requires_exact_frozen_checks=True, old_failed_checkpoint_reused=False,
            execution_policy='concurrent' if args.run_now else 'wait_idle',
            wait_for_declared_memory=not args.run_now, wait_for_existing_probe=not args.run_now,
            wait_for_gpu_idle=not args.run_now, soft_wait_for_gpu_idle=not args.run_now,
            source_root=args.sar_root if args.target == 'sar-soft' else 'new passed SAR Q root',
            training_repair_proven=False, test_accessed=False), ensure_ascii=False, indent=2))
    else:
        start(args)


if __name__ == '__main__':
    main()
