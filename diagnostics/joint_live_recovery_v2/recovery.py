#!/usr/bin/env python3
"""Capture the live pooling discrepancy, verify an explicit CPU backend, then continue.

The coordinator imports no torch. Probe, CPU analysis, and continuation always
run in separate processes. Original protocols, checkpoints and logs are read only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import source_binding
from source_binding import (AFTER, FAILED, TOOL, beneath, check_preserved as _check_original_preserved, child_environment,
                            digest, inspect_failed, read, source_hash, terminal_source,
                            verify_release, wait_sources, write)

HERE = Path(__file__).resolve().parent
FORMAT = 'SRAC_JOINT_LIVE_RECOVERY_V2'
RESULT_BASE = Path('/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_LIVE_RECOVERY_V2')
LATEST_FAILED = '/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_POOL_RECOVERY_V1/run_20261013_123313_019791_4018193'
LIVE_BACKEND = 'live_contiguous_probe'
CPU_BACKEND = 'cpu_fp32_reference'


def check_preserved(evidence):
    _check_original_preserved(evidence)
    for filename, expected in evidence.get('diagnostic_source_sha256', {}).items():
        if not Path(filename).is_file() or digest(filename) != expected:
            raise ValueError('Preserved latest numeric-failure evidence changed: ' + filename)


def inspect_sources(failed_run, tool, latest_failed_recovery):
    evidence = inspect_failed(failed_run, tool)
    evidence['original_preserved_sha256'] = dict(evidence['preserved_sha256'])
    latest = Path(latest_failed_recovery).expanduser().resolve()
    if latest == Path(evidence['failed_run']).resolve():
        raise ValueError('Select the latest numeric recovery failure separately from the original CUDA failure')
    debug = beneath(latest / 'train/pool_debug', latest)
    paths = {name: beneath(path, latest) for name, path in (
        ('failure', debug / 'failure.json'), ('evidence', latest / 'evidence.json'),
        ('runtime', latest / 'runtime.json'), ('summary', latest / 'train/summary.json'),
        ('manifest', latest / 'train/manifest.json'), ('workflow', latest / 'workflow_state.json'),
    )}
    hashes = {str(path): digest(path) for path in paths.values()}
    failure, previous, runtime, summary, manifest, workflow = (
        read(paths[key]) for key in ('failure', 'evidence', 'runtime', 'summary', 'manifest', 'workflow'))
    if failure.get('phase') != 'cpu_reference_gate':
        raise ValueError('The latest recovery must be the saved cpu_reference_gate failure')
    image, call = failure.get('image'), failure.get('call')
    if not isinstance(image, str) or not image or type(call) is not int or call < 1:
        raise ValueError('Latest recovery lacks a valid probe image and call number')
    filename = failure.get('input')
    if not isinstance(filename, str) or not filename:
        raise ValueError('Latest numeric failure has no saved snapshot')
    snapshot = Path(filename).expanduser()
    snapshot = beneath(snapshot if snapshot.is_absolute() else debug / snapshot, debug)
    if not snapshot.is_file():
        raise FileNotFoundError(snapshot)
    hashes[str(snapshot)] = digest(snapshot)
    if runtime != evidence['runtime'] or runtime != previous.get('runtime'):
        raise ValueError('Latest recovery runtime differs from the original source binding')
    for field in ('failed_run', 'checkpoint', 'checkpoint_step', 'original_source_hash', 'arguments'):
        if previous.get(field) != evidence.get(field):
            raise ValueError('Latest recovery belongs to a different original binding: ' + field)
    protocol = dict(manifest.get('protocol_info', {}))
    protocol.pop('recovery_binding', None)
    if protocol != evidence['protocol_info'] or manifest.get('source_hash') != evidence['original_source_hash']:
        raise ValueError('Latest recovery model/source protocol binding differs')
    expected_updates = {arm: evidence['checkpoint_step'] for arm in ('r0', 'rt', 'rf')}
    if (summary.get('phase') != 'checkpoint_revalidation' or summary.get('complete') is True
            or not summary.get('error') or summary.get('updates_per_arm') != expected_updates
            or summary.get('new_updates_per_arm') != {arm: 0 for arm in expected_updates}
            or workflow.get('complete') is True or not workflow.get('failure')):
        raise ValueError('Latest recovery is not a terminated checkpoint revalidation with zero new updates')
    evidence.update(latest_failed_recovery=str(latest), latest_snapshot=str(snapshot),
                    latest_snapshot_sha256=hashes[str(snapshot)], probe_image=image, probe_call=call,
                    latest_failure=failure, explicit_pool_backend=CPU_BACKEND,
                    exact_historical_resume_claim=False)
    evidence['diagnostic_source_sha256'] = hashes
    check_preserved(evidence)
    return evidence


def log_tail(path, count=24):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 65536))
        return stream.read().decode('utf-8', errors='replace').splitlines()[-count:]


def run_child(command, log, env):
    with Path(log).open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=HERE)
        started = time.monotonic()
        try:
            while True:
                try:
                    return process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    lines = log_tail(log, 1)
                    print('RUNNING ' + Path(log).name + ' elapsed_seconds=' + str(int(time.monotonic() - started))
                          + (' last_line=' + lines[0][:320] if lines else ''), flush=True)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
            raise


@contextmanager
def gpu_environment(env):
    keys = ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER')
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            if key in env:
                os.environ[key] = env[key]
            else:
                os.environ.pop(key, None)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def recovery_lock(base=RESULT_BASE, legacy_base=source_binding.RESULT_BASE):
    """Share the V1 lock so an old and new continuation cannot run together."""
    with source_binding.recovery_lock(legacy_base):
        with source_binding.recovery_lock(base):
            yield


def _capture(record, destination, expected_image, expected_call, expected_backend):
    if record.get('capture_ready') is not True:
        raise ValueError('Probe lacks an adequate actual-forward capture')
    filename = record.get('capture_path') or record.get('packet_path') or record.get('capture') or 'capture.pth'
    path = Path(filename)
    path = beneath(path if path.is_absolute() else Path(destination) / path, destination)
    if not path.is_file():
        raise FileNotFoundError('Actual-forward capture missing: ' + str(path))
    expected_hash = record.get('capture_sha256') or record.get('packet_sha256')
    if not isinstance(expected_hash, str) or digest(path) != expected_hash:
        raise ValueError('Actual-forward capture checksum mismatch')
    image, call = record.get('image'), record.get('call')
    if (image != expected_image or type(call) is not int or call != expected_call
            or record.get('backend') != expected_backend):
        raise ValueError('Actual-forward capture identity/backend differs from probe configuration')
    if record.get('probe_only') is not True or record.get('new_updates_per_arm') != {'r0': 0, 'rt': 0, 'rf': 0}:
        raise ValueError('Probe must not perform optimizer updates')
    return dict(path=str(path), sha256=expected_hash, image=image, call=call, backend=expected_backend)


def _live_probe_ready(record):
    phase = record.get('failure_phase')
    failure = record.get('failure')
    if isinstance(failure, dict):
        phase = phase or failure.get('phase')
    return (record.get('execution_complete') is True and record.get('capture_ready') is True
            and (not failure or phase == 'cpu_reference_gate'))


def _analysis_ready(record, cpu=False):
    fields = ['execution_complete', 'diagnostic_complete', 'adequate_capture',
              'cpu64_reference_agreement', 'cpu32_reference_reproduced']
    if cpu:
        fields += ['CPU_backend_vs_FP64_passed', 'cpu_continuation_checks_passed']
    return not record.get('failure') and all(record.get(key) is True for key in fields)


def check_train_gates(evidence_path):
    """Validate persisted probe proofs even when someone invokes the worker directly."""
    evidence_path = Path(evidence_path).expanduser().resolve()
    root = evidence_path.parent
    evidence = read(evidence_path)
    check_preserved(evidence)
    state = read(root / 'workflow_state.json')
    if (state.get('format') != FORMAT or state.get('source_binding_sha256') != digest(evidence_path)
            or state.get('probe_gates_passed') is not True
            or state.get('arguments', {}).get('stage') != 'recover'
            or state.get('current_stage') != 'train'
            or state.get('stages', {}).get('train', {}).get('status') != 'running'):
        raise ValueError('Train worker requires the bound parent recovery workflow and completed probe gates')
    if (source_hash(HERE) != evidence.get('repair_source_hash')
            or verify_release(HERE) != evidence.get('repair_release_sha256')):
        raise ValueError('Train preflight package binding changed')
    proofs = {}
    for name in ('live-probe', 'live_analysis', 'cpu-probe', 'cpu_analysis'):
        row = state.get('stages', {}).get(name, {})
        folder = beneath(row.get('output', ''), root)
        path = beneath(folder / 'summary.json', root)
        if (row.get('status') != 'complete' or row.get('returncode') != 0
                or not path.is_file() or digest(path) != row.get('summary_sha256')):
            raise ValueError('Missing or changed persisted preflight stage: ' + name)
        proofs[name] = read(path)
    captures = {}
    for mode, backend in (('live-probe', LIVE_BACKEND), ('cpu-probe', CPU_BACKEND)):
        record = proofs[mode]
        if mode == 'live-probe':
            if not _live_probe_ready(record):
                raise ValueError('Persisted live probe is inadequate')
        elif (record.get('execution_complete') is not True or record.get('strict_cpu_gate_passed') is not True
              or record.get('full_single_image_forward_complete') is not True or record.get('failure')):
            raise ValueError('Persisted CPU full-image probe has not passed')
        captures[mode] = _capture(record, state['stages'][mode]['output'], evidence['probe_image'],
                                  evidence['probe_call'], backend)
    for mode, probe_mode in (('live', 'live-probe'), ('cpu', 'cpu-probe')):
        record, capture = proofs[mode + '_analysis'], captures[probe_mode]
        meta = record.get('capture_meta', {})
        if (not _analysis_ready(record, cpu=mode == 'cpu')
                or record.get('capture_sha256') != capture['sha256']
                or Path(record.get('capture', '')).resolve() != Path(capture['path']).resolve()
                or meta.get('image') != capture['image'] or meta.get('call') != capture['call']
                or meta.get('backend') != capture['backend']):
            raise ValueError('Persisted independent reference analysis binding/gate failed: ' + mode)
        if record.get('evidence_sha256') != digest(evidence_path):
            raise ValueError('Analysis belongs to a different parent evidence binding')
    return dict(source_binding_sha256=digest(evidence_path), probe_gates_verified=True,
                cpu_backend_explicit=True, backend=CPU_BACKEND,
                checkpoint_val450_still_required=True,
                captures=captures,
                stage_summary_sha256={name: state['stages'][name]['summary_sha256'] for name in proofs})


def write_report(root, state):
    lines = [FORMAT, '结果目录: ' + str(root), '当前阶段: ' + str(state.get('current_stage')),
             '训练子阶段: ' + str(state.get('current_training_phase')),
             '显式冻结前端汇聚后端: ' + CPU_BACKEND + '；主干和Reader保持原GPU配置。',
             '原生前缀更新数: ' + str(state.get('prefix_updates')),
             '当前三组更新数: ' + str(state.get('updates_per_arm')),
             '新增三组更新数: ' + str(state.get('new_updates_per_arm')),
             'training_started=' + str(state.get('training_started')),
             'complete=' + str(state.get('complete')) + '; full_validation_evidence='
             + str(state.get('full_validation_evidence')),
             '固定原门槛atol=rtol=2e-5；没有自动回退、跳过图片或放宽容差。',
             '本次是显式CPU汇聚的独立继续训练，不宣称精确复现未中断的GPU运行。']
    for name, row in state.get('stages', {}).items():
        lines.append(name + ': ' + str(row.get('status')) + '; log=' + str(row.get('log')))
        record = row.get('summary', {})
        selected = {key: record.get(key) for key in (
            'execution_complete', 'capture_ready', 'strict_cpu_gate_passed', 'adequate_capture',
            'cpu64_reference_agreement', 'CPU_backend_vs_FP64_passed', 'cpu32_reference_reproduced',
            'phase', 'updates_per_arm', 'new_updates_per_arm', 'complete') if key in record}
        if selected:
            import json
            lines.append('  ' + json.dumps(selected, ensure_ascii=False))
        if row.get('error'):
            lines.append('  错误: ' + str(row['error'])[:1200])
        if row.get('status') == 'failed':
            lines += ['  ' + line[:1000] for line in row.get('log_tail', [])[-12:]]
    if state.get('failure'):
        lines.append('流程失败: ' + str(state['failure']))
    lines.append('Test未访问；旧失败目录及旧模型/案例库保持原记录。')
    (Path(root) / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def report(root):
    root = Path(root).expanduser().resolve()
    state = read(root / 'workflow_state.json')
    summary = root / 'train/summary.json'
    if summary.is_file():
        saved = read(summary)
        state.update(current_training_phase=saved.get('phase'),
                     updates_per_arm=saved.get('updates_per_arm'),
                     new_updates_per_arm=saved.get('new_updates_per_arm'),
                     training_started=any(count > 0 for count in saved.get('new_updates_per_arm', {}).values()))
    import json
    write_report(root, state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    print('PASTEABLE_REPORT=' + str(root / 'report.txt'), flush=True)
    return state


def pipeline(args):
    release = verify_release(HERE)
    evidence = inspect_sources(args.failed_run, args.tool, args.latest_failed_recovery)
    if args.stage == 'inspect':
        import json
        print(json.dumps({key: evidence[key] for key in (
            'failed_run', 'latest_failed_recovery', 'checkpoint', 'checkpoint_step', 'probe_image', 'probe_call',
            'explicit_pool_backend')}, ensure_ascii=False, indent=2))
        return evidence
    parent = Path(args.output_parent).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else parent / (
        'run_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '_' + str(os.getpid()))
    output = beneath(output, parent)
    for forbidden in (HERE, Path(evidence['failed_run']), Path(evidence['latest_failed_recovery']),
                      Path(evidence['tool']), Path(evidence['repo'])):
        if output == forbidden or forbidden in output.parents:
            raise ValueError('Use a new output directory outside preserved source/package/failed runs')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    evidence.update(repair_source_hash=source_hash(HERE), repair_release_sha256=release,
                    recovery_format=FORMAT, cpu_backend_explicit=True,
                    numeric_tolerance={'atol': 2e-5, 'rtol': 2e-5})
    write(output / 'evidence.json', evidence)
    evidence_sha256 = digest(output / 'evidence.json')
    write(output / 'runtime.json', evidence['runtime'])
    runtime_sha256 = digest(output / 'runtime.json')
    env = child_environment(evidence['runtime'])
    state = dict(format=FORMAT, complete=False, full_validation_evidence=False,
                 execution_complete=False, exact_historical_resume_claim=False,
                 original_run_complete=False, pool_backend=CPU_BACKEND, automatic_fallback=False,
                 tolerance_relaxed=False, current_stage='waiting_declared_sar', stages={},
                 prefix_updates=evidence['checkpoint_step'], target_updates=5000,
                 updates_per_arm={arm: evidence['checkpoint_step'] for arm in ('r0', 'rt', 'rf')},
                 new_updates_per_arm={arm: 0 for arm in ('r0', 'rt', 'rf')},
                 training_started=False, training_gate_passed=False, probe_gates_passed=False, test_accessed=False,
                 failure=None, arguments=vars(args), source_binding_sha256=evidence_sha256,
                 repair_source_hash=evidence['repair_source_hash'], repair_release_sha256=release)

    def save():
        write(output / 'workflow_state.json', state)
        write(output / 'summary.json', state)
        write_report(output, state)

    def intact():
        check_preserved(evidence)
        if digest(output / 'evidence.json') != evidence_sha256 or digest(output / 'runtime.json') != runtime_sha256:
            raise ValueError('New worker source/runtime binding changed during recovery')
        if source_hash(HERE) != evidence['repair_source_hash'] or verify_release(HERE) != release:
            raise ValueError('Live recovery package changed during execution')

    def stage(name, command, destination, acceptable):
        intact()
        state['current_stage'] = name
        log = output / 'logs' / (name + '.log')
        row = dict(status='running', command=command, output=str(destination), log=str(log))
        state['stages'][name] = row
        save()
        print('START ' + name + ' log=' + str(log), flush=True)
        code = run_child(command, log, env)
        record = read(destination / 'summary.json') if (destination / 'summary.json').is_file() else {}
        row.update(returncode=code, summary=record,
                   summary_sha256=digest(destination / 'summary.json') if (destination / 'summary.json').is_file() else None,
                   status='complete' if code == 0 and acceptable(record) else 'failed')
        if name == 'train':
            state.update(updates_per_arm=record.get('updates_per_arm', state['updates_per_arm']),
                         new_updates_per_arm=record.get('new_updates_per_arm', state['new_updates_per_arm']),
                         training_started=any(count > 0 for count in record.get('new_updates_per_arm', {}).values()))
        if row['status'] == 'failed':
            row['error'] = str(record.get('failure') or record.get('error') or 'Child exited without required complete evidence')
            row['log_tail'] = log_tail(log)
            for line in row['log_tail']:
                print('LOG_TAIL ' + name + ' ' + line[:1000], flush=True)
        save()
        intact()
        if row['status'] != 'complete':
            raise RuntimeError('Stage failed: ' + name + '; ' + row['error'] + '; read ' + str(log))
        print('END ' + name + ' complete', flush=True)
        return record

    def probe(mode, backend):
        destination = output / mode
        command = [sys.executable, '-u', str(HERE / 'worker.py'), '--mode', mode,
                   '--evidence', str(output / 'evidence.json'), '--output', str(destination)]
        def acceptable(record):
            if mode == 'live-probe':
                return _live_probe_ready(record)
            return (record.get('execution_complete') is True and record.get('strict_cpu_gate_passed') is True
                    and record.get('full_single_image_forward_complete') is True and not record.get('failure'))
        record = stage(mode, command, destination, acceptable)
        capture = _capture(record, destination, evidence['probe_image'], evidence['probe_call'], backend)
        state.setdefault('captures', {})[mode] = capture
        save()
        return capture

    def analyze(capture, mode):
        destination = output / (mode + '_analysis')
        command = [sys.executable, '-u', str(HERE / 'analysis.py'), '--input', capture['path'],
                   '--output', str(destination), '--expected-sha256', capture['sha256'],
                   '--expected-image', capture['image'], '--expected-call', str(capture['call']),
                   '--expected-backend', capture['backend'], '--mode', mode,
                   '--evidence', str(output / 'evidence.json')]
        return stage(mode + '_analysis', command, destination, lambda record: _analysis_ready(record, cpu=mode == 'cpu'))

    save()
    print('OUTPUT_ROOT=' + str(output), flush=True)
    try:
        state['sar_terminal'] = wait_sources(args.after_sar, args.wait_timeout)
        state['current_stage'] = 'waiting_gpu'
        save()
        from gpu_queue import wait_for_idle
        device = evidence['arguments'].get('device')
        if not isinstance(device, str) or not re.fullmatch(r'cuda:\d+', device):
            raise ValueError('Original source must bind an explicit cuda:N device')
        with gpu_environment(env):
            state['gpu_idle_observation'] = wait_for_idle(int(device[5:]), log=lambda text: print(text, flush=True))
        if not all(terminal_source(root)[0] for root in args.after_sar):
            raise ValueError('Declared SAR task became nonterminal; live recovery stopped')
        intact()
        live = probe('live-probe', LIVE_BACKEND)
        analyze(live, 'live')
        cpu = probe('cpu-probe', CPU_BACKEND)
        analyze(cpu, 'cpu')
        state.update(probe_gates_passed=True, probes_complete=True)
        save()
        if args.stage == 'probe':
            state.update(complete=True, execution_complete=True, current_stage='probe_complete', training_started=False,
                         interpretation='Live captures and CPU backend gates passed; probe-only, no training updates')
            save()
            return state
        for capture in (live, cpu):
            if digest(capture['path']) != capture['sha256']:
                raise ValueError('Approved actual-forward capture changed before training')
        if not all(terminal_source(root)[0] for root in args.after_sar):
            raise ValueError('Declared SAR task became nonterminal before continuation')
        train = output / 'train'
        record = stage('train', [sys.executable, '-u', str(HERE / 'worker.py'), '--mode', 'train',
                                '--evidence', str(output / 'evidence.json'), '--output', str(train)], train,
                       lambda value: value.get('complete') is True and value.get('full_validation_evidence') is True
                       and value.get('updates_per_arm') == {'r0': 5000, 'rt': 5000, 'rf': 5000})
        state.update(complete=True, execution_complete=True, current_stage='complete',
                     full_validation_evidence=True, training_started=True, training_gate_passed=True,
                     updates_per_arm=record['updates_per_arm'],
                     new_updates_per_arm=record.get('new_updates_per_arm'),
                     interpretation='Original native 1500-update prefix plus explicit CPU-pool continuation; no exact historical resume claim')
        save()
        print('LIVE RECOVERY TRAIN/VAL450 COMPLETE. No Test. New explicit CPU backend experiment.', flush=True)
        return state
    except BaseException as error:
        state.update(complete=False, execution_complete=False, full_validation_evidence=False,
                     failure=type(error).__name__ + ': ' + str(error))
        active = state['stages'].get(state['current_stage'])
        if active is not None and active.get('status') != 'failed':
            active.update(status='failed', error=state['failure'], log_tail=log_tail(active['log']))
        summary = output / 'train/summary.json'
        if summary.is_file():
            saved = read(summary)
            state.update(updates_per_arm=saved.get('updates_per_arm', state['updates_per_arm']),
                         new_updates_per_arm=saved.get('new_updates_per_arm', state['new_updates_per_arm']),
                         training_started=any(count > 0 for count in saved.get('new_updates_per_arm', {}).values()))
        save()
        (output / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('LIVE RECOVERY FAILED: ' + str(state['failure']) + '; report=' + str(output / 'report.txt'), flush=True)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--stage', choices=('inspect', 'probe', 'recover', 'report'), default='recover')
    result.add_argument('--failed-run', default=FAILED)
    result.add_argument('--tool', default=TOOL)
    result.add_argument('--latest-failed-recovery', default=LATEST_FAILED)
    result.add_argument('--output-parent', default=str(RESULT_BASE), help='Parent for unique new live-recovery result roots')
    result.add_argument('--output', help='NEW live-recovery result root; existing roots are never overwritten')
    result.add_argument('--root', help='Existing root for the read-only report')
    result.add_argument('--after-sar', nargs=2, default=list(AFTER), metavar=('Q_ROOT', 'SOFT_ROOT'))
    result.add_argument('--wait-timeout', type=int, default=0, help='SAR terminal-state deadline; 0 keeps waiting')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.wait_timeout < 0:
        raise ValueError('--wait-timeout must be nonnegative')
    if args.stage == 'report':
        if not args.root:
            raise ValueError('--root is required for report')
        return report(args.root)
    if args.stage == 'inspect':
        return pipeline(args)
    with recovery_lock(base=args.output_parent):
        return pipeline(args)


if __name__ == '__main__':
    main()
