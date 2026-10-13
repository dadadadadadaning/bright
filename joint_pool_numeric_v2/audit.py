#!/usr/bin/env python3
"""Diagnose one saved pooling numeric discrepancy; never train or relax gates.

This coordinator imports only the standard library. Every torch/CUDA worker
has a separate process, so a device assertion cannot poison the next worker.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
FORMAT = 'SRAC_JOINT_POOL_NUMERIC_AUDIT_V2'
RECOVERY = '/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_POOL_RECOVERY_V1/run_20261013_113711_128984_3991264'
RESULT_BASE = '/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_POOL_NUMERIC_AUDIT_V2'
SAR_ROOTS = (
    '/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_Q_MATCHED_V1/run_20261013_082551_3863093',
    '/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_SOFT_CROSS_V1/run_20261013_092358_3887540',
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    result = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError('Expected a JSON object: ' + str(path))
    return result


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def beneath(path, folder):
    path, folder = Path(path).resolve(), Path(folder).resolve()
    if folder not in path.parents:
        raise ValueError('Path must be beneath ' + str(folder) + ': ' + str(path))
    return path


def verify_release(folder=HERE):
    folder = Path(folder)
    manifest = folder / 'SHA256SUMS.json'
    hashes = read(manifest)
    if not hashes:
        raise ValueError('Empty release checksum manifest')
    for name, expected in hashes.items():
        if (not isinstance(name, str) or Path(name).name != name
                or not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected)
                or not (folder / name).is_file() or digest(folder / name) != expected):
            raise ValueError('Numeric-audit release checksum mismatch: ' + str(name))
    if not {path.name for path in folder.glob('*.py')} <= set(hashes):
        raise ValueError('Unbound Python source in numeric-audit release')
    return digest(manifest)


def inspect_recovery(root):
    """Bind the new numeric failure, not the original CUDA-assert snapshot."""
    root = Path(root).expanduser().resolve()
    debug = beneath(root / 'train/pool_debug', root)
    paths = {name: beneath(path, root) for name, path in (
        ('failure', debug / 'failure.json'), ('runtime', root / 'runtime.json'),
        ('evidence', root / 'evidence.json'),
    )}
    hashes = {str(path): digest(path) for path in paths.values()}
    failure, runtime, evidence = (read(paths[key]) for key in ('failure', 'runtime', 'evidence'))
    if failure.get('phase') != 'cpu_reference_gate':
        raise ValueError('This audit requires the captured cpu_reference_gate failure')
    image, call = failure.get('image'), failure.get('call')
    if not isinstance(image, str) or not image or type(call) is not int or call < 1:
        raise ValueError('Numeric failure lacks an explicit image and positive call number')
    filename = failure.get('input')
    if not isinstance(filename, str) or not filename:
        raise ValueError('Numeric failure has no saved input snapshot')
    snapshot = Path(filename).expanduser()
    if not snapshot.is_absolute():
        snapshot = debug / snapshot
    snapshot = beneath(snapshot, debug)
    if not snapshot.is_file():
        raise FileNotFoundError('Captured numeric-failure snapshot missing: ' + str(snapshot))
    hashes[str(snapshot)] = digest(snapshot)
    if not runtime or runtime != evidence.get('runtime'):
        raise ValueError('Recovery runtime.json differs from its preserved evidence binding')
    protocol_runtime = evidence.get('protocol_info', {}).get('source_runtime_settings')
    if protocol_runtime is not None and protocol_runtime != runtime:
        raise ValueError('Recovery protocol runtime differs from evidence/runtime.json')
    arguments = evidence.get('arguments')
    if not isinstance(arguments, dict) or not re.fullmatch(r'cuda:\d+', str(arguments.get('device', ''))):
        raise ValueError('Preserved recovery arguments must name an explicit cuda:N device')
    check_preserved({'preserved_sha256': hashes})
    return dict(recovery_root=str(root), snapshot=str(snapshot), snapshot_sha256=hashes[str(snapshot)],
                failed_image=image, failed_call=call, failure=failure,
                runtime=runtime, arguments=arguments, preserved_sha256=hashes,
                original_training_run=evidence.get('failed_run'),
                checkpoint_step=evidence.get('checkpoint_step'),
                interpretation='One saved numeric discrepancy; original failure, protocol and training remain untouched')


def check_preserved(evidence):
    for path, expected in evidence['preserved_sha256'].items():
        if not Path(path).is_file() or digest(path) != expected:
            raise ValueError('Preserved numeric-failure evidence changed: ' + str(path))


def child_environment(runtime):
    """Restore recorded process knobs before worker imports initialize torch."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_LAUNCH_BLOCKING='1')
    recorded = runtime.get('environment', {})
    if not isinstance(recorded, dict):
        raise TypeError('Invalid saved runtime environment')
    for key in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_TF32_OVERRIDE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        value = recorded.get(key)
        if value is None:
            env.pop(key, None)
        elif isinstance(value, str):
            env[key] = value
        else:
            raise TypeError('Invalid saved environment value: ' + key)
    if 'CUDA_DEVICE_ORDER' in recorded:
        value = recorded['CUDA_DEVICE_ORDER']
        if value is None:
            env.pop('CUDA_DEVICE_ORDER', None)
        elif isinstance(value, str):
            env['CUDA_DEVICE_ORDER'] = value
        else:
            raise TypeError('Invalid saved environment value: CUDA_DEVICE_ORDER')
    workspace = runtime.get('cublas_workspace_config')
    if workspace is None:
        env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    elif isinstance(workspace, str):
        env['CUBLAS_WORKSPACE_CONFIG'] = workspace
    else:
        raise TypeError('Invalid saved CUBLAS_WORKSPACE_CONFIG')
    return env


@contextmanager
def selected_gpu_environment(env):
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


def tail(path, count=30):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 65536))
        lines = stream.read().decode('utf-8', errors='replace').splitlines()
    return list(deque(lines, maxlen=count))


def _stage_rows(stages):
    if isinstance(stages, dict):
        return [(str(name), row) for name, row in stages.items() if isinstance(row, dict)]
    if isinstance(stages, list):
        return [(str(row.get('name', index)), row) for index, row in enumerate(stages)
                if isinstance(row, dict)]
    return []


def collect_sar_status(roots):
    """Read only recorded states/logs; a state file alone does not prove liveness."""
    results = []
    for value in roots:
        root = Path(value).expanduser().resolve()
        path = root / 'workflow_state.json'
        item = dict(root=str(root), state_file=str(path), exists=path.is_file(),
                    process_liveness_checked=False, read_only=True, failed_stages=[])
        if not path.is_file():
            item['status'] = 'state_missing'
            results.append(item)
            continue
        try:
            state = read(path)
            item.update(state_sha256=digest(path), state=state,
                        status='complete' if state.get('complete') is True else
                        ('failed' if state.get('failure') or state.get('error') else
                         state.get('current_stage', 'incomplete')))
            for name, row in _stage_rows(state.get('stages')):
                code = row.get('returncode')
                if (row.get('status') not in ('failed', 'error') and not row.get('failure')
                        and not row.get('error') and not (type(code) is int and code != 0)):
                    continue
                stage = dict(name=name, record=row)
                candidates = []
                explicit = row.get('log') or row.get('log_file')
                if isinstance(explicit, str):
                    candidates.append(Path(explicit) if Path(explicit).is_absolute() else root / explicit)
                candidates.append(root / 'logs' / (name + '.log'))
                for candidate in candidates:
                    try:
                        candidate = beneath(candidate, root)
                    except ValueError as error:
                        stage.setdefault('log_errors', []).append(str(error))
                        continue
                    if candidate.is_file():
                        stage.update(log=str(candidate), log_sha256=digest(candidate), log_tail=tail(candidate))
                        break
                item['failed_stages'].append(stage)
        except (OSError, ValueError, TypeError) as error:
            item.update(status='state_read_failed', error=str(error))
        results.append(item)
    return dict(read_only=True, process_liveness_checked=False, runs=results,
                interpretation='Recorded SAR state and confined failed-stage log tails; no restart and no live-process claim')


def run_child(command, log, env):
    """Each invocation creates a clean worker process; never reuse CUDA state."""
    with Path(log).open('x', encoding='utf-8') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=HERE)
        started = time.monotonic()
        while True:
            try:
                return child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                latest = tail(log, 1)
                print('RUNNING ' + Path(log).name + ' elapsed_seconds=' + str(int(time.monotonic() - started))
                      + (' last_line=' + latest[0][:320] if latest else ''), flush=True)


def numeric_passed(record):
    def verdicts_in(value):
        if isinstance(value, dict):
            if type(value.get('passed')) is bool:
                yield value['passed']
            else:
                for nested in value.values():
                    yield from verdicts_in(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from verdicts_in(nested)
    verdicts = list(verdicts_in(record.get('comparisons', {})))
    verdicts += list(verdicts_in(record.get('repeats', [])))
    return all(verdicts) if verdicts else None


def _comparison_lines(label, comparison):
    if not isinstance(comparison, dict) or not comparison:
        return [label + ': unavailable']
    lines = [label + ': passed=' + str(comparison.get('passed'))]
    for name in ('tokens', 'mass', 'reconstruction'):
        value = comparison.get(name)
        if not isinstance(value, dict):
            continue
        outside = value.get('elements_outside_tolerance', value.get('outside_tolerance'))
        lines.append('  ' + name + ': passed=' + str(value.get('passed'))
                     + '; max_abs=' + str(value.get('max_abs'))
                     + '; outside=' + str(outside) + '/' + str(value.get('total_elements'))
                     + '; finite=' + str(value.get('finite')))
    return lines


def render_report(root, state, sar_status=None):
    """Compact pasteable evidence; never emit per-element examples or all tensors."""
    root = Path(root)
    lines = [FORMAT, '结果目录: ' + str(root),
             '失败图片: ' + str(state.get('image')) + '; pool call=' + str(state.get('call')),
             '当前阶段: ' + str(state.get('current_stage')),
             '调度结束 execution_complete=' + str(state.get('execution_complete')),
             '诊断证据完整 diagnostic_complete=' + str(state.get('diagnostic_complete')),
             '所有比较均通过 all_numeric_passed=' + str(state.get('all_numeric_passed')),
             'training_started=false; training_gate_passed=false; Test=false',
             '原训练门槛和2e-5容差未修改；本包不启动或恢复训练。',
             'CPU-only只提供部分证据，不能判断CUDA表现。' if state.get('arguments', {}).get('cpu_only') else
             'CUDA后端使用独立进程；等待GPU空闲只是观察，不是独占预留。']
    if state.get('failure'):
        lines.append('父调度错误: ' + str(state['failure']))
    cpu = state.get('stages', {}).get('cpu', {})
    cpu_record = cpu.get('summary', {})
    lines += ['', 'CPU控制: status=' + str(cpu.get('status'))]
    comparisons = cpu_record.get('comparisons', {})
    for key, title in (
        ('cpu64_scatter_vs_grouped', 'FP64 scatter 对独立FP64分组参考'),
        ('cpu32_vs_cpu64', '原CPU FP32 对独立FP64参考'),
    ):
        lines += _comparison_lines(title, comparisons.get(key))
    for name in ('native', 'contiguous'):
        stage = state.get('stages', {}).get(name)
        if not stage:
            lines += ['', name + ': 未运行']
            continue
        record = stage.get('summary', {})
        lines += ['', name + ': status=' + str(stage.get('status'))
                  + '; execution_complete=' + str(stage.get('execution_complete'))
                  + '; finite_outputs=' + str(stage.get('finite_outputs')),
                  '日志: ' + str(stage.get('log'))]
        if record.get('failure') or stage.get('error'):
            lines.append('执行错误: ' + str(record.get('failure') or stage.get('error'))[:1000])
        for row in record.get('repeats', []):
            lines.append('repeat=' + str(row.get('repeat')))
            for key, title in (('original_cpu32_gate', '原CPU32训练门槛'),
                               ('vs_cpu64', '对独立FP64参考'), ('vs_first', '对本后端第一次结果')):
                lines += _comparison_lines(title, row.get(key))
            for key, title in (('cpu32_token_detail', 'CPU32 token定位'), ('cpu64_token_detail', 'FP64 token定位')):
                detail = row.get(key)
                if not isinstance(detail, dict):
                    continue
                selected = {field: detail.get(field) for field in (
                    'distinct_tokens_outside_tolerance', 'valid_token_elements_outside',
                    'invalid_token_elements_outside', 'validity_disagreements',
                    'saved_input_is_raw_and_fp32_square', 'mean_channels_outside', 'second_moment_channels_outside')
                    if field in detail}
                lines.append(title + ': ' + json.dumps(selected, ensure_ascii=False))
                if isinstance(detail.get('derived_std'), dict):
                    lines.append('派生std诊断: ' + json.dumps(detail['derived_std'], ensure_ascii=False))
    aggregate = state.get('stages', {}).get('compare', {}).get('summary', {})
    lines += ['', 'CUDA后端直接比较:']
    for key in ('contiguous_vs_native', 'native_vs_contiguous'):
        lines += _comparison_lines(key, aggregate.get('comparisons', {}).get(key))
    lines.append('解释: ' + str(aggregate.get('interpretation', '比较未完成；不能确定数值原因。')))
    lines.append('缺失证据: ' + str(aggregate.get('missing_workers', state.get('missing_workers', []))))
    lines += ['', 'SAR状态只读记录（未检查进程存活）:']
    for item in (sar_status or {}).get('runs', []):
        lines.append(str(item.get('root')) + ': ' + str(item.get('status')))
        saved = item.get('state', {})
        lines.append('  current_stage=' + str(saved.get('current_stage')) + '; failure='
                     + str(saved.get('failure') or saved.get('error'))[:1000])
        for failed in item.get('failed_stages', []):
            lines.append('  失败阶段=' + str(failed.get('name')) + '; log=' + str(failed.get('log')))
            lines += ['    ' + line[:800] for line in failed.get('log_tail', [])[-8:]]
    lines += ['', '当前结果只诊断这一个已保存输入；数值相近不代表训练或Memory设计已验证。']
    return '\n'.join(lines) + '\n'


def write_report(root, state):
    path = Path(root) / 'sar_status.json'
    sar = read(path) if path.is_file() else {}
    destination = Path(root) / 'report.txt'
    destination.write_text(render_report(root, state, sar), encoding='utf-8')
    return destination


def report(root):
    path = Path(root) / 'summary.json'
    state = read(path)
    print('AUDIT_SUMMARY=' + str(path), flush=True)
    print('execution_complete=', state.get('execution_complete'),
          '; diagnostic_complete=', state.get('diagnostic_complete'),
          '; all_numeric_passed=', state.get('all_numeric_passed'), flush=True)
    for name, row in state.get('stages', {}).items():
        print(name, row.get('status'), 'numeric_passed=', row.get('numeric_passed'),
              'log=', row.get('log'), flush=True)
    print('PASTEABLE_REPORT=' + str(Path(root) / 'report.txt'), flush=True)
    print('DIAGNOSTIC ONLY. No training or Test. No numeric tolerance changed. No training gate passed.', flush=True)
    return state


def run(args):
    release = verify_release(HERE)
    if not 1 <= args.repeats <= 10:
        raise ValueError('Use 1..10 worker repeats')
    evidence = inspect_recovery(args.recovery_root)
    env = child_environment(evidence['runtime'])
    parent = Path(args.output_parent).expanduser().resolve()
    output = (Path(args.output).expanduser().resolve() if args.output else parent / (
        'run_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '_' + str(os.getpid())))
    output = beneath(output, parent)
    protected = [Path(evidence['recovery_root']), HERE]
    if evidence.get('original_training_run'):
        protected.append(Path(evidence['original_training_run']).resolve())
    for forbidden in protected:
        if output == forbidden or forbidden in output.parents:
            raise ValueError('Audit output must be outside preserved recovery/package directories')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    write(output / 'source_binding.json', evidence)
    write(output / 'runtime.json', evidence['runtime'])
    runtime_copy_sha256 = digest(output / 'runtime.json')
    write(output / 'sar_status.json', collect_sar_status(args.sar_roots))
    state = dict(format=FORMAT, diagnostic_only=True, execution_complete=False,
                 diagnostic_complete=False, all_numeric_passed=None, training_gate_passed=False,
                 training_started=False, training=False, test_accessed=False,
                 full_validation_evidence=False, current_stage='cpu', stages={}, failure=None,
                 recovery_root=evidence['recovery_root'], snapshot=evidence['snapshot'],
                 snapshot_sha256=evidence['snapshot_sha256'], image=evidence['failed_image'],
                 call=evidence['failed_call'], release_sha256=release, arguments=vars(args),
                 tolerance_intervention=False, exact_historical_resume_claim=False)
    write(output / 'summary.json', state)
    write_report(output, state)
    print('OUTPUT_ROOT=' + str(output), flush=True)

    def stage(backend):
        check_preserved(evidence)
        if digest(output / 'runtime.json') != runtime_copy_sha256:
            raise ValueError('Bound worker runtime copy changed during audit')
        destination = output / backend
        log = output / 'logs' / (backend + '.log')
        command = [sys.executable, '-u', str(HERE / 'worker.py'), '--backend', backend,
                   '--input', evidence['snapshot'], '--output', str(destination),
                   '--runtime-json', str(output / 'runtime.json'),
                   '--expected-image', evidence['failed_image'], '--expected-call', str(evidence['failed_call']),
                   '--expected-sha256', evidence['snapshot_sha256'], '--repeats', str(args.repeats),
                   '--device', evidence['arguments']['device']]
        if backend == 'compare':
            command += ['--audit-root', str(output)]
        row = dict(status='running', command=command, output=str(destination), log=str(log),
                   diagnostic_only=True, training_gate_passed=False)
        state['current_stage'] = backend
        state['stages'][backend] = row
        write(output / 'summary.json', state)
        print('START ' + backend + ' log=' + str(log), flush=True)
        try:
            code = run_child(command, log, env)
            record = read(destination / 'summary.json') if (destination / 'summary.json').is_file() else {}
            execution_ok = code == 0 and record.get('execution_complete') is True and not record.get('failure')
            if backend != 'compare' and execution_ok:
                if (record.get('image') != evidence['failed_image'] or record.get('call') != evidence['failed_call']
                        or record.get('input_sha256') != evidence['snapshot_sha256']):
                    raise ValueError('Worker summary identity differs from captured failure: ' + backend)
            passed = numeric_passed(record)
            row.update(returncode=code, execution_complete=execution_ok,
                       finite_outputs=record.get('finite_outputs') is True,
                       numeric_passed=passed, summary=record,
                       status=('execution_failed' if not execution_ok else
                               ('numeric_disagreement' if passed is False else 'complete')))
        except Exception as error:
            row.update(status='execution_failed', execution_complete=False, numeric_passed=None,
                       error=type(error).__name__ + ': ' + str(error))
        check_preserved(evidence)
        if digest(output / 'runtime.json') != runtime_copy_sha256:
            raise ValueError('Bound worker runtime copy changed during audit')
        if row['status'] == 'execution_failed':
            row['log_tail'] = tail(log)
            detail = row.get('error') or row.get('summary', {}).get('failure') or row.get('summary', {}).get('error')
            print('STAGE_DETAIL ' + backend + ' ' + str(detail), flush=True)
            for line in row['log_tail']:
                print('LOG_TAIL ' + backend + ' ' + line[:800], flush=True)
        elif row['status'] == 'numeric_disagreement':
            print('NUMERIC_DISAGREEMENT ' + backend + '; expected diagnostic evidence; original gate unchanged; '
                  'report=' + str(output / 'report.txt'), flush=True)
        write(output / 'summary.json', state)
        write_report(output, state)
        print('END ' + backend + ' status=' + row['status'] + '; numeric_passed=' + str(row['numeric_passed']), flush=True)
        return row

    try:
        cpu = stage('cpu')
        if not cpu['execution_complete'] or not cpu.get('finite_outputs'):
            raise RuntimeError('CPU input/reference execution failed; no CUDA worker started')
        if not args.cpu_only:
            state['current_stage'] = 'waiting_gpu'
            write(output / 'summary.json', state)
            from gpu_queue import wait_for_idle
            device_index = int(evidence['arguments']['device'].split(':', 1)[1])
            with selected_gpu_environment(env):
                state['gpu_idle_observation'] = wait_for_idle(device_index, log=lambda line: print(line, flush=True),
                                                              poll_seconds=30, consecutive=2)
            write(output / 'summary.json', state)
            for backend in ('native', 'contiguous'):
                # A failed native worker never prevents isolated candidate evidence.
                stage(backend)
        comparison = stage('compare')
        check_preserved(evidence)
        if verify_release(HERE) != release:
            raise ValueError('Numeric-audit release changed while running')
        aggregate = comparison.get('summary', {})
        observed = [row.get('numeric_passed') for row in state['stages'].values()]
        all_numeric = (False if False in observed else
                       (True if comparison['execution_complete'] and aggregate.get('diagnostic_complete') is True
                        and all(value is True for value in observed) else None))
        state.update(execution_complete=True, current_stage='complete',
                     diagnostic_complete=comparison['execution_complete'] and aggregate.get('diagnostic_complete') is True,
                     all_numeric_passed=all_numeric,
                     all_workers_executed_successfully=all(row.get('execution_complete') is True
                                                         for row in state['stages'].values()),
                     source_evidence_unchanged=True,
                     missing_workers=aggregate.get('missing_workers', []),
                     interpretation='Numeric disagreements are observations, not relaxed training gates; no automatic restart')
        write(output / 'sar_status.json', collect_sar_status(args.sar_roots))
        write(output / 'summary.json', state)
        write_report(output, state)
        report(output)
        return state
    except Exception as error:
        state.update(failure=type(error).__name__ + ': ' + str(error), current_stage='failed',
                     execution_complete=False, diagnostic_complete=False, all_numeric_passed=None)
        write(output / 'summary.json', state)
        write_report(output, state)
        (output / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('AUDIT FAILED: ' + state['failure'] + '; read ' + str(output / 'summary.json'), flush=True)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--recovery-root', default=RECOVERY)
    result.add_argument('--output-parent', default=RESULT_BASE)
    result.add_argument('--output', help='NEW directory beneath --output-parent; default creates a unique root')
    result.add_argument('--cpu-only', action='store_true', help='Run only CPU references and available comparisons; no GPU wait')
    result.add_argument('--repeats', type=int, default=3)
    result.add_argument('--sar-roots', nargs=2, default=list(SAR_ROOTS), metavar=('Q_ROOT', 'SOFT_ROOT'))
    result.add_argument('--report', metavar='AUDIT_ROOT', help='Read one existing report only')
    return result


if __name__ == '__main__':
    arguments = parser().parse_args()
    if arguments.report:
        report(arguments.report)
    else:
        run(arguments)
