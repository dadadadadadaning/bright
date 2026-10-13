#!/usr/bin/env python3
"""Read failed joint evidence and run isolated, non-training pool diagnostics."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
DEFAULT_FAILED = '/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_MEMORY_JOINT_V1/run_20261013_143518_3825644'
DEFAULT_TOOL = '/root/shared-nvme/BRIGHT-master/New/joint_train_ZAtNI2Wi/region_memory_joint_train_v1'
DEFAULT_REPO = '/root/shared-nvme/BRIGHT-master'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def verify_release():
    hashes = read(HERE / 'SHA256SUMS.json')
    if not hashes:
        raise ValueError('Missing release manifest; download the complete ZIP')
    for name, expected in hashes.items():
        if Path(name).name != name or digest(HERE / name) != expected:
            raise ValueError('Diagnostic package checksum mismatch: ' + name)
    if not {p.name for p in HERE.glob('*.py')} <= set(hashes):
        raise ValueError('Unbound diagnostic Python source')
    return digest(HERE / 'SHA256SUMS.json')


def inspect_failed(root):
    """Read only this explicit failed run, never choose the newest run implicitly."""
    root = Path(root).resolve()
    records, hashes = {}, {}
    for name in ('workflow_state.json', 'smoke/summary.json', 'train/summary.json',
                 'train/manifest.json', 'train/pool_debug/failure.json',
                 'train/pool_debug/guard.json', 'train/pool_debug/progress.json',
                 'train/pool_debug/manifest.json'):
        path = root / name
        records[name] = read(path)
        if path.is_file():
            hashes[str(path)] = digest(path)
    manifest = records['train/manifest.json']
    if not manifest or not isinstance(manifest.get('protocol_info'), dict):
        raise ValueError('Missing train/manifest.json with bound source runtime')
    runtime = manifest['protocol_info'].get('source_runtime_settings')
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError('Historical runtime settings missing; will not guess or force deterministic mode')
    failure = records['train/pool_debug/failure.json'] or {}
    pool = root / 'train/pool_debug'
    saved = failure.get('input')
    candidates = []
    if saved:
        # The result directory may have moved, but the captured filename must
        # belong to this selected run. Never load an arbitrary external pickle.
        candidate = pool / Path(saved).name
        if candidate.is_file():
            candidates = [candidate.resolve()]
    if not candidates:
        candidates = sorted(p.resolve() for p in pool.glob('input_*.pth'))
    if len(candidates) != 1:
        raise ValueError('Need exactly the one failure snapshot; found %d in %s. Read failure.json first.' % (len(candidates), pool))
    captured = candidates[0]
    if pool.resolve() not in captured.parents:
        raise ValueError('Snapshot resolves outside selected failure directory')
    hashes[str(captured)] = digest(captured)
    return dict(failed_run=str(root), records=records, input=str(captured),
                historical_runtime=runtime, source_files_sha256=hashes,
                failed_image=failure.get('image'),
                statement='Historical evidence read only. No model/bank changes or training.')


def child_environment(runtime):
    env = os.environ.copy()
    env['CUDA_LAUNCH_BLOCKING'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    workspace = runtime.get('cublas_workspace_config')
    if workspace is None:
        env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    elif isinstance(workspace, str):
        env['CUBLAS_WORKSPACE_CONFIG'] = workspace
    else:
        raise ValueError('Invalid source cublas_workspace_config')
    return env


def verify_native_source(evidence):
    """Bind the standalone baseline to the actual server operator, not its name."""
    record = evidence['records'].get('train/pool_debug/manifest.json') or {}
    source = Path(record.get('original_source', ''))
    if not source.is_file() or digest(source) != record.get('original_sha256'):
        raise ValueError('Original pooling source is missing or differs from the failed guard manifest')
    def body(path, name):
        tree = ast.parse(Path(path).read_text(encoding='utf-8'))
        function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name == name), None)
        if function is None:
            raise ValueError('Missing pool function: ' + name)
        rows = list(function.body)
        if rows and isinstance(rows[0], ast.Expr) and isinstance(rows[0].value, ast.Constant) and isinstance(rows[0].value.value, str):
            rows = rows[1:]
        return ast.dump(ast.Module(body=rows, type_ignores=[]), include_attributes=False)
    if body(source, 'soft_pool_unpool') != body(HERE / 'pool_ops.py', 'native_original_algorithm'):
        raise ValueError('Server pool arithmetic differs from diagnostic baseline; refusing misleading replay')
    evidence['source_files_sha256'][str(source.resolve())] = digest(source)
    evidence['native_arithmetic_matches_server_ast'] = True


def run_child(command, destination, env):
    log = destination.with_suffix('.log')
    with log.open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, env=env, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end='', flush=True)
        code = process.wait()
    result = read(destination / 'summary.json') or {}
    return dict(returncode=code, passed=code == 0 and result.get('gates_passed') is True,
                log=str(log), output=str(destination), summary=result)


def run(args):
    output = Path(args.output).resolve()
    source = Path(args.failed_run).resolve()
    tool = Path(args.tool).resolve()
    if output.exists() or output == source or source in output.parents or output == tool or tool in output.parents:
        raise ValueError('Use a NEW output directory outside old evidence/tool directories')
    if args.repeats < 1 or args.repeats > 10:
        raise ValueError('Use 1--10 independent arithmetic repeats per worker')
    release = verify_release()
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(format='SRAC_JOINT_POOL_DIAGNOSE_V1', diagnostic_complete=False,
                   training=False, test_accessed=False, full_validation_evidence=False,
                   formal_training_completed=False, root_cause_confirmed=False,
                   runtime_intervention='CUDA_LAUNCH_BLOCKING=1 only; source numeric flags preserved',
                   release_sha256=release, arguments=vars(args), stages={})
    def save():
        write(output / 'summary.json', summary)
    save()
    try:
        evidence = inspect_failed(source)
        verify_native_source(evidence)
        write(output / 'historical_evidence.json', evidence)
        runtime_path = output / 'historical_runtime.json'
        write(runtime_path, evidence['historical_runtime'])
        env = child_environment(evidence['historical_runtime'])
        summary.update(failed_image=evidence['failed_image'], input=evidence['input'])
        print('原失败图片:', evidence['failed_image'], flush=True)
        print('原池化现场:', evidence['input'], flush=True)
        save()
        for backend in ('cpu', 'native', 'contiguous'):
            destination = output / ('replay_' + backend)
            command = [sys.executable, '-u', str(HERE / 'replay_worker.py'),
                       '--input', evidence['input'], '--output', str(destination),
                       '--backend', backend, '--device', args.device,
                       '--repeats', str(args.repeats), '--runtime-json', str(runtime_path)]
            print('START replay_' + backend, flush=True)
            summary['stages']['replay_' + backend] = run_child(command, destination, env)
            save()
            if backend == 'cpu' and not summary['stages']['replay_cpu']['passed']:
                raise RuntimeError('CPU snapshot/reference failed; CUDA tests not started. Inspect replay_cpu.log')
        # A CUDA child failure never contaminates the next worker's context.
        for scope in ('failed_image', 'val450'):
            if scope == 'val450' and args.no_sequence:
                summary['stages']['sequence_val450'] = {'skipped': True, 'reason': '--no-sequence'}
                continue
            if scope == 'val450' and not summary['stages']['sequence_failed_image']['passed']:
                summary['stages']['sequence_val450'] = {'skipped': True, 'reason': 'single image already failed; inspect its synchronous traceback'}
                continue
            destination = output / ('sequence_' + scope)
            command = [sys.executable, '-u', str(HERE / 'sequence_worker.py'),
                       '--tool', str(tool), '--repo', str(Path(args.repo).resolve()),
                       '--failed-run', str(source), '--output', str(destination), '--scope', scope]
            print('START sequence_' + scope + ' (no training)', flush=True)
            summary['stages']['sequence_' + scope] = run_child(command, destination, env)
            save()
        changed = [p for p, h in evidence['source_files_sha256'].items()
                   if not Path(p).is_file() or digest(p) != h]
        if changed:
            raise RuntimeError('Historical evidence changed during diagnosis: ' + repr(changed))
        summary.update(diagnostic_complete=True, historical_evidence_unchanged=True,
                       all_executed_stages_passed=all(row.get('passed') is True
                           for row in summary['stages'].values() if not row.get('skipped')),
                       interpretation='A passing contiguous candidate is isolated operator evidence, not a production fix. Original sequence uses unchanged operator. No automatic restart or formal training.')
        save()
        report(output)
        return summary
    except BaseException as error:
        summary.update(error=type(error).__name__ + ': ' + str(error))
        save()
        (output / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def report(root):
    data = read(Path(root) / 'summary.json')
    if data is None:
        raise FileNotFoundError('No diagnostic summary at ' + str(root))
    print('诊断报告:', Path(root) / 'summary.json')
    print('失败图片:', data.get('failed_image'))
    for name, row in data.get('stages', {}).items():
        print(name, 'SKIPPED' if row.get('skipped') else ('PASS' if row.get('passed') else 'FAIL'),
              row.get('reason', row.get('log', '')))
        detail = row.get('summary', {})
        if detail.get('error'):
            print('  原因:', detail['error'])
        for repeat in detail.get('repeats', []):
            comparison = repeat.get('vs_cpu', {})
            if comparison.get('passed') is False:
                failed = {key: value for key, value in comparison.items()
                          if isinstance(value, dict) and value.get('passed') is False}
                print('  CPU参考差异:', json.dumps(failed, ensure_ascii=False))
    print('诊断执行完成:', data.get('diagnostic_complete'), '；训练: false；Test: false')
    print('单算子/初始化检查通过不代表旧故障根因已确认，也不代表获得训练后Val450成绩。')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('all')
    s.add_argument('--failed-run', default=DEFAULT_FAILED)
    s.add_argument('--tool', default=DEFAULT_TOOL)
    s.add_argument('--repo', default=DEFAULT_REPO)
    s.add_argument('--output', required=True)
    s.add_argument('--device', default='cuda:0')
    s.add_argument('--repeats', type=int, default=3)
    s.add_argument('--no-sequence', action='store_true', help='Skip optional full initial Val450; failed-image test still runs')
    s = sub.add_parser('report')
    s.add_argument('--root', required=True)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.command == 'report':
        report(args.root)
    else:
        result = run(args)
        raise SystemExit(0 if result.get('all_executed_stages_passed') else 1)
