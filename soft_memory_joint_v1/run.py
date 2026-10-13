#!/usr/bin/env python3
"""Read-only planning, guarded paired-value building and matched A/B/C/D runs."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
import traceback

from common import FORMAT, digest, source_hash, write_json

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = '/root/shared-nvme/BRIGHT-master'
DEFAULT_V3 = '/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_REBUILD_V3/run_20261013_100220_049144_3778665'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('plan', 'preflight', 'find-baseline', 'baseline-check', 'check', 'full',
                    'build', 'smoke', 'train', 'evaluate'):
        s = sub.add_parser(command)
        s.add_argument('--repo', default=DEFAULT_REPO)
        s.add_argument('--v3-root', default=DEFAULT_V3)
        s.add_argument('--protocol', help='Source E0 protocol; read from source V3 when omitted')
        s.add_argument('--baseline-root', help='Completed original R0/RT/RF result; auto-discover if omitted')
        s.add_argument('--device', default='cuda:0')
        s.add_argument('--workers', type=int, default=4)
        s.add_argument('--seed', type=int, default=3407)
        s.add_argument('--key-name', choices=('t128', 'f256'), default='t128')
        s.add_argument('--topk', type=int, choices=(64,), default=64,
                       help='64 parent cases; at most 128 child slots, not 128 independent cases')
        s.add_argument('--chunk-size', type=int, default=32)
        s.add_argument('--lr', type=float, default=1e-4)
        s.add_argument('--weight-decay', type=float, default=5e-3)
        if command in ('check', 'full', 'build', 'smoke', 'train', 'evaluate'):
            s.add_argument('--output', required=True, help='New directory; existing outputs are preserved')
        if command in ('smoke', 'train', 'evaluate'):
            s.add_argument('--values', required=True, help='Completed child-value bank built by this package')
            s.add_argument('--allow-smoke-values', action='store_true',
                           help='Allow a small bank for smoke/checkpoint-smoke only; never formal training')
        if command == 'build':
            s.add_argument('--limit-images', type=int, choices=(0, 4), default=0,
                           help='0=full fixed support inventory; 4=smoke source selection')
        if command == 'evaluate':
            s.add_argument('--checkpoint', required=True)
    report_parser = sub.add_parser('report')
    report_parser.add_argument('--root', required=True)
    return p


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object: ' + str(path))
    return value


def verify_release():
    """Bind all Python entry points, including tests; reject escapes and symlinks."""
    manifest = HERE / 'SHA256SUMS.json'
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError('Missing SHA256SUMS.json; obtain the complete unmodified package')
    hashes = read_json(manifest)
    if not hashes:
        raise ValueError('Empty package checksum manifest')
    for name, expected in hashes.items():
        relative = PurePosixPath(name)
        if (not isinstance(name, str) or '\\' in name or relative.is_absolute()
                or not relative.parts or any(part in ('.', '..') for part in relative.parts)
                or relative.as_posix() != name or name == 'SHA256SUMS.json'):
            raise ValueError('Invalid relative package path: ' + str(name))
        target = HERE.joinpath(*relative.parts)
        if (any(parent.is_symlink() for parent in [target, *target.parents] if parent != HERE.parent)
                or not target.is_file() or not target.resolve().is_relative_to(HERE.resolve())
                or not isinstance(expected, str) or re.fullmatch(r'[0-9a-f]{64}', expected) is None
                or digest(target) != expected):
            raise ValueError('Package checksum mismatch: ' + name)
    code = {p.relative_to(HERE).as_posix() for p in HERE.rglob('*.py')}
    if not code <= set(hashes):
        raise ValueError('Unbound Python files in package: ' + ', '.join(sorted(code - set(hashes))))
    return {'manifest_sha256': digest(manifest), 'code_sha256': source_hash(), 'files': len(hashes)}


def _settings(args):
    if (args.workers < 0 or args.chunk_size < 1 or args.topk != 64
            or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        raise ValueError('Invalid workers/chunk-size/Top64/optimizer configuration')
    if not re.fullmatch(r'cuda:\d+', args.device):
        raise ValueError('Real BRIGHT execution requires an explicit CUDA device, e.g. cuda:0')
    if args.command in ('train', 'full') and getattr(args, 'allow_smoke_values', False):
        raise ValueError('Formal training cannot use --allow-smoke-values')


def check_inputs(args):
    """Read-only checks; strict model/bank hash replay happens in real stages."""
    _settings(args)
    repo = Path(args.repo).expanduser().resolve()
    if not (repo / 'bda_benchmark').is_dir():
        raise FileNotFoundError('BRIGHT repository missing: ' + str(repo))
    source = repo / 'bda_benchmark/experiments/damageformer_srac_memory_rebuild_v3'
    for name in ('shared.py', 'paired.py', 'core.py'):
        if not (source / name).is_file():
            raise FileNotFoundError('Installed V3 source missing: ' + str(source / name))
    keys = Path(args.v3_root).expanduser().resolve() / 'keys'
    for name in ('paired_bank.pt', 'manifest.json', 'summary.json', 'protocol.json', 'images.jsonl'):
        if not (keys / name).is_file():
            raise FileNotFoundError('Formal paired bank file missing: ' + str(keys / name))
    for name in ('manifest.json', 'summary.json'):
        data = read_json(keys / name)
        if data.get('complete') is not True or data.get('smoke') is not False:
            raise ValueError('Source bank must be complete and full, not keys_smoke: ' + str(keys / name))
    from base_runtime import source_protocol
    protocol = source_protocol(args)
    args.protocol = str(protocol)
    if getattr(args, 'values', None):
        value_summary = read_json(Path(args.values) / 'summary.json')
        if value_summary is None or value_summary.get('complete') is not True:
            raise ValueError('Child-value bank has no complete summary: ' + str(args.values))
        if args.command == 'train' and value_summary.get('smoke') is not False:
            raise ValueError('Formal training requires a full child-value bank')
    return {'repo': str(repo), 'v3_root': str(keys.parent), 'protocol': str(protocol),
            'source_protocol_sha256': digest(protocol), 'strict_bank_loading_pending': True}


def validate_start(args):
    """Foreground launcher checks: no training, output directory or GPU process."""
    record = {'release': verify_release(), 'inputs': check_inputs(args)}
    if args.command in ('full', 'train'):
        from gates import validate_baseline
        record['baseline'] = validate_baseline(args)
    return record


def preflight(args, *, run_tests=True):
    record = validate_start(args)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; CPU synthetic tests do not establish real BRIGHT execution')
    repo = Path(args.repo).expanduser().resolve()
    if str(repo) not in sys.path:
        sys.path.insert(1, str(repo))
    imports = {}
    for module, names in (
        ('damageformer_srac_memory_rebuild_v3.shared', ('load_bundle', 'load_paired_bank', 'verify_all')),
        ('damageformer_srac_memory_rebuild_v3.paired', ('extract_pair',)),
        ('damageformer_srac_memory_rebuild_v3.core', ('select_neighbors',)),
        ('damageformer_srac_memory_metric_eval_v2.native', ('native_grid_probq', 'NativeA1LogitsCapture')),
    ):
        loaded = importlib.import_module('bda_benchmark.experiments.' + module)
        if not all(callable(getattr(loaded, name, None)) for name in names):
            raise ValueError('Installed V3 interface differs: ' + module)
        filename = Path(loaded.__file__).resolve()
        if not filename.is_relative_to(repo):
            raise ValueError('Imported V3 source is outside requested repository: ' + str(filename))
        imports[module] = str(filename)
    if run_tests:
        print('CPU synthetic contracts only; these tests are not BRIGHT Train/Val/CUDA evidence.', flush=True)
        checked = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE),
                                  '-p', 'test_*.py', '-q'], cwd=HERE)
        if checked.returncode:
            raise RuntimeError('CPU contracts failed; no build/smoke/training was started')
    device = torch.device(args.device)
    free, total = torch.cuda.mem_get_info(device)
    record.update(format=FORMAT, stage='preflight', complete=True, real_bright_forward=False,
                  full_validation_evidence=False, training=False, test_accessed=False,
                  imported=imports, torch=torch.__version__, device=str(device),
                  gpu=torch.cuda.get_device_name(device), free_vram_gib=free/(1024**3),
                  total_vram_gib=total/(1024**3),
                  cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING'),
                  interpretation='Read-only inputs/interfaces/CPU contracts/capability checks; not a pooling fix or real-data validation.')
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    return record


def forwarded_args(args):
    result = []
    for name in ('repo', 'v3_root', 'protocol', 'baseline_root', 'device', 'workers', 'seed',
                 'key_name', 'topk', 'chunk_size', 'lr', 'weight_decay'):
        value = getattr(args, name, None)
        if value is not None:
            result.extend(['--' + name.replace('_', '-'), str(value)])
    return result


def plan(args):
    """No GPU import, source-bank loading, output mutation or baseline requirement."""
    result = {'format': FORMAT, 'execution': 'plan_only', 'read_only': True,
              'repo': args.repo, 'v3_root': args.v3_root, 'key_name': args.key_name,
              'topk_parents': args.topk, 'maximum_child_slots_per_parent': 2,
              'arms': {'a': 'spatial two-slot + within-image references',
                       'b': 'building-probability soft slots + within-image references',
                       'c': 'spatial two-slot + Train historical cases',
                       'd': 'building-probability soft slots + Train historical cases'},
              'check': ['preflight', 'build_smoke:4 source images', 'smoke:20 updates per arm'],
              'full': ['require completed original R0/RT/RF evidence', 'preflight',
                       'build_smoke', 'smoke:20', 'build:all fixed parent members',
                       'train:5000 updates per arm + saved-checkpoint Val450'],
              'baseline_root': args.baseline_root or 'auto-discover and strictly validate; no failed-run fallback',
              'existing_pooling_fault_fixed': False, 'test_accessed': False,
              'note': 'check success is smoke only. Original frozen frontend, parent keys and member identities stay fixed. No Memory rejection gate.'}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def find_baselines(args):
    from gates import find_baselines as find
    found = find(args)
    print('旧 R0/RT/RF 结果检查（只读）：')
    for row in found:
        print(json.dumps(row, ensure_ascii=False, indent=2))
    accepted = [row for row in found if row.get('accepted') is True]
    if not accepted:
        print('未找到通过验证的正式旧三组结果。可以运行 check；full/train 不会启动。')
    return found


def _safe_read(path):
    try:
        return read_json(path)
    except (OSError, ValueError) as exc:
        print('报告读取失败（保留原文件）：', path, str(exc))
        return None


def report(root):
    """Inspect partial/failed evidence without invoking launch or resume gates."""
    root = Path(root).expanduser().resolve()
    print('结果目录：', root)
    state = _safe_read(root / 'workflow_state.json')
    if state:
        print('workflow:', json.dumps(state, ensure_ascii=False, indent=2))
    for relative in ('values_smoke', 'smoke', 'values', 'train'):
        summary = _safe_read(root / relative / 'summary.json')
        if summary:
            fields = ('complete', 'execution_complete', 'full_validation_evidence', 'smoke',
                      'phase', 'updates_per_arm', 'error', 'image_count', 'last_checkpoint')
            print(relative + ':', json.dumps({k: summary[k] for k in fields if k in summary}, ensure_ascii=False))
    run = root / 'train' if (root / 'train').is_dir() else root
    if run == root and (root / 'smoke').is_dir():
        run = root / 'smoke'
    summary = _safe_read(run / 'summary.json') or {}
    if run == root and summary:
        fields = ('complete', 'execution_complete', 'full_validation_evidence', 'smoke',
                  'phase', 'updates_per_arm', 'error', 'image_count', 'last_checkpoint')
        print('summary:', json.dumps({k: summary[k] for k in fields if k in summary}, ensure_ascii=False))
    final = _safe_read(run / 'final_val/result.json') or _safe_read(run / 'val/result.json')
    if final is None:
        print('尚无末步验证报告；请读取保存的 traceback.txt 与 logs/，无需修改失败状态。')
        return {'workflow': state, 'summary': summary, 'final': None}
    metrics = final.get('metrics', {})
    print('Val 前景三类 mIoU，百分比单位；背景不计入该均值：')
    for name in ('grid_probq', 'a', 'b', 'c', 'd'):
        row = metrics.get(name, {})
        value = row.get('foreground_miou', row.get('valid_miou'))
        if value is not None:
            ious = row.get('iou', row.get('class_iou', row.get('class_ious', [])))
            print(f'{name}: {100*value:.4f}%; class_IoU={[None if x is None else round(100*x, 4) for x in ious]}')
    def value(name):
        row = metrics.get(name, {})
        return row.get('foreground_miou', row.get('valid_miou'))
    for left, right in (('b', 'a'), ('d', 'c'), ('c', 'a'), ('d', 'b')):
        if value(left) is not None and value(right) is not None:
            print(f'{left}-{right}: {100*(value(left)-value(right)):+.4f} pp')
    if all(value(name) is not None for name in ('a', 'b', 'c', 'd')):
        print(f'interaction d-c-b+a: {100*(value("d")-value("c")-value("b")+value("a")):+.4f} pp')
    formal = (summary.get('complete') is True and summary.get('full_validation_evidence') is True
              and summary.get('smoke') is False and final.get('full_validation_evidence') is True
              and final.get('final_integrity_pending') is False)
    print('正式完成证据：', formal, '；Test 未运行。' if formal else '；仅可按已保存的 smoke/中间/失败状态解读。')
    return {'workflow': state, 'summary': summary, 'final': final, 'formal': formal}


def _child(stage, command, log, state, output):
    state['current_stage'] = stage
    row = {'status': 'running', 'command': command, 'log': str(log)}
    state['stages'][stage] = row
    write_json(output / 'workflow_state.json', state)
    print('START', stage, flush=True)
    with log.open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=HERE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        row['pid'] = process.pid
        write_json(output / 'workflow_state.json', state)
        assert process.stdout is not None
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()
    row.update(returncode=code, log_sha256=digest(log))
    write_json(output / 'workflow_state.json', state)
    if code:
        row['status'] = 'failed'
        raise RuntimeError(f'{stage} failed with returncode {code}; later stages stopped; see {log}')
    return row


def pipeline(args):
    formal = args.command == 'full'
    # Check the prerequisite before producing a fresh apparent run.
    baseline = None
    if formal:
        from gates import validate_baseline
        baseline = validate_baseline(args)
        args.baseline_root = baseline['root']
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    state = {'format': FORMAT, 'complete': False, 'execution_complete': False,
             'full_validation_evidence': False, 'training_complete': False,
             'smoke': not formal, 'test_accessed': False, 'current_stage': 'preflight',
             'stages': {}, 'arguments': vars(args).copy(), 'code_sha256': source_hash(),
             'cuda_launch_blocking': os.environ.get('CUDA_LAUNCH_BLOCKING')}
    write_json(output / 'workflow_state.json', state)
    if baseline:
        write_json(output / 'baseline_evidence.json', baseline)
        state['baseline_evidence_sha256'] = digest(output / 'baseline_evidence.json')
    started = time.monotonic()
    try:
        shared = forwarded_args(args)
        stages = [('preflight', ['preflight', *shared], None, None),
                  ('build_smoke', ['build', *shared, '--limit-images', '4', '--output', str(output / 'values_smoke')], 'values_smoke', True),
                  ('smoke', ['smoke', *shared, '--values', str(output / 'values_smoke'), '--allow-smoke-values', '--output', str(output / 'smoke')], 'smoke', True)]
        if formal:
            stages.extend([
                ('build', ['build', *shared, '--limit-images', '0', '--output', str(output / 'values')], 'values', False),
                ('train', ['train', *shared, '--values', str(output / 'values'), '--output', str(output / 'train')], 'train', False)])
        for name, child_args, directory, smoke in stages:
            command = [sys.executable, '-u', str(HERE / 'run.py'), *child_args]
            row = _child(name, command, output / 'logs' / (name + '.log'), state, output)
            if directory is not None:
                path = output / directory / 'summary.json'
                summary = read_json(path) or {}
                if summary.get('complete') is not True or summary.get('smoke') is not smoke:
                    raise RuntimeError(name + ' exited without its completed matching full/smoke summary')
                if name == 'train' and summary.get('full_validation_evidence') is not True:
                    raise RuntimeError('Training did not pass final formal Val450 integrity checks')
                row['summary_sha256'] = digest(path)
            row['status'] = 'complete'
            write_json(output / 'workflow_state.json', state)
        if formal:
            revalidated = validate_baseline(args)
            if revalidated != baseline:
                raise RuntimeError('Original baseline evidence changed during the new experiment')
        state.update(complete=True, execution_complete=True, training_complete=formal,
                     full_validation_evidence=formal, current_stage='complete' if formal else 'complete_smoke',
                     elapsed_seconds=time.monotonic()-started)
        write_json(output / 'workflow_state.json', state)
        report(output)
        return state
    except BaseException as exc:
        row = state['stages'].get(state['current_stage'])
        if row is not None:
            row['status'] = 'failed'
        state.update(complete=False, execution_complete=False, full_validation_evidence=False,
                     error=type(exc).__name__ + ': ' + str(exc), elapsed_seconds=time.monotonic()-started)
        write_json(output / 'workflow_state.json', state)
        (output / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'report':
        return report(args.root)
    if args.command == 'plan':
        return plan(args)
    if args.command == 'find-baseline':
        return find_baselines(args)
    verify_release()
    if args.command == 'baseline-check':
        from gates import validate_baseline
        evidence = validate_baseline(args)
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
        return evidence
    os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')
    if args.command == 'preflight':
        return preflight(args)
    if args.command in ('check', 'full'):
        return pipeline(args)
    check_inputs(args)
    if args.command == 'build':
        from build_values import run_build
        return run_build(args)
    if args.command == 'train':
        from gates import validate_baseline
        args.baseline_evidence = validate_baseline(args)
    from train import run_training, run_evaluation
    if args.command == 'evaluate':
        return run_evaluation(args)
    args.mode = args.command
    args.smoke = args.mode == 'smoke'
    args.steps = 20 if args.mode == 'smoke' else 5000
    args.eval_every = 20 if args.mode == 'smoke' else 500
    return run_training(args)


if __name__ == '__main__':
    main()
