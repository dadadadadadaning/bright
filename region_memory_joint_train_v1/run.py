#!/usr/bin/env python3
"""Standalone server entry: checks -> real GPU smoke -> matched joint training."""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
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
    for command in ('preflight', 'full', 'smoke', 'train', 'evaluate'):
        s = sub.add_parser(command)
        s.add_argument('--repo', default=DEFAULT_REPO)
        s.add_argument('--v3-root', default=DEFAULT_V3)
        s.add_argument('--protocol', help='Source E0 protocol; auto-read from V3 pipeline_state when omitted')
        s.add_argument('--device', default='cuda:0')
        s.add_argument('--workers', type=int, default=4)
        s.add_argument('--seed', type=int, default=3407)
        s.add_argument('--topk', type=int, choices=(8, 64), default=64)
        s.add_argument('--chunk-size', type=int, default=64)
        s.add_argument('--lr', type=float, default=1e-4)
        s.add_argument('--weight-decay', type=float, default=5e-3)
        if command != 'preflight':
            s.add_argument('--output', required=True, help='NEW result directory; never overwrite an existing run')
        if command == 'evaluate':
            s.add_argument('--checkpoint', required=True)
    r = sub.add_parser('report')
    r.add_argument('--root', required=True)
    return p


def verify_release():
    path = HERE / 'SHA256SUMS.json'
    if not path.is_file():
        raise ValueError('Missing SHA256SUMS.json; use the complete unmodified training package')
    hashes = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Invalid package checksum manifest')
    for name, expected in hashes.items():
        if Path(name).name != name or not (HERE / name).is_file() or digest(HERE / name) != expected:
            raise ValueError('Training package checksum mismatch: ' + str(name))
    expected_code = {p.name for p in HERE.glob('*.py')}
    if not expected_code <= set(hashes):
        raise ValueError('Package contains unbound Python source files')
    return {'package_sha256': digest(path), 'code_sha256': source_hash(), 'files': len(hashes)}


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def preflight(args, *, run_tests=True):
    import torch
    release = verify_release()
    repo = Path(args.repo).resolve()
    if not (repo / 'bda_benchmark').is_dir():
        raise ValueError('BRIGHT repository missing: ' + str(repo))
    if str(repo) not in sys.path:
        sys.path.insert(1, str(repo))
    v3 = repo / 'bda_benchmark/experiments/damageformer_srac_memory_rebuild_v3'
    required = ('shared.py', 'core.py', 'paired.py')
    if not all((v3 / name).is_file() for name in required):
        raise ValueError('Existing V3 source is missing from this repository; this package does not replace it: ' + str(v3))
    keys = Path(args.v3_root).resolve() / 'keys'
    for name in ('paired_bank.pt', 'manifest.json', 'summary.json', 'protocol.json', 'images.jsonl'):
        if not (keys / name).is_file():
            raise ValueError('Formal paired bank file missing: ' + str(keys / name))
    manifest, summary = read_json(keys / 'manifest.json'), read_json(keys / 'summary.json')
    for data in (manifest, summary):
        if data.get('complete') is not True or data.get('smoke') is not False:
            raise ValueError('Requires the completed full paired bank, not keys_smoke or a failed bank')
    if args.workers < 0 or args.chunk_size < 1 or args.lr <= 0 or args.weight_decay < 0:
        raise ValueError('Invalid workers/chunk size/optimizer configuration')
    if args.device != 'cuda:0' and not args.device.startswith('cuda:'):
        raise ValueError('Real BRIGHT training requires the CUDA device bound by the source E0 protocol')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment; CPU synthetic tests are not real BRIGHT training')
    # Import the actual installed interfaces; do not patch/copy the V3 source.
    imported = {}
    for module, names in (
        ('damageformer_srac_memory_rebuild_v3.shared', ('load_bundle', 'load_paired_bank', 'verify_all')),
        ('damageformer_srac_memory_rebuild_v3.paired', ('extract_pair',)),
        ('damageformer_srac_memory_rebuild_v3.core', ('select_neighbors',)),
        ('damageformer_srac_memory_metric_eval_v2.native', ('native_grid_probq', 'NativeA1LogitsCapture')),
    ):
        m = importlib.import_module('bda_benchmark.experiments.' + module)
        if not all(callable(getattr(m, name, None)) for name in names):
            raise ValueError('Installed V3 interface differs: ' + module)
        imported[module] = str(Path(m.__file__).resolve())
    if run_tests:
        print('CPU synthetic contracts: no real BRIGHT/CUDA experiment is run by these tests.', flush=True)
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE),
                                 '-p', 'test_*.py', '-q'], cwd=HERE)
        if result.returncode:
            raise RuntimeError('CPU contracts failed; GPU smoke/training not started')
    device = torch.device(args.device)
    free, total = torch.cuda.mem_get_info(device)
    record = dict(format=FORMAT, stage='preflight', complete=True,
                  real_bright_forward=False, full_validation_evidence=False,
                  release=release, repo=str(repo), v3_root=str(Path(args.v3_root).resolve()),
                  torch=torch.__version__, device=str(device), gpu=torch.cuda.get_device_name(device),
                  free_vram_gib=free/(1024**3), total_vram_gib=total/(1024**3),
                  imported=imported, test_accessed=False,
                  previous_diagnose_summary=read_json(Path(args.v3_root) / 'diagnose/summary.json'),
                  interpretation='Files/interfaces/CPU contracts checked. Strict bank loading and true CUDA forward occur in smoke.')
    print('PREFLIGHT:', json.dumps({k: record[k] for k in ('gpu', 'free_vram_gib', 'total_vram_gib', 'torch')}, ensure_ascii=False), flush=True)
    return record


def forwarded_args(args):
    result = []
    for name in ('repo', 'v3_root', 'device', 'workers', 'seed', 'topk', 'chunk_size', 'lr', 'weight_decay', 'protocol'):
        value = getattr(args, name, None)
        if value is not None:
            result.extend(['--' + name.replace('_', '-'), str(value)])
    return result


def report(root):
    root = Path(root)
    run = root / 'train' if (root / 'train').is_dir() else root
    state = read_json(root / 'workflow_state.json')
    summary = read_json(run / 'summary.json')
    print('结果目录：', root)
    if state is not None:
        print('workflow:', json.dumps(state, ensure_ascii=False, indent=2))
    if summary is None:
        print('尚无训练 summary.json。')
        return
    fields = ('complete', 'training_complete', 'full_validation_evidence', 'smoke', 'phase',
              'updates_per_arm', 'error', 'last_checkpoint', 'elapsed_seconds')
    print('training:', json.dumps({k: summary[k] for k in fields if k in summary}, ensure_ascii=False, indent=2))
    # Both a partial summary and complete results can be inspected read-only.
    final = read_json(run / 'final_val/result.json') or read_json(run / 'val/result.json')
    if final is None:
        print('尚无 final_val/result.json；先查看 traceback.txt 和训练日志。')
        return
    print('末步 Val 指标（百分比）：')
    metrics = final.get('metrics', {})
    for name, row in metrics.items():
        value = row.get('valid_miou')
        if value is not None:
            iou = [None if x is None else round(100*x, 4) for x in row.get('iou', [])]
            print(f'{name:12s} foreground_mIoU={100*value:.4f}; IoU={iou}')
    for a, b in (('r0', 'grid_probq'), ('rt', 'r0'), ('rf', 'r0'), ('rf', 'rt')):
        if a in metrics and b in metrics:
            av, bv = metrics[a].get('valid_miou'), metrics[b].get('valid_miou')
            if av is not None and bv is not None:
                print(f'{a}-{b}: {100*(av-bv):+.4f} pp')
    print('formal:', summary.get('full_validation_evidence') is True,
          '; smoke:', summary.get('smoke'), '; Test not run.')


def full(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(format=FORMAT, complete=False, full_validation_evidence=False,
                 training_complete=False, test_accessed=False, current_stage='preflight', stages={},
                 arguments=vars(args), code_sha256=source_hash())
    write_json(output / 'workflow_state.json', state)
    started = time.perf_counter()
    try:
        write_json(output / 'preflight.json', preflight(args))
        state['stages']['preflight'] = {'status': 'complete'}
        (output / 'logs').mkdir()
        for stage in ('smoke', 'train'):
            command = [sys.executable, '-u', str(HERE / 'run.py'), stage,
                       *forwarded_args(args), '--output', str(output / stage)]
            log = output / 'logs' / (stage + '.log')
            state['current_stage'] = stage
            state['stages'][stage] = {'status': 'running', 'command': command, 'log': str(log)}
            write_json(output / 'workflow_state.json', state)
            print('START', stage, 'OUTPUT=', output / stage, flush=True)
            with log.open('x', encoding='utf-8') as stream:
                process = subprocess.Popen(command, cwd=Path(args.repo), stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                state['stages'][stage]['pid'] = process.pid
                write_json(output / 'workflow_state.json', state)
                assert process.stdout is not None
                for line in process.stdout:
                    stream.write(line); stream.flush()
                    print(line, end='', flush=True)
                returncode = process.wait()
            state['stages'][stage]['returncode'] = returncode
            state['stages'][stage]['log_sha256'] = digest(log)
            write_json(output / 'workflow_state.json', state)
            if returncode:
                state['stages'][stage]['status'] = 'failed'
                raise RuntimeError(f'{stage} failed with returncode {returncode}; later stages stopped; read {log}')
            stage_summary = read_json(output / stage / 'summary.json') or {}
            if stage_summary.get('complete') is not True:
                raise RuntimeError(f'{stage} child exited without a complete summary; inspect preserved stage artifacts')
            if stage == 'smoke' and stage_summary.get('smoke') is not True:
                raise RuntimeError('Smoke child has inconsistent experiment identity')
            if stage == 'train' and stage_summary.get('full_validation_evidence') is not True:
                raise RuntimeError('Training child has not passed final formal Val450 checks')
            state['stages'][stage]['status'] = 'complete'
            state['stages'][stage]['summary_sha256'] = digest(output / stage / 'summary.json')
            write_json(output / 'workflow_state.json', state)
        state.update(complete=True, training_complete=True, full_validation_evidence=True,
                     current_stage='complete', elapsed_seconds=time.perf_counter()-started)
        write_json(output / 'workflow_state.json', state)
        report(output)
    except BaseException as exc:
        row = state['stages'].get(state['current_stage'])
        if row is not None and row.get('status') != 'complete':
            row['status'] = 'failed'
        state.update(complete=False, full_validation_evidence=False,
                     error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.perf_counter()-started)
        saved = read_json(output / 'train/summary.json') or {}
        state['training_complete'] = saved.get('training_complete') is True
        write_json(output / 'workflow_state.json', state)
        (output / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'report':
        return report(args.root)
    # Unmodified package is required before any real-data mutation/output.
    verify_release()
    if str(Path(args.repo).resolve()) not in sys.path:
        sys.path.insert(1, str(Path(args.repo).resolve()))
    if args.command == 'preflight':
        return preflight(args)
    if args.command == 'full':
        return full(args)
    from train import run_training, run_evaluation
    if args.command == 'evaluate':
        return run_evaluation(args)
    args.mode = args.command
    args.steps = 20 if args.mode == 'smoke' else 5000
    args.eval_every = 20 if args.mode == 'smoke' else 500
    return run_training(args)


if __name__ == '__main__':
    main()
