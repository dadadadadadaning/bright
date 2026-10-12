"""Replay captured single-image pool inputs in separate fresh processes.

One CUDA failure cannot poison the next backend's process. No model training,
fallback, retries, input edits or tolerance changes are performed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture_run', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--backends', nargs='+', choices=('cpu', 'native', 'grouped'), default=['cpu', 'native', 'grouped'])
    return p


def run(args):
    source = Path(args.capture_run).resolve()
    pool = source/'pool_debug' if (source/'pool_debug').is_dir() else source
    inputs = sorted(pool.glob('input_*.pth'))
    if not 1 <= len(inputs) <= 32:
        raise ValueError('Requires 1–32 captures from a single-image run; do not replay a full dataset here')
    if len(set(args.backends)) != len(args.backends) or args.repeats < 1:
        raise ValueError('Choose unique backends and a positive repeat count')
    output = Path(args.output).resolve()
    if output == source or source in output.parents or output.exists():
        raise ValueError('Choose a new replay output outside the captured run')
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, CUDA_LAUNCH_BLOCKING='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    summary = dict(complete=False, gates_passed=False, full_validation_evidence=False,
                   training=False, test_accessed=False, captures=len(inputs), results=[])
    def save():
        (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    save()
    for path in inputs:
        for backend in args.backends:
            stem = path.stem+'_'+backend
            result_dir = output/stem
            cmd = [sys.executable, '-u', '-m', __package__+'.replay_pool',
                   '--input', str(path), '--output', str(result_dir), '--backend', backend,
                   '--device', args.device, '--repeats', str(args.repeats)]
            with (output/(stem+'.log')).open('x', encoding='utf-8') as log:
                code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=environment).returncode
            result = {}
            if (result_dir/'summary.json').is_file():
                result = json.loads((result_dir/'summary.json').read_text())
            row = dict(input=str(path), backend=backend, returncode=code,
                       passed=code == 0 and result.get('gates_passed') is True,
                       result=str(result_dir), log=str(output/(stem+'.log')))
            summary['results'].append(row)
            save()
            print(json.dumps(row), flush=True)
    summary['complete'] = True  # All requested independent tests ran, not all passed.
    summary['backend_passed'] = {backend: all(r['passed'] for r in summary['results'] if r['backend'] == backend)
                                 for backend in args.backends}
    summary['gates_passed'] = all(summary['backend_passed'].values())
    save()
    print('Backend results:', summary['backend_passed'], flush=True)
    print('Single-image pool tests only. A failed native backend remains failed; no full-Val claim.', flush=True)
    return summary


if __name__ == '__main__':
    result = run(parser().parse_args())
    raise SystemExit(0 if result['gates_passed'] else 1)
