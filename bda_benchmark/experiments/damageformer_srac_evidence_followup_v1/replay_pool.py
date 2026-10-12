"""Replay one PRE-kernel CPU snapshot in a NEW process, without BRIGHT or training.

Use a different nonexistent output directory for every backend invocation.
Do not run native and grouped CUDA replays sequentially in a process that has
already raised an illegal-memory-access exception.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from .native_pool import (_ORIGINAL_POOL, compare_results, cpu_reference, grouped_pool,
                          validate_cpu_snapshot)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='Trusted PoolGuard CPU .pth input snapshot')
    p.add_argument('--output', required=True, help='New nonexistent replay directory')
    p.add_argument('--backend', choices=('cpu', 'native', 'grouped'), required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--layout', choices=('captured', 'contiguous'), default='captured',
                   help='Replay captured strides/offsets; contiguous is a separate explicit layout intervention')
    p.add_argument('--repeats', type=int, default=3)
    return p


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def restore_layout(tensor, layout, device):
    """Recreate captured positive/zero strides and offset without original storage.

    Broadcast views are populated through a non-broadcast slice. Other internal
    overlap is rejected by copy_ rather than silently choosing a new layout.
    Input checkpoints must be trusted; a malicious stride can request huge storage.
    """
    shape = tuple(layout['shape'])
    stride = tuple(layout['stride'])
    offset = int(layout['storage_offset'])
    if (shape != tuple(tensor.shape) or len(shape) != len(stride)
            or any(s < 0 for s in stride) or offset < 0):
        raise ValueError('Invalid captured layout')
    storage_size = offset + 1 + sum((dim-1)*step for dim, step in zip(shape, stride))
    storage = torch.empty(storage_size, dtype=tensor.dtype, device=device)
    view = storage.as_strided(shape, stride, offset)
    source = tensor.to(device)
    destination = view
    for dim, step in enumerate(stride):
        if step == 0 and shape[dim] > 1:
            destination = destination.narrow(dim, 0, 1)
            source = source.narrow(dim, 0, 1)
    destination.copy_(source)
    if not torch.equal(view.detach().cpu(), tensor):
        raise ValueError('Captured layout reconstruction changed input values')
    return view


@torch.no_grad()
def run(args):
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    source, output = Path(args.input), Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError('Preserve old results; choose a new output directory')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Tensor/dict-only capture format. Never load an untrusted checkpoint.
    snapshot = torch.load(source, map_location='cpu', weights_only=True)
    if snapshot.get('format') != 'native_pool_input_v1':
        raise ValueError('Not a PoolGuard input snapshot')
    check = validate_cpu_snapshot(snapshot)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'input': str(source.resolve()), 'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'image': snapshot.get('image'), 'call': snapshot.get('call'),
                'backend': args.backend, 'device': 'cpu' if args.backend == 'cpu' else args.device,
                'repeats': args.repeats, 'torch_version': torch.__version__,
                'cuda_version': torch.version.cuda, 'input_check': check,
                'source_layout': snapshot.get('layout'),
                'replay_layout': args.layout,
                'training': False, 'test_accessed': False, 'full_validation_evidence': False}
    save(output/'manifest.json', manifest)
    results = []
    summary = {'complete': False, 'gates_passed': False, 'backend': args.backend,
               'training': False, 'test_accessed': False, 'full_validation_evidence': False,
               'repeats': results, 'failure': None}
    phase = 'cpu_reference'
    try:
        reference = cpu_reference(snapshot)
        device = torch.device('cpu' if args.backend == 'cpu' else args.device)
        tensors = [restore_layout(snapshot[key], snapshot['layout'][key], device)
                   if args.layout == 'captured' else snapshot[key].to(device)
                   for key in ('content', 'weights', 'ids')]
        first = None
        for repeat in range(args.repeats):
            phase = 'pool_'+args.backend
            if args.backend == 'cpu':
                result = _ORIGINAL_POOL(*tensors, snapshot['token_count'])
            elif args.backend == 'native':
                result = _ORIGINAL_POOL(*tensors, snapshot['token_count'])
            else:
                result = grouped_pool(*tensors, snapshot['token_count'])
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            phase = 'comparison'
            host = {key: value.detach().cpu().clone() for key, value in result.items()}
            if first is None:
                first = host
            row = {'repeat': repeat+1, 'vs_cpu': compare_results(host, reference),
                   'vs_first': compare_results(host, first)}
            results.append(row)
            save(output/'summary.json', summary)
            print(json.dumps(row), flush=True)
            if not row['vs_cpu']['passed'] or not row['vs_first']['passed']:
                raise ValueError('Fixed CPU-reference/repeatability tolerance failed; no tolerance relaxation')
        summary.update(complete=True, gates_passed=True)
    except Exception as error:
        summary['failure'] = {'phase': phase, 'type': type(error).__name__, 'message': str(error)}
        save(output/'summary.json', summary)
        raise
    save(output/'summary.json', summary)
    print('POOL REPLAY COMPLETE. One captured input only; full Val still required. No training.', flush=True)
    return summary


if __name__ == '__main__':
    run(parser().parse_args())
