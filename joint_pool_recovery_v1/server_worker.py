"""CUDA worker for a separately bound repaired continuation of the old heads."""
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import random
from types import SimpleNamespace

from recovery import (check_preserved, digest, read, source_hash, verify_release,
                      HERE, inspect_failed)


def state_hash(state):
    result = hashlib.sha256()
    for name, value in state.items():
        tensor = value.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(tensor.dtype).encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(tensor.view(__import__('torch').uint8).numpy().tobytes())
    return result.hexdigest()


def validate_loaded(saved, evidence):
    import torch
    import numpy as np
    arms = {'r0', 'rt', 'rf'}
    if (saved.get('format') != 'SRAC_REGION_MEMORY_JOINT_TRAIN_V1'
            or saved.get('mode') != 'train' or saved.get('smoke') is not False
            or saved.get('training_complete') is not False):
        raise ValueError('Checkpoint is not an incomplete formal joint checkpoint')
    if saved.get('config') != evidence['arguments']:
        raise ValueError('Checkpoint configuration differs from the original manifest')
    if (saved.get('source_hash') != evidence['original_source_hash']
            or saved.get('protocol_info') != evidence['protocol_info']):
        raise ValueError('Checkpoint source/bank/runtime binding differs')
    step = evidence['checkpoint_step']
    if saved.get('step') != step or saved.get('updates_per_arm') != {a: step for a in arms}:
        raise ValueError('Checkpoint updates differ across arms')
    if set(saved.get('heads', {})) != arms or set(saved.get('optimizers', {})) != arms:
        raise ValueError('Checkpoint is missing a head or optimizer')
    if saved.get('initial_head_hashes') != evidence['initial_head_hashes']:
        raise ValueError('Checkpoint initial-head binding differs')
    for arm in arms:
        if state_hash(saved['heads'][arm]) != evidence['sidecar']['head_hashes'][arm]:
            raise ValueError('Saved head checksum mismatch: ' + arm)
        if any(not bool(torch.isfinite(t).all()) for t in saved['heads'][arm].values()):
            raise ValueError('Nonfinite checkpoint head: ' + arm)
        optimizer = saved['optimizers'][arm]
        if not optimizer.get('state') or not optimizer.get('param_groups'):
            raise ValueError('Checkpoint optimizer state is empty: ' + arm)
        parameter_ids = []
        for group in optimizer['param_groups']:
            ids = group.get('params')
            if not isinstance(ids, list) or not ids or any(type(p) is not int for p in ids):
                raise ValueError('Invalid optimizer parameter IDs: ' + arm)
            parameter_ids.extend(ids)
        if len(set(parameter_ids)) != len(parameter_ids) or not set(optimizer['state']) <= set(parameter_ids):
            raise ValueError('Optimizer state/parameter mapping differs: ' + arm)
        for parameter in optimizer['state'].values():
            if not {'step', 'exp_avg', 'exp_avg_sq'} <= set(parameter):
                raise ValueError('Incomplete AdamW state: ' + arm)
            if (not torch.is_tensor(parameter['exp_avg']) or not torch.is_tensor(parameter['exp_avg_sq'])
                    or parameter['exp_avg'].shape != parameter['exp_avg_sq'].shape):
                raise ValueError('Inconsistent AdamW moment shapes: ' + arm)
            count = parameter['step']
            count = float(count.item()) if torch.is_tensor(count) and count.numel() == 1 else float(count)
            if not 0 < count <= step or count != int(count):
                raise ValueError('Invalid AdamW update count: ' + arm)
            for value in parameter.values():
                if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                    raise ValueError('Nonfinite optimizer state: ' + arm)
    if set(saved.get('rng', {})) != {'python', 'numpy', 'torch_cpu', 'torch_cuda'}:
        raise ValueError('Checkpoint RNG is incomplete')
    rng = saved['rng']
    try:
        random.Random().setstate(rng['python'])
        np.random.RandomState().set_state(rng['numpy'])
        torch.Generator(device='cpu').set_state(rng['torch_cpu'])
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError('Invalid checkpoint RNG state: ' + str(error)) from error
    if not isinstance(rng['torch_cuda'], list) or any(
            not torch.is_tensor(t) or t.device.type != 'cpu' or t.dtype != torch.uint8 or t.ndim != 1
            for t in rng['torch_cuda']):
        raise ValueError('Invalid CUDA RNG state schema')
    ledger = saved.get('input_file_sha256')
    if not isinstance(ledger, dict) or not ledger:
        raise ValueError('Checkpoint dataset fingerprint ledger is missing')
    if any(not isinstance(path, str) or not Path(path).is_absolute()
           or not isinstance(value, str) or len(value) != 64
           or any(c not in '0123456789abcdef' for c in value) for path, value in ledger.items()):
        raise ValueError('Invalid dataset fingerprint ledger schema')


def run(args):
    verify_release(HERE)
    evidence = read(args.evidence)
    check_preserved(evidence)
    current = inspect_failed(evidence['failed_run'], evidence['tool'])
    for field in ('arguments', 'protocol_info', 'checkpoint', 'checkpoint_step',
                  'original_source_hash', 'preserved_sha256'):
        if current[field] != evidence[field]:
            raise ValueError('Failure binding changed before the CUDA worker: ' + field)
    if source_hash(HERE) != evidence['repair_source_hash']:
        raise ValueError('Repair source binding changed')
    import torch
    # Trusted local checkpoint; its sidecar checksum was checked before loading.
    saved = torch.load(evidence['checkpoint'], map_location='cpu', weights_only=False)
    validate_loaded(saved, evidence)
    tool, repo = Path(evidence['tool']).resolve(), Path(evidence['repo']).resolve()
    sys.path.insert(0, str(tool))
    sys.path.insert(1, str(repo))
    original_train = importlib.import_module('train')
    original_server = importlib.import_module('server_runtime')
    for name in ('train', 'server_runtime', 'common', 'model'):
        if Path(sys.modules[name].__file__).resolve().parent != tool:
            raise ValueError('Unexpected original module import: ' + name)
    if original_train.source_hash() != evidence['original_source_hash']:
        raise ValueError('Loaded original source differs')
    pool = importlib.import_module('bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool')
    from pool_repair import patched_pool_guard
    from recover_training import continue_training
    for name in ('pool_repair', 'pool_ops', 'recover_training'):
        if Path(sys.modules[name].__file__).resolve().parent != HERE:
            raise ValueError('Unexpected repair module import: ' + name)
    binding = dict(format='JOINT_CONTIGUOUS_INDEX_RECOVERY_V1',
                   backend='native_contiguous_indices_candidate',
                   repair_source_hash=evidence['repair_source_hash'],
                   repair_release_sha256=evidence['repair_release_sha256'],
                   pool_ops_sha256=digest(HERE / 'pool_ops.py'),
                   pool_repair_sha256=digest(HERE / 'pool_repair.py'),
                   original_checkpoint_sha256=evidence['sidecar']['sha256'],
                   original_history_sha256=digest(evidence['history']),
                   fixed_reference_tolerance={'atol': 2e-5, 'rtol': 2e-5},
                   exact_historical_resume_claim=False, cuda_launch_blocking='1')

    class RecoveryRuntime(original_server.Runtime):
        def __enter__(self):
            try:
                super().__enter__()
                if self.protocol_info != saved['protocol_info']:
                    raise ValueError('Current E0/V3/feature configuration differs from saved checkpoint')
                ledger = saved['input_file_sha256']
                required = set()
                for name in self.train_names + self.val_names:
                    required.update(str(Path(p).resolve()) for p in self.deps.sample_paths(
                        self.bundle['args'].dataset, name).values())
                if not required <= {str(Path(p).resolve()) for p in ledger}:
                    raise ValueError('Saved fingerprint ledger does not cover complete Train2918 and Val450')
                for index, (path, expected) in enumerate(ledger.items(), 1):
                    if digest(path) != expected:
                        raise ValueError('Saved dataset fingerprint changed: ' + path)
                    self.input_ledger[Path(path).resolve()] = expected
                    if index % 1000 == 0 or index == len(ledger):
                        print('VERIFY_SAVED_INPUT_FILES %d/%d' % (index, len(ledger)), flush=True)
                return self
            except BaseException:
                self.close(*sys.exc_info())
                raise

    config = copy.deepcopy(evidence['arguments'])
    config['output'] = str(Path(args.output).resolve())
    history = [json.loads(line) for line in Path(evidence['history']).read_text(encoding='utf-8').splitlines()]
    with patched_pool_guard(pool):
        summary = continue_training(original_train, SimpleNamespace(**config), RecoveryRuntime,
                                    saved, history, binding)
    check_preserved(evidence)
    if source_hash(HERE) != evidence['repair_source_hash']:
        raise ValueError('Repair code changed during execution')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
