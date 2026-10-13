#!/usr/bin/env python3
"""Capture exact frozen-state changes around the existing SAR Q trainer.

This is instrumentation, not a repaired model or a checkpoint certification.
Original training, numerical settings and completion checks remain in force.
Default action only inspects files/processes. Probe runs 20 smoke updates in a
fresh directory after observing GPU idle; it cannot certify a 2000-step run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

REPO = '/root/shared-nvme/BRIGHT-master'
OLD = '/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_Q_MATCHED_V1/run_20261013_082551_3863093'
TOOL = REPO + '/New/sar_q_s3iW28Cy/sar_q_matched_v1'
PROTOCOL = '/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34/fresh_e0/protocol.json'
MODULE = 'bda_benchmark.experiments.damageformer_srac_q_factorial_v1.train_q'


def write_json(path, record):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def clone_state(module):
    return {name: value.detach().to('cpu').contiguous().clone()
            for name, value in module.state_dict().items()}


def state_fingerprint(state):
    """Identical byte stream to the original module_fingerprint function."""
    h = hashlib.sha256()
    for name, tensor in state.items():
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def differences(before, after):
    import torch
    rows = []
    for name in sorted(set(before) | set(after)):
        a, b = before.get(name), after.get(name)
        if a is None or b is None:
            rows.append(dict(name=name, reason='missing_or_added_key'))
            continue
        if a.dtype != b.dtype or a.shape != b.shape:
            rows.append(dict(name=name, reason='dtype_or_shape_changed'))
            continue
        aa = a.reshape(-1).view(torch.uint8)
        bb = b.reshape(-1).view(torch.uint8)
        count = int((aa != bb).sum())
        if count:
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            # Byte comparison includes signed zeros and NaN payloads.
            maximum = float((a.double() - b.double()).abs().max()) if finite and a.numel() else None
            rows.append(dict(name=name, reason='bytes_changed', changed_bytes=count,
                             shape=list(a.shape), dtype=str(a.dtype), finite=finite,
                             max_abs=maximum))
    if not rows and list(before) != list(after):
        rows.append(dict(name=None, reason='state_dict_order_only'))
    return rows


class FrozenWitness:
    def __init__(self, root, original_fingerprint, model, interval=20):
        self.root = Path(root)
        self.original_fingerprint = original_fingerprint
        self.model = model
        self.interval = interval
        self.snapshots = {}
        self.names = {id(model.base): 'a1', id(model.adapter): 'adapter',
                      id(model.embedding): 'embedding', id(model.region_classifier): 'region_classifier'}
        self.step = 0
        self.last_verified_step = 0
        self.phase = 'initialization'
        self.optimizer = None
        self.handles = []

    def fingerprint(self, module):
        name = self.names.get(id(module))
        if name is None:
            return self.original_fingerprint(module)
        state = clone_state(module)
        actual = state_fingerprint(state)
        # Never substitute a saved hash for the current hash.
        independently_read = self.original_fingerprint(module)
        if independently_read != actual:
            self.fail(name, state, 'unstable_consecutive_state_reads', second_sha256=independently_read)
        if name not in self.snapshots:
            self.snapshots[name] = (module, state, actual)
            parameter_names = set(dict(module.named_parameters()))
            write_json(self.root / (name + '_initial_tensors.json'), {
                key: dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                          kind='parameter' if key in parameter_names else 'buffer',
                          sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
                for key, tensor in state.items()})
            self.handles.append(module.register_forward_pre_hook(self._pre_forward))
            self.record('baseline_captured', name=name, sha256=actual,
                        parameters=sum(p.numel() for p in module.parameters()))
        elif actual != self.snapshots[name][2]:
            self.fail(name, state, 'frozen_state_changed')
        return actual

    def _pre_forward(self, module, _inputs):
        # Inspect modes, never silently change them.
        bad_modes = [name for name, child in module.named_modules() if child.training]
        bad_grads = [name for name, p in module.named_parameters() if p.requires_grad]
        if bad_modes or bad_grads:
            name = self.names[id(module)]
            self.fail(name, clone_state(module), 'frozen_mode_or_gradient_contract',
                      bad_modes=bad_modes, bad_grads=bad_grads)

    def record(self, event, **details):
        row = dict(event=event, step=self.step, phase=self.phase, time=time.time(), **details)
        with (self.root / 'witness.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')

    def check_all(self, phase):
        self.phase = phase
        for module, _state, _sha in list(self.snapshots.values()):
            self.fingerprint(module)
        self.last_verified_step = self.step
        self.record('all_frozen_states_exact')
        print('FROZEN_EXACT step=%d phase=%s' % (self.step, phase), flush=True)

    def fail(self, name, after, reason, **details):
        import torch
        baseline = self.snapshots.get(name)
        before = baseline[1] if baseline else {}
        rows = differences(before, after)
        parameter_names = set(dict(baseline[0].named_parameters())) if baseline else set()
        for row in rows:
            row['kind'] = 'parameter' if row['name'] in parameter_names else 'buffer_or_metadata'
        report = dict(format='SAR_FROZEN_FAILURE_V1', group=name, reason=reason,
                      step=self.step, phase=self.phase, differences=rows,
                      last_all_groups_exact_step=self.last_verified_step,
                      step_is_detection_point_not_exact_mutation_time=True,
                      before_sha256=baseline[2] if baseline else None,
                      after_sha256=state_fingerprint(after),
                      training_certified=False, test_accessed=False, **details)
        # Save the evidence before propagating failure. No old files are touched.
        names = {row['name'] for row in rows if row['name'] is not None}
        if reason == 'unstable_consecutive_state_reads':
            names = set(before) | set(after)
        torch.save(dict(report=report,
                        before={k: before[k] for k in names if k in before},
                        after={k: after[k] for k in names if k in after}),
                   self.root / 'frozen_change.pth')
        write_json(self.root / 'failure.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        raise RuntimeError('Frozen witness failed: %s/%s; see %s' % (name, reason, self.root / 'failure.json'))

    def close(self):
        for handle in self.handles:
            handle.remove()


def inspect(args):
    root = Path(args.old_root)
    print('READ ONLY. No model forward or training.', flush=True)
    for name in ('workflow_state.json', 'formal/train_baseline/summary.json'):
        p = root / name
        if p.is_file():
            value = json.loads(p.read_text())
            keys = ('format', 'complete', 'current_stage', 'phase', 'failure', 'error')
            print(name, json.dumps({k: value.get(k) for k in keys}, ensure_ascii=False), flush=True)
    p = root / 'formal/train_baseline/last.pth'
    if p.is_file():
        import torch
        value = torch.load(p, map_location='cpu', weights_only=False)
        keys = ('step', 'run_complete', 'completed_steps', 'a1_unchanged',
                'downstream_unchanged', 'full_validation_evidence', 'valid_miou',
                'a1_checkpoint', 'a1_sha256', 'source_hash', 'arguments')
        print('CHECKPOINT', json.dumps({k: value.get(k) for k in keys}, ensure_ascii=False), flush=True)
        print('BASE_SNAPSHOT_PRESENT', any(k in value for k in ('base', 'model', 'frozen_a1_state')), flush=True)
        print('CHECKPOINT_SHA256', file_sha(p), flush=True)
    for command in (['nvidia-smi'], ['ps', '-eo', 'pid,ppid,etime,args']):
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        if command[0] == 'ps':
            lines = [line for line in result.stdout.splitlines()
                     if any(s in line for s in ('sar_q_', 'sar_soft_', 'joint_live', 'server_worker', '4041006'))]
            print('\n'.join(lines), flush=True)
        else:
            print(result.stdout or result.stderr, flush=True)


def child(args, extra):
    import torch
    sys.path.insert(0, args.repo)
    module = importlib.import_module(MODULE)
    train_args = module.parser().parse_args(extra)
    evidence = Path(args.evidence)
    evidence.mkdir(parents=True, exist_ok=False)
    write_json(evidence / 'declaration.json', dict(
        format='SAR_FROZEN_WITNESS_V1', script_sha256=file_sha(__file__),
        original_train_sha256=file_sha(module.__file__), arguments=vars(train_args),
        check_interval=20, original_checks_enabled=True,
        changes_training_math=False, diagnostic_only=bool(train_args.smoke),
        old_checkpoint_reused=False, test_accessed=False))
    original_class = module.QVariantFront
    original_fp = module.module_fingerprint
    original_adam = torch.optim.AdamW
    original_eval = module.evaluate_front
    witness = None

    class ObservedFront(original_class):
        def __init__(self, *a, **kw):
            nonlocal witness
            super().__init__(*a, **kw)
            witness = FrozenWitness(evidence, original_fp, self)

    class ObservedAdam(original_adam):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            witness.optimizer = self
            witness.phase = 'Train'

        def step(self, closure=None):
            value = super().step(closure)
            witness.step += 1
            if witness.step == 1 or witness.step % witness.interval == 0:
                witness.check_all('after_optimizer_step')
            witness.phase = 'Train'
            return value

    def observed_eval(*a, **kw):
        witness.check_all('before_Val')
        value = original_eval(*a, **kw)
        witness.check_all('after_Val')
        return value

    module.QVariantFront = ObservedFront
    module.module_fingerprint = lambda m: witness.fingerprint(m)
    module.evaluate_front = observed_eval
    torch.optim.AdamW = ObservedAdam
    try:
        result = module.run(train_args)
        witness.check_all('original_training_checks_passed')
        write_json(evidence / 'summary.json', dict(complete=True, exact_frozen_checks=True,
            updates=witness.step, smoke=train_args.smoke, original_summary=result,
            diagnostic_only=bool(train_args.smoke), test_accessed=False))
    except BaseException as exc:
        write_json(evidence / 'summary.json', dict(complete=False,
            updates=witness.step if witness else 0, error=repr(exc),
            failure_file=str(evidence / 'failure.json'), test_accessed=False))
        (evidence / 'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        if witness:
            witness.close()
        module.QVariantFront = original_class
        module.module_fingerprint = original_fp
        module.evaluate_front = original_eval
        torch.optim.AdamW = original_adam


def declared_environment(protocol):
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    settings = protocol['runtime_settings']
    for name in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_TF32_OVERRIDE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        value = settings['environment'][name]
        if value is None:
            env.pop(name, None)
        elif isinstance(value, str):
            env[name] = value
        else:
            raise ValueError('Invalid protocol environment: ' + name)
    workspace = settings.get('cublas_workspace_config')
    if workspace is None:
        env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    elif isinstance(workspace, str):
        env['CUBLAS_WORKSPACE_CONFIG'] = workspace
    else:
        raise ValueError('Invalid CUBLAS workspace configuration')
    return env


def probe(args):
    import fcntl
    parent = Path(args.output_parent)
    parent.mkdir(parents=True, exist_ok=True)
    # Only coordinates this diagnostic, never locks/kills unrelated experiments.
    with (parent / '.probe.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A SAR freeze probe is already queued/running in this output parent')
        return _probe_locked(args)


def _probe_locked(args):
    # No torch/model import in this parent before the existing GPU queue clears.
    sys.path.insert(0, args.tool)
    queue = importlib.import_module('gpu_queue')
    if Path(queue.__file__).resolve() != Path(args.tool).resolve() / 'gpu_queue.py':
        raise RuntimeError('Wrong GPU queue module')
    protocol = json.loads(Path(args.protocol).read_text())
    env = declared_environment(protocol)
    device = protocol['arguments']['device']
    if not device.startswith('cuda:') or not device[5:].isdigit():
        raise ValueError('Expected an explicit CUDA device in the protocol')
    # The queue must resolve the same visibility that the child will use.
    visible_before = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        if 'CUDA_VISIBLE_DEVICES' in env:
            os.environ['CUDA_VISIBLE_DEVICES'] = env['CUDA_VISIBLE_DEVICES']
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        queue.wait_for_idle(int(device[5:]), log=lambda s: print(s, flush=True))
    finally:
        if visible_before is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = visible_before
    parent = Path(args.output_parent)
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='probe_', dir=str(parent)))
    print('OUTPUT_ROOT=' + str(root), flush=True)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), 'child',
               '--repo', args.repo, '--evidence', str(root / 'witness'), '--',
               '--protocol', args.protocol, '--arm', 'baseline', '--output', str(root / 'fit'),
               '--steps', '20', '--eval_every', '20', '--batch', '2', '--accum', '8',
               '--workers', '4', '--lr', '0.0001', '--aux_weight', '0.1', '--cue_gain', '0.1',
               '--amp', 'bf16', '--smoke']
    with (root / 'run.log').open('x') as f:
        process = subprocess.Popen(command, cwd=args.repo, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                f.write(line)
                f.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    print('PROBE_EXIT_CODE=' + str(code), flush=True)
    print('20-step diagnostic only; no old checkpoint certified; no formal pipeline launched.', flush=True)
    return code


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('action', choices=('inspect', 'probe', 'child'), nargs='?', default='inspect')
    p.add_argument('--repo', default=REPO)
    p.add_argument('--old-root', default=OLD)
    p.add_argument('--tool', default=TOOL)
    p.add_argument('--protocol', default=PROTOCOL)
    p.add_argument('--output-parent', default='/root/shared-nvme/BRIGHT-checkpoints/SRAC_FROZEN_WITNESS_V1')
    p.add_argument('--evidence')
    args, extra = p.parse_known_args(argv)
    if args.action == 'child':
        if not args.evidence:
            p.error('child requires --evidence')
        child(args, extra[1:] if extra[:1] == ['--'] else extra)
    else:
        if extra:
            p.error('Unknown arguments: ' + ' '.join(extra))
        return inspect(args) if args.action == 'inspect' else probe(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
