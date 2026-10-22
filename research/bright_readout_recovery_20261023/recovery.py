"""Pinned read_aux checkpoint800 recovery. New outputs; original recipe; no Test."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

FORMAT = 'BRIGHT_MEMORY_READOUT_DESIGN_V1'
ROOT = Path('/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_READOUT_DESIGN_V1')
PARENT = ROOT / 'read_aux_001_3407_night_01'
PACKAGE = Path('/root/shared-nvme/BRIGHT-master/New/bright_memory_readout_design_20261022')
CHECKPOINT_SHA = 'ed806287f8466a897f67ab18b853fd2b5efa40702e648138796f5e1e337ffa42'
IDENTITY = {
 'experiment.py': '4bbbbfd171b1b39b23e38b7ea7c3c07bcbabdf14a6dca133e92c2448340e71f9',
 'model_adapter.py': '9a81006d0191a3379145286d74a74f7d236d230534a980b0671ff3da4b1e8ba4',
 'supervision.py': 'a90c36512fbe84d70510c7c594c9b7d401554a9943209314f58e36810d5be35e',
}

def require(value, message):
    if not value:
        raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str)+'\n')
    temporary.replace(path)

def validate_prefix(rows, manifest, restored_step):
    plan = manifest['plan']
    require(len(rows) >= restored_step, 'Journal shorter than checkpoint')
    names = set(manifest['actual_training_names'])
    for expected, row in enumerate(rows[:restored_step], 1):
        require(row.get('step') == expected and row.get('optimizer_step') == expected,
                'Nonconsecutive journal/optimizer step')
        require(isinstance(row.get('epoch'), int) and row['epoch'] >= 0 and
                row.get('epoch_seed') == manifest['seed'] + row['epoch'], 'Journal epoch mismatch')
        require(row.get('variant') == plan['variant'] and row.get('image') in names,
                'Journal variant or training membership mismatch')
    return rows[:restored_step]

def read_parent(package, parent):
    package, parent = Path(package).resolve(), Path(parent).resolve()
    for name, digest in IDENTITY.items():
        require(sha(package / name) == digest, 'Unsupported installed version: '+name)
    plan, manifest, status = [read(parent / n) for n in ('plan.json', 'manifest.json', 'summary.json')]
    require(manifest['plan'] == plan, 'Parent manifest/plan mismatch')
    expected = dict(format=FORMAT, variant='read_aux', effective_mode='read_aux', seed=3407,
                    endpoint_step=2500, read_weight=.01, extension_files=IDENTITY,
                    scheduler_horizon=5000, maximum_lr=.0001, weight_decay=.005,
                    warmup_steps=500, grad_clip=1., optimizer='AdamW', support_arm='d',
                    post500_lr_multiplier=.2, test_accessed=False)
    for key, value in expected.items():
        require(plan.get(key) == value, 'Unexpected parent recipe: '+key)
    require(status.get('phase') == 'failed' and status.get('last_complete_step') == 803
            and status.get('failure_phase') == 'update_804', 'This entry requires the known step804 failure')
    require(manifest['seed'] == 3407 and manifest['actual_training_names'] == manifest['train_names'],
            'Not the full Train scope')
    require(len(manifest['train_names']) == len(set(manifest['train_names'])) == 2918 and
            len(manifest['val_names']) == len(set(manifest['val_names'])) == 450, 'Train/Val scope differs')
    checkpoint = parent / 'checkpoints/checkpoint_000800.pt'
    require(sha(checkpoint) == CHECKPOINT_SHA, 'Checkpoint800 SHA256 mismatch; stopping before torch.load')
    sidecar = read(checkpoint.with_suffix('.pt.json'))
    require(sidecar.get('step') == 800 and sidecar.get('sha256') == CHECKPOINT_SHA
            and sidecar.get('roundtrip_verified') is True, 'Checkpoint sidecar mismatch')
    rows = [json.loads(s) for s in (parent / 'train.jsonl').read_text().splitlines() if s.strip()]
    validate_prefix(rows, manifest, 800)
    paths = [parent / n for n in ('plan.json', 'manifest.json', 'summary.json', 'train.jsonl')]
    paths += [checkpoint, checkpoint.with_suffix('.pt.json')]
    fingerprints = {str(p): sha(p) for p in paths}
    sys.path.insert(0, str(package))
    spec = importlib.util.spec_from_file_location('_verified_readout_experiment', package / 'experiment.py')
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    return exp, plan, manifest, rows, checkpoint, fingerprints

def restore_model_optimizer(torch, state, model, optimizer, saved, step):
    require(saved.get('step') == step, 'Checkpoint step mismatch')
    require(all(k in saved for k in ('model', 'optimizer', 'rng')), 'Checkpoint missing model/optimizer/RNG')
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    for key, value in (('model', model.state_dict()), ('optimizer', optimizer.state_dict())):
        require(state.state_fingerprint(value) == state.state_fingerprint(saved[key]), 'Restored '+key+' differs')
    counts = [int(v['step']) for v in optimizer.state.values() if 'step' in v]
    require(counts and all(v == step for v in counts), 'Restored optimizer step differs')
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite checkpoint parameters')

def reconstruct_iterator(exp, runtime, manifest, rows, restored_step):
    prefix = validate_prefix(rows, manifest, restored_step)
    iterator = exp.batches(runtime, manifest['actual_training_names'], manifest['seed'])
    for index, row in enumerate(prefix, 1):
        epoch, batch = next(iterator)
        require(epoch == row['epoch'] and str(batch[-1][0]) == row['image'],
                'Reconstructed data order differs at step '+str(index))
        del batch
        if index % 200 == 0:
            print('RECOVERY DATA ORDER verified '+str(index)+'/'+str(restored_step), flush=True)
    return iterator

def assert_unchanged(fingerprints):
    for path, digest in fingerprints.items():
        require(sha(path) == digest, 'Parent evidence changed: '+path)

def protocol_delta(before, after):
    left, right = copy.deepcopy(before), copy.deepcopy(after)
    a = left.get('execution_settings', {}).pop('cuda_launch_blocking', None)
    b = right.get('execution_settings', {}).pop('cuda_launch_blocking', None)
    require(left == right, 'Runtime protocol differs beyond explicit CUDA_LAUNCH_BLOCKING instrumentation')
    require(b == '1', 'Observed recovery requires CUDA_LAUNCH_BLOCKING=1')
    return {'execution_settings.cuda_launch_blocking': {'before': a, 'after': b}}

def run_worker(args):
    output = args.output.resolve()
    try:
        require(os.environ.get('CUDA_LAUNCH_BLOCKING') == '1', 'Worker requires CUDA_LAUNCH_BLOCKING=1 before Torch import')
        require(not (output / 'manifest.json').exists(), 'Output already used; never rerun this directory')
        lockpath = args.parent.parent / '.readout_active.lock'
        if args.lease_fd is None:
            lease = lockpath.open('a+')
        else:
            require(Path(os.readlink(f'/proc/self/fd/{args.lease_fd}')).resolve() == lockpath.resolve(),
                    'Inherited lease does not belong to this run root')
            lease = os.fdopen(args.lease_fd, 'a+')
        with lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            gpu_idle()
            return run_locked(args, output)
    except BaseException as exc:
        # Includes pre-runtime lease/GPU failures, so a detached job cannot remain "starting".
        summary = read(output / 'summary.json')
        if summary.get('phase') != 'failed':
            summary.update(phase='failed', complete=False, error=type(exc).__name__+': '+str(exc))
            write(output / 'summary.json', summary)
        raise

def run_locked(args, output):
    status = dict(format=FORMAT, variant='read_aux', effective_mode='read_aux', seed=3407,
                  phase='preflight', complete=False, training_complete=False, full_validation_evidence=False,
                  smoke=False, test_accessed=False, last_complete_step=800, endpoint_step=2500,
                  restored_step=800, new_updates=0, historical_failure_resolved=False,
                  exact_historical_replay=False, observation_synchronizes=True,
                  optimizer_state_restored=False, pid=os.getpid(), output=str(output), evaluations={})
    write(output / 'summary.json', status)
    started = time.monotonic()
    reports = []
    try:
        exp, origin_plan, parent_manifest, rows, checkpoint, fingerprints = read_parent(args.package, args.parent)
        print('RECOVERY PREFLIGHT: verified checkpoint800; checking frozen source and bank', flush=True)
        bridge = exp.load_file(Path(origin_plan['addon_dir']) / 'runtime_bridge.py', 'recovery_bridge')
        exp.verify_plan(origin_plan, bridge)
        lineage = dict(parent=str(args.parent.resolve()), checkpoint=str(checkpoint), checkpoint_sha256=CHECKPOINT_SHA,
                       restored_step=800, detailed_steps=[801,802,803,804], original_failed_runs_unchanged=True,
                       exact_historical_replay=False, historical_failure_resolved=False,
                       cuda_launch_blocking='1', observation_changes_timing=True,
                       detailed_checks=['anomaly_detection', 'canonical_route_immutability', 'saved_integer_immutability'],
                       recovery_runner_sha256=sha(output / 'runner.pyz'),
                       parent_protocol_info=parent_manifest['protocol_info'],
                       inherited_training_rows=800, new_training_starts_at=801,
                       prior_evaluations_not_promoted=True, parent_fingerprints=fingerprints)
        plan = dict(origin_plan, fresh_initialization=False, old_head_weights_reused=True, recovery=lineage)
        write(output / 'plan.json', plan)
        write(output / 'recovery.json', lineage)
        bindings = bridge.load_runtime(origin_plan['source'])
        torch, original, state = bindings.torch, bindings.original_train, bindings.legacy.state
        require(torch.cuda.is_available(), 'This job needs CUDA; local CPU tests are not GPU evidence')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(saved.get('format') == FORMAT+'_CHECKPOINT' and saved.get('plan') == origin_plan,
                'Checkpoint recipe does not match parent plan')
        for key in ('variant','effective_mode','read_weight','test_accessed'):
            require(saved.get(key) == origin_plan[key], 'Checkpoint field mismatch: '+key)
        require(saved.get('protocol_info') == parent_manifest['protocol_info'], 'Checkpoint protocol mismatch')
        from model_adapter import make_classifier
        from recovery_observer import observed_update
        classifier = make_classifier(original.JointRegionMemoryClassifier, mode='read_aux')
        runtime_args = bridge.runtime_arguments(origin_plan['source'], seed=3407, output=output)
        old_config, new_config = dict(parent_manifest['runtime_config']), dict(vars(runtime_args))
        old_config.pop('output', None); new_config.pop('output', None)
        require(old_config == new_config, 'Runtime configuration differs beyond output')
        addon = Path(plan['addon_dir'])
        with exp.addon_imports(addon, bridge):
            support_module = bridge.named(addon / 'supports.py', 'readout_supports')
            import slots
            require(Path(slots.__file__).resolve() == (Path(plan['source']['package']) / 'deps/original/slots.py').resolve(),
                    'Unexpected slots module')
            print('RECOVERY LOAD: frozen model, Memory and input verification; this can take several minutes', flush=True)
            with bindings.Runtime(runtime_args, output) as runtime:
                original._formal_runtime(runtime, False)
                full_train, full_val = original._names(runtime, False)
                require(list(full_train) == parent_manifest['train_names'] and list(full_val) == parent_manifest['val_names'],
                        'Train/Val membership changed')
                lineage['protocol_execution_delta'] = protocol_delta(saved['protocol_info'], runtime.protocol_info)
                write(output / 'recovery.json', lineage)
                write(output / 'plan.json', plan)
                original._checked_verify(runtime)
                factory = support_module.SupportFactory(runtime, seed=3407)
                original._seed(3407)
                model = classifier(chunk_size=int(runtime_args.chunk_size)).to(runtime.device)
                require(state.state_fingerprint(model.state_dict()) == parent_manifest['initial_head_sha256'],
                        'Fresh initialization identity no longer matches parent')
                optimizer = torch.optim.AdamW(model.parameters(), lr=float(runtime_args.lr),
                                             weight_decay=float(runtime_args.weight_decay))
                restore_model_optimizer(torch, state, model, optimizer, saved, 800)
                print('RECOVERY RESTORED: model and AdamW step800 verified; reconstructing data order', flush=True)
                iterator = reconstruct_iterator(exp, runtime, parent_manifest, rows, 800)
                state.restore_rng(saved['rng'], require_cuda=True)
                require(state.state_fingerprint(state.capture_rng()) == state.state_fingerprint(saved['rng']),
                        'Restored RNG differs')
                status.update(optimizer_state_restored=True, rng_restored=True, data_order_prefix_verified=True)
                manifest = dict(parent_manifest, plan=plan, runtime_config=vars(runtime_args),
                                protocol_info=runtime.protocol_info, fresh_initialization=False, recovery=lineage)
                write(output / 'manifest.json', manifest)
                write(output / 'support_provenance.json', factory.provenance())
                write(output / 'summary.json', status)
                del saved
                def assess(step, split, names, folder):
                    status['phase'] = f'evaluate_{split}_{step}'
                    write(output / 'summary.json', status)
                    report = exp.evaluate(runtime, model, factory, state, original, list(names), output / folder,
                                          step=step, split=split, seed=3407, smoke=False, variant='read_aux')
                    report['recovery'] = lineage
                    reports.append((output / folder / 'result.json', report))
                    status['evaluations'][folder] = dict(step=step, image_count=report['image_count'], metrics=report['metrics'])
                    write(output / 'summary.json', status)
                with (output / 'train.jsonl').open('x') as journal:
                    for row in rows[:800]:
                        journal.write(json.dumps(row, allow_nan=False)+'\n')
                    journal.flush()
                    for step in range(801, 2501):
                        status['phase'] = f'prepare_{step}'
                        epoch, batch = next(iterator)
                        example = runtime.prepare(batch)
                        original._assert_example(example)
                        require(example.image in full_train, 'Training example outside Train')
                        if step <= len(rows):
                            require(example.image == rows[step-1]['image'] and epoch == rows[step-1]['epoch'],
                                    'Recorded continuation sample order differs')
                        status['phase'] = f'update_{step}'
                        write(output / 'summary.json', status)
                        row = observed_update(exp, model, optimizer, example, runtime, factory, state, slots,
                                              step=step, read_weight=.01, multiplier=plan['post500_lr_multiplier'],
                                              output=output, detailed=step <= 804)
                        row.update(step=step, epoch=epoch, epoch_seed=3407+epoch,
                                   image=example.image, event=example.event, variant='read_aux')
                        journal.write(json.dumps(row, allow_nan=False)+'\n'); journal.flush()
                        status.update(last_complete_step=step, new_updates=step-800)
                        if step == 2500:
                            status['training_complete'] = True
                        if step <= 804 or step % 20 == 0:
                            print('RECOVERY TRAIN '+json.dumps(row, allow_nan=False), flush=True)
                        if step == 804:
                            status['local_replay_completed'] = True
                            print('LOCAL 801-804 PASSED in this observed run; continuing same recipe to2500; historical cause unresolved', flush=True)
                        if step == 804 or step % 50 == 0:
                            os.fsync(journal.fileno())
                            status['last_checkpoint'] = exp.save_checkpoint(output / 'checkpoints' / f'checkpoint_{step:06d}.pt',
                                                                           model, optimizer, state, runtime, plan, step)
                        write(output / 'summary.json', status)
                        del row, example, batch
                        if step in (1000,1500,2000,2500):
                            assess(step, 'val', full_val, f'val_step_{step:06d}')
                assess(2500, 'fixed_train', parent_manifest['fixed_train_names'], 'fixed_train_endpoint')
                status['phase'] = 'final_integrity'
                status['integrity_checks'] = original._checked_verify(runtime)
                exp.verify_plan(plan, bridge)
                bindings.legacy_bridge.verify_dependencies(bindings.legacy)
                write(output / 'input_fingerprints.json', {str(p): h for p,h in runtime.input_ledger.items()})
            guard = runtime.guard_report
            require(guard.get('restored') is True and not guard.get('failures') and not guard.get('restore_conflicts'),
                    'PoolGuard final check failed')
            status['pool_guard'] = guard
        exp.verify_plan(plan, bridge)
        assert_unchanged(fingerprints)
        for path, report in reports:
            report.update(complete=True, final_integrity_pending=False, phase='complete',
                          full_validation_evidence=bool(report['whole_val450_processed']))
            write(path, report)
            path.with_suffix('.json.sha256').write_text(sha(path)+'  result.json\n')
        status.update(phase='complete', complete=True, full_validation_evidence=True,
                      original_failed_runs_unchanged=True, elapsed_seconds=time.monotonic()-started, error=None)
        write(output / 'summary.json', status)
        print('RECOVERY COMPLETE '+str(output), flush=True)
    except BaseException as exc:
        status.update(failure_phase=status['phase'], phase='failed', complete=False, full_validation_evidence=False,
                      error=type(exc).__name__+': '+str(exc), elapsed_seconds=time.monotonic()-started)
        write(output / 'summary.json', status)
        (output / 'traceback.txt').write_text(traceback.format_exc())
        raise

def gpu_idle():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            text=True, capture_output=True, timeout=15)
    require(result.returncode == 0, 'Cannot inspect GPU processes: '+result.stderr.strip())
    require(not result.stdout.strip(), 'GPU already has a compute process; wait for it to finish before start')

def start(args):
    read_parent(args.package, args.parent)
    gpu_idle()
    root = args.parent.parent
    archive = Path(sys.argv[0]).resolve()
    require(archive.is_file() and archive.suffix == '.pyz', 'Use the published .pyz for background launch')
    with (root / '.readout_active.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = Path(tempfile.mkdtemp(prefix='read_aux_recover800_', dir=root))
        shutil.copyfile(archive, output / 'runner.pyz')
        write(output / 'launch.json', dict(parent=str(args.parent.resolve()), package=str(args.package.resolve()),
                                         runner_sha256=sha(output / 'runner.pyz'), cuda_launch_blocking='1'))
        write(output / 'summary.json', dict(phase='starting', complete=False, training_complete=False,
                                          restored_step=800, last_complete_step=800, test_accessed=False))
        env = dict(os.environ, CUDA_LAUNCH_BLOCKING='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        with (output / 'train.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, '-B', '-u', str(output / 'runner.pyz'), 'worker',
                                      '--parent', str(args.parent.resolve()), '--package', str(args.package.resolve()),
                                      '--output', str(output), '--lease-fd', str(lease.fileno())],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True, env=env, pass_fds=(lease.fileno(),))
    write(root / '.read_aux_recovery_latest.json', dict(output=str(output), pid=child.pid))
    print(f'已在后台启动恢复：800 → 801–804局部检查 → 2500 → 完整Val450；不访问Test。\nPID={child.pid}\nOUTPUT={output}')
    print(f'日志：{sys.executable} -B {output}/runner.pyz logs --output {output}')
    print(f'状态：{sys.executable} -B {output}/runner.pyz status --output {output}')

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('start','worker','status','logs','check','compare'))
    p.add_argument('--parent', type=Path, default=PARENT)
    p.add_argument('--package', type=Path, default=PACKAGE)
    p.add_argument('--output', type=Path)
    p.add_argument('--baseline', type=Path, default=ROOT / 'original_3407_01')
    p.add_argument('--lease-fd', type=int, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.command == 'start':
        return start(args)
    if args.command == 'check':
        read_parent(args.package, args.parent)
        print('PARENT CHECK PASSED; no Torch import or training; checkpoint800 verified')
        return
    if args.output is None:
        args.output = Path(read(args.parent.parent / '.read_aux_recovery_latest.json')['output'])
    if args.command == 'worker':
        return run_worker(args)
    if args.command == 'compare':
        from compare_recovery import compare
        result, directory = compare(args.package, args.baseline, args.output)
        print(json.dumps(result['differences'], ensure_ascii=False, indent=2))
        print('grid_confusion_identical =', result['grid_confusion_identical'])
        print('Recovery execution delta =', json.dumps(result['execution_protocol_delta']))
        print('Saved:', directory)
        return
    if args.command == 'status':
        print(json.dumps(read(args.output / 'summary.json'), ensure_ascii=False, indent=2))
    else:
        from collections import deque
        with (args.output / 'train.log').open(errors='replace') as stream:
            print(''.join(deque(stream, maxlen=70)), end='')

if __name__ == '__main__':
    main()
