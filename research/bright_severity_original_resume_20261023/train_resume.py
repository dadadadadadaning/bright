"""Resume one original branch from checkpoint150 through2500 with the pinned recipe.

No Test, automatic retry, graph observer, anomaly detection or other variant is
started. The successful diagnostic windows are evidence, not saved training state.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import time
import traceback

import evidence
import preflight

FORMAT = 'BRIGHT_SEVERITY_ORIGINAL_RESUME_V1'
RESTORED_STEP = 150
END_STEP = 2500
EVALUATION_STEPS = (500, 1000, 1500, 2000, 2500)
require, read, write, sha = evidence.require, evidence.read, evidence.write, evidence.sha


def initial_status(output, parent):
    return dict(format=FORMAT, variant='original', effective_mode='original', seed=3407,
                phase='preflight', complete=False, recovery_complete=False, study_complete=False,
                training_complete=False, validation_complete=False, fixed_train_complete=False,
                full_validation_evidence=False, full_test_evidence=False, test_accessed=False,
                restored_step=RESTORED_STEP, endpoint_step=END_STEP,
                last_complete_step=RESTORED_STEP, last_journaled_step=RESTORED_STEP,
                inherited_updates=RESTORED_STEP, new_complete_updates=0, completed_update_calls=0,
                optimizer_updates_returned=0,
                optimizer_counter_semantics='Only fully returned original update calls; a failed call may have executed optimizer.step',
                failed_step_optimizer_execution_unknown=False, training_performed=False,
                backward_index_observer=False, anomaly_detection=False, saved_tensors_hooks_used=False,
                cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING'),
                historical_failure_resolved=False, exact_historical_replay=False,
                observation_changes_timing=True, tensor_values_repaired=False,
                failed_samples_skipped=False, automatic_retry=False, other_variants_started=False,
                evaluations={}, parent=str(Path(parent).resolve()), output=str(Path(output).resolve()),
                pid=os.getpid())


def persist_status(output, status):
    write(Path(output) / 'summary.json', status)


def checkpoint_due(step):
    return step > RESTORED_STEP and (step % 50 == 0 or step == END_STEP)


def write_inherited_prefix(journal, rows, parent):
    """Keep every original scalar/field from1–150; later parent rows are excluded."""
    require(len(rows) >= RESTORED_STEP, 'Parent journal shorter than restored checkpoint')
    for step, original in enumerate(rows[:RESTORED_STEP], 1):
        require(original.get('step') == step and original.get('optimizer_step') == step,
                'Inherited journal is nonconsecutive')
        row = dict(original)
        row.update(inherited=True, inherited_source=str(Path(parent).resolve() / 'train.jsonl'))
        journal.write(json.dumps(row, allow_nan=False) + '\n')
    journal.flush()
    os.fsync(journal.fileno())


def original_update(exp, model, optimizer, example, runtime, factory, state, slots,
                    *, step, status, multiplier):
    """One exact original call; failure accounting never queries a poisoned GPU."""
    status.update(phase=f'update_{step}', training_performed=True)
    try:
        row = exp.update(model, optimizer, example, runtime, factory, state, slots,
                         step=step, read_weight=0, multiplier=multiplier)
    except BaseException:
        status.update(failed_step_optimizer_execution_unknown=True, failed_update_step=step)
        raise
    status['completed_update_calls'] += 1
    status['optimizer_updates_returned'] += 1
    require(row.get('optimizer_step') == step, 'Original update returned wrong optimizer step')
    status.update(last_complete_step=step, new_complete_updates=step-RESTORED_STEP)
    if step == END_STEP:
        status['training_complete'] = True
    return row


def run_updates(exp, model, optimizer, runtime, factory, state, slots, original, iterator,
                full_train, parent_rows, plan, output, status, journal, assess):
    """Original update/checkpoint/Val order, with a checkpoint before every Val."""
    for step in range(RESTORED_STEP+1, END_STEP+1):
        status.update(phase=f'prepare_{step}', current_image=None)
        with exp.capture_training_failure(output, status):
            epoch, batch = next(iterator)
            example = runtime.prepare(batch)
            status['current_image'] = example.image
            original._assert_example(example)
            require(example.image in full_train, 'Training image outside original Train scope')
            if step <= len(parent_rows):
                require(example.image == parent_rows[step-1]['image'] and epoch == parent_rows[step-1]['epoch'],
                        'Resumed data order differs from parent at step '+str(step))
            if step == 158:
                require(example.image == evidence.TARGET_IMAGE and epoch == 0,
                        'Known step158 image differs from restored order')
            row = original_update(exp, model, optimizer, example, runtime, factory, state, slots,
                                  step=step, status=status, multiplier=plan['post500_lr_multiplier'])
        row.update(step=step, epoch=epoch, epoch_seed=plan['seed']+epoch,
                   image=example.image, event=example.event, variant='original', inherited=False)
        if step <= len(parent_rows):
            row['parent_loss_absolute_difference'] = abs(row['original_loss']-parent_rows[step-1]['original_loss'])
        journal.write(json.dumps(row, allow_nan=False)+'\n')
        journal.flush()
        status['last_journaled_step'] = step
        if step == RESTORED_STEP+1 or step % 20 == 0 or step == END_STEP:
            print('RESUME ORIGINAL TRAIN '+json.dumps(row, allow_nan=False), flush=True)
            persist_status(output, status)
        if checkpoint_due(step):
            status['phase'] = f'checkpoint_{step}'
            os.fsync(journal.fileno())
            status['last_checkpoint'] = exp.save_checkpoint(
                output / 'checkpoints' / f'checkpoint_{step:06d}.pt',
                model, optimizer, state, runtime, plan, step)
            persist_status(output, status)
            print(f'RESUME ORIGINAL CHECKPOINT step={step}', flush=True)
        del row, example, batch
        if step in EVALUATION_STEPS:
            assess(step, 'val', runtime.val_names, f'val_step_{step:06d}')


def assess_original(exp, runtime, model, factory, state, original, names, output,
                    status, results, *, step, split, folder):
    names = list(names)
    require(split in ('val', 'fixed_train'), 'Only original Val/fixed Train evaluations are allowed')
    if split == 'val':
        require(step in EVALUATION_STEPS and names == list(runtime.val_names) and len(names) == 450,
                'Validation must use a fixed scheduled checkpoint and whole Val450')
    else:
        require(step == END_STEP, 'Fixed Train is evaluated only at the endpoint')
    status.update(phase=f'evaluate_{split}_{step}', current_image=None)
    persist_status(output, status)
    print(f'RESUME ORIGINAL ASSESS: {split} step={step} requested_images={len(names)}', flush=True)
    report = exp.evaluate(runtime, model, factory, state, original, names, output / folder,
                          step=step, split=split, seed=3407, smoke=False, variant='original')
    require(report.get('execution_complete') is True and report.get('image_count') == len(names)
            and report.get('requested_names') == names and report.get('journal_verified') is True
            and report.get('test_accessed') is False and report.get('head_state_unchanged') is True
            and report.get('rng_restored') is True, 'Original evaluation evidence incomplete')
    require(report.get('step') == step and report.get('split') == split and
            report.get('variant') == 'original', 'Original evaluation identity mismatch')
    if split == 'val':
        require(report.get('whole_val450_processed') is True, 'Val450 not completely evaluated')
    report['recovery'] = dict(format=FORMAT, source_parent=status['parent'],
                              restored_checkpoint_step=RESTORED_STEP,
                              evaluated_checkpoint=status['last_checkpoint'],
                              historical_failure_resolved=False)
    path = output / folder / 'result.json'
    write(path, report)
    results.append((path, report))
    status['evaluations'][folder] = dict(step=step, split=split, image_count=report['image_count'],
                                        execution_complete=True, final_integrity_pending=True,
                                        result=str(path), metrics=report['metrics'], inherited=False)
    if split == 'fixed_train':
        status['fixed_train_complete'] = True
    status['validation_complete'] = all(
        status['evaluations'].get(f'val_step_{value:06d}', {}).get('execution_complete') is True
        for value in EVALUATION_STEPS)
    persist_status(output, status)
    return report


def verify_prior_evidence(parent_fingerprints, observed_info, plain_info):
    for fingerprints in (parent_fingerprints, observed_info['fingerprints'], plain_info['fingerprints']):
        evidence.assert_unchanged(fingerprints)


def promote_new_results(exp, output, status, results):
    """Only called after runtime, source, dependency and prior-evidence verification."""
    expected = {f'val_step_{step:06d}' for step in EVALUATION_STEPS} | {'fixed_train_endpoint'}
    require({path.parent.name for path, _ in results} == expected and len(results) == len(expected),
            'Cannot finish without all five Val450 evaluations and fixed Train')
    require(status['training_complete'] is True and status['validation_complete'] is True and
            status['fixed_train_complete'] is True and status['last_complete_step'] == END_STEP and
            status['last_journaled_step'] == END_STEP and
            status['new_complete_updates'] == END_STEP-RESTORED_STEP and
            status['completed_update_calls'] == END_STEP-RESTORED_STEP and
            status['optimizer_updates_returned'] == END_STEP-RESTORED_STEP and
            (status.get('last_checkpoint') or {}).get('step') == END_STEP and
            (status.get('last_checkpoint') or {}).get('roundtrip_verified') is True,
            'Recovery execution incomplete')
    for path, report in results:
        report.update(complete=True, phase='complete', final_integrity_pending=False,
                      full_validation_evidence=bool(report['whole_val450_processed']))
        write(path, report)
        path.with_suffix('.json.sha256').write_text(sha(path)+'  result.json\n')
        status['evaluations'][path.parent.name].update(complete=True, final_integrity_pending=False)
    status.update(phase='complete', complete=True, recovery_complete=True,
                  full_validation_evidence=True, study_complete=False, error=None)


def run_worker(args):
    output, parent, package = (Path(getattr(args, name)).resolve() for name in ('output', 'parent', 'package'))
    observed = Path(getattr(args, 'observed', getattr(args, 'baseline', evidence.BASELINE))).resolve()
    plain = Path(getattr(args, 'plain', preflight.PLAIN)).resolve()
    # These checks precede every write, including error/status recording.
    require(output not in (parent, observed, plain), 'Never overwrite input evidence directories')
    require(output.is_dir(), 'Launcher must create a new output directory')
    for name in ('manifest.json', 'plan.json', 'train.jsonl', 'checkpoints', 'first_failure.json'):
        require(not (output / name).exists(), 'Resume output already used: '+name)
    if (output/'summary.json').exists():
        require(read(output/'summary.json').get('phase') == 'recovery_starting',
                'Resume output has an existing worker summary')
    status = initial_status(output, parent)
    status.update(observed_diagnostic=str(observed), plain_diagnostic=str(plain))
    persist_status(output, status)
    started = time.monotonic()
    checked = None
    results = []
    try:
        require(os.environ.get('CUDA_LAUNCH_BLOCKING') == '1', 'CUDA_LAUNCH_BLOCKING=1 required before Torch import')
        with (parent.parent / '.severity_active.lock').open('a+') as active:
            fcntl.flock(active, fcntl.LOCK_EX | fcntl.LOCK_NB)
            checked = preflight.check_inputs(package, parent, observed, plain)
            exp, plan, parent_manifest, parent_rows, checkpoint, parent_hashes, observed_info, plain_info = checked
            bridge = exp.load_file(Path(plan['addon_dir'])/'runtime_bridge.py', 'original_resume_bridge')
            print('RESUME ORIGINAL PREFLIGHT: original150 + both successful diagnostic windows; no historical fix claimed', flush=True)
            exp.verify_plan(plan, bridge)
            write(output/'plan.json', plan)  # Preserve the original checkpoint recipe exactly.
            status['last_checkpoint'] = {**read(checkpoint.with_suffix('.pt.json')), 'inherited': True}
            initial_val = plain_info['initial_val']
            status['inherited_initial_val'] = initial_val
            write(output/'initial_val_reference.json', initial_val)
            recovery = dict(format=FORMAT, parent=str(parent), checkpoint=str(checkpoint),
                            checkpoint_sha256=sha(checkpoint), restored_step=RESTORED_STEP,
                            endpoint_step=END_STEP, inherited_steps=[1, RESTORED_STEP],
                            new_steps=[RESTORED_STEP+1, END_STEP], discarded_unsaved_parent_steps=[151,157],
                            parent_fingerprints=parent_hashes, observed_diagnostic=observed_info,
                            plain_diagnostic=plain_info, initial_val_reference_only=True,
                            parent_plan_unchanged=True, formal_checkpoints_use_original_format=True,
                            diagnostic_updated_weights_reused=False, historical_failure_resolved=False,
                            exact_historical_replay=False, test_accessed=False, other_variants_started=False)
            write(output/'recovery_lineage.json', recovery)
            bindings = bridge.load_runtime(plan['source'])
            torch, original, state = bindings.torch, bindings.original_train, bindings.legacy.state
            require(torch.cuda.is_available(), 'Formal resumed training requires CUDA')
            require(not torch.is_anomaly_enabled(), 'Original resume requires anomaly detection disabled')
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            require(saved.get('format') == exp.FORMAT+'_CHECKPOINT' and saved.get('plan') == plan,
                    'Checkpoint format or plan differs')
            for key in ('variant', 'effective_mode', 'read_weight', 'test_accessed'):
                require(saved.get(key) == plan[key], 'Checkpoint field mismatch: '+key)
            require(saved.get('protocol_info') == parent_manifest['protocol_info'], 'Checkpoint protocol differs')
            runtime_args = bridge.runtime_arguments(plan['source'], seed=3407, output=output)
            before, after = dict(parent_manifest['runtime_config']), dict(vars(runtime_args))
            before.pop('output', None)
            after.pop('output', None)
            require(before == after, 'Runtime config changed beyond output directory')
            old_adapter = exp.load_file(package/'old_model_adapter.py', 'resume_old_adapter')
            adapter = exp.load_file(package/'severity_adapter.py', 'resume_severity_adapter')
            addon = Path(plan['addon_dir'])
            with exp.addon_imports(addon, bridge):
                support_module = bridge.named(addon/'supports.py', 'readout_supports')
                import slots
                require(Path(slots.__file__).resolve() == (Path(plan['source']['package'])/'deps/original/slots.py').resolve(),
                        'Unexpected slots module')
                classifier = adapter.make_classifier(original.JointRegionMemoryClassifier, old_adapter, slots, mode='original')
                status['phase'] = 'load_runtime'
                persist_status(output, status)
                print('RESUME ORIGINAL LOAD: frozen model/Memory and inherited input verification', flush=True)
                with bindings.Runtime(runtime_args, output) as runtime:
                    # Capture before Runtime.__exit__ can encounter a poisoned CUDA context.
                    with exp.capture_training_failure(output, status):
                        original._formal_runtime(runtime, False)
                        full_train, full_val = map(list, original._names(runtime, False))
                        require(full_train == parent_manifest['train_names'] and full_val == parent_manifest['val_names'],
                                'Runtime Train/Val scope differs')
                        require(runtime.protocol_info == saved['protocol_info'], 'Runtime protocol differs')
                        original._checked_verify(runtime)
                        factory = support_module.SupportFactory(runtime, seed=3407)
                        original._seed(3407)
                        model = classifier(chunk_size=int(runtime_args.chunk_size)).to(runtime.device)
                        require(state.state_fingerprint(model.state_dict()) == parent_manifest['initial_head_sha256'],
                                'Original fresh initialization identity differs')
                        optimizer = torch.optim.AdamW(model.parameters(), lr=float(runtime_args.lr),
                                                     weight_decay=float(runtime_args.weight_decay))
                        evidence.restore_model_optimizer(torch, state, model, optimizer, saved, RESTORED_STEP)
                        status.update(phase='reconstruct_data_order', optimizer_state_restored=True)
                        persist_status(output, status)
                        iterator = evidence.reconstruct_iterator(exp, runtime, parent_manifest, parent_rows, RESTORED_STEP)
                        state.restore_rng(saved['rng'], require_cuda=True)
                        require(state.state_fingerprint(state.capture_rng()) == state.state_fingerprint(saved['rng']),
                                'Checkpoint RNG restore differs')
                        manifest = dict(parent_manifest)
                        manifest.update(format=FORMAT, parent_manifest_path=str(parent/'manifest.json'),
                                        runtime_config=vars(runtime_args), recovery=recovery,
                                        fresh_initialization=False, current_process_started_from_checkpoint=True,
                                        rng_restored=True, optimizer_state_restored=True, data_order_prefix_verified=True,
                                        inherited_initial_val=initial_val)
                        write(output/'manifest.json', manifest)
                        write(output/'support_provenance.json', factory.provenance())
                        del saved
                        def assess(step, split, names, folder):
                            return assess_original(exp, runtime, model, factory, state, original, names,
                                                   output, status, results, step=step, split=split, folder=folder)
                        with (output/'train.jsonl').open('x', encoding='utf-8') as journal:
                            write_inherited_prefix(journal, parent_rows, parent)
                            run_updates(exp, model, optimizer, runtime, factory, state, slots, original,
                                        iterator, set(full_train), parent_rows, plan, output, status, journal, assess)
                        assess(END_STEP, 'fixed_train', parent_manifest['fixed_train_names'], 'fixed_train_endpoint')
                        status['phase'] = 'final_integrity'
                        persist_status(output, status)
                        status['integrity_checks'] = original._checked_verify(runtime)
                        exp.verify_plan(plan, bridge)
                        bindings.legacy_bridge.verify_dependencies(bindings.legacy)
                        write(output/'input_fingerprints.json', {str(path): digest for path,digest in runtime.input_ledger.items()})
                guard = runtime.guard_report
                require(guard.get('restored') is True and not guard.get('failures') and not guard.get('restore_conflicts'),
                        'PoolGuard final restoration failed')
                status['pool_guard'] = guard
            exp.verify_plan(plan, bridge)
            verify_prior_evidence(parent_hashes, observed_info, plain_info)
            status.update(original_failed_run_unchanged=True, original_failed_runs_unchanged=True,
                          diagnostic_inputs_unchanged=True)
            promote_new_results(exp, output, status, results)
        status['elapsed_seconds'] = time.monotonic()-started
        persist_status(output, status)
        print('RESUME ORIGINAL COMPLETE:2500 updates + five Val450 + fixed Train; no Test or other variants started', flush=True)
        return status
    except BaseException as error:
        failure = read(output/'first_failure.json') if (output/'first_failure.json').is_file() else {}
        status.update(failure_phase=failure.get('phase', status['phase']), phase='failed', complete=False,
                      recovery_complete=False, full_validation_evidence=False,
                      error=failure.get('error', type(error).__name__+': '+str(error)),
                      elapsed_seconds=time.monotonic()-started)
        if checked:
            try:
                verify_prior_evidence(checked[5], checked[6], checked[7])
                status.update(original_failed_run_unchanged=True, original_failed_runs_unchanged=True,
                              diagnostic_inputs_unchanged=True)
            except Exception as integrity_error:
                status.update(original_failed_run_unchanged=False, original_failed_runs_unchanged=False,
                              diagnostic_inputs_unchanged=False, integrity_error=str(integrity_error))
        persist_status(output, status)
        current_traceback = traceback.format_exc()
        primary_traceback = failure.get('traceback') or current_traceback
        (output/'traceback.txt').write_text(primary_traceback)
        if current_traceback != primary_traceback:
            (output/'outer_exception_traceback.txt').write_text(current_traceback)
        raise
