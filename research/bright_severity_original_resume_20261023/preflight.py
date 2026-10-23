"""CPU/stdlib evidence gates for one original-model continuation, not a fault fix."""
from pathlib import Path
import json
import random

import evidence

PLAIN = evidence.ROOT / 'backward158_plain_82l__kmb'
PLAIN_SHA256 = 'f8793b2a56b0f7bdaa7d911f50b6de7972916a21707584bf0d511d8f9c08070c'


def check_inputs(package, parent, observed=evidence.BASELINE, plain=PLAIN):
    parent, observed, plain = map(lambda p: Path(p).resolve(), (parent, observed, plain))
    result = evidence.check_pair(package, parent, observed)
    exp, plan, manifest, parent_rows, checkpoint, parent_hashes, observed_info = result
    need, read, sha = evidence.require, evidence.read, evidence.sha
    need(len({parent, observed, plain}) == 3, 'Recovery inputs must be distinct')
    need(sha(plain / 'runner.pyz') == PLAIN_SHA256, 'Plain replay runner SHA differs')
    summary = read(plain / 'summary.json')
    expected = dict(format='BRIGHT_BACKWARD158_PLAIN_V1', phase='plain_window_complete',
        complete=False, diagnostic_complete=True, training_complete=False,
        full_validation_evidence=False, full_test_evidence=False, test_accessed=False,
        optimizer_updates_returned=8, completed_update_calls=8, new_complete_updates=8,
        restored_step=150, last_complete_step=158, bounded_endpoint=158,
        original_endpoint=2500, optimizer_state_restored=True,
        backward_index_observer=False, anomaly_detection=False, cuda_launch_blocking='1',
        failed_step_optimizer_execution_unknown=False, saved_tensors_hooks_used=False,
        tensor_values_repaired=False, failed_samples_skipped=False, automatic_retry=False,
        original_failed_runs_unchanged=True, observed_baseline_unchanged=True,
        historical_failure_resolved=False, exact_historical_replay=False)
    for key, value in expected.items():
        need(summary.get(key) == value, 'Plain replay incomplete/mismatched: '+key)
    need(summary.get('error') is None and Path(summary['parent']).resolve() == parent and
         Path(summary['baseline']).resolve() == observed, 'Plain replay lineage/error differs')
    integrity = summary.get('runtime_integrity') or {}
    guard = summary.get('pool_guard') or {}
    need(integrity and all(value is True for value in integrity.values()) and
         guard.get('restored') is True and not guard.get('failures') and not guard.get('restore_conflicts'),
         'Plain replay final integrity incomplete')
    supervisor = read(plain / 'supervisor.json')
    need(supervisor.get('job_finished') is True and supervisor.get('returncode') == 0 and
         supervisor.get('phase') == 'plain_window_complete' and supervisor.get('error') is None,
         'Plain supervisor did not finish successfully')
    replay_manifest = read(plain / 'manifest.json')
    need(replay_manifest.get('parent_manifest') == manifest and
         replay_manifest.get('protocol_info') == manifest['protocol_info'],
         'Plain manifest/protocol differs')
    for key in ('rng_restored', 'optimizer_state_restored', 'data_order_prefix_verified'):
        need(replay_manifest.get(key) is True, 'Plain restore evidence missing: '+key)
    before, after = dict(manifest['runtime_config']), dict(replay_manifest['runtime_config'])
    before.pop('output', None)
    after.pop('output', None)
    need(before == after, 'Plain runtime config differs beyond output')
    lineage = read(plain / 'lineage.json')
    need(Path(lineage['parent']).resolve() == parent and
         Path(lineage['checkpoint']).resolve() == checkpoint.resolve() and
         lineage.get('checkpoint_sha256') == sha(checkpoint) and
         lineage.get('parent_fingerprints') == parent_hashes and
         Path(lineage['baseline']).resolve() == observed and
         lineage.get('baseline_fingerprints') == observed_info['fingerprints'],
         'Plain lineage inputs changed')
    need(lineage.get('steps') == list(range(151, 159)) and
         lineage.get('formal_checkpoint_written') is False and
         lineage.get('test_accessed') is False, 'Plain scope differs')
    need(not (plain / 'first_failure.json').exists(), 'Plain replay contains a first failure')
    rows = [json.loads(line) for line in (plain / 'diagnostic_updates.jsonl').read_text().splitlines()
            if line.strip()]
    need(len(rows) == 8 and all(row.get('optimizer_step') == step and
         row.get('training_loss') == row.get('original_loss') for step, row in zip(range(151, 159), rows)),
         'Plain update journal incomplete')
    comparison = evidence.compare_rows(observed_info['rows'], rows)
    need(read(plain / 'comparison.json') == comparison, 'Plain comparison differs from saved journals')
    # This remains a descriptive comparison; no threshold promotes it into a fault fix.
    fixed = manifest['fixed_train_names']
    fixed_count = plan['fixed_train_count']
    need(isinstance(fixed_count, int) and 1 <= fixed_count <= 2918, 'Invalid fixed Train count')
    expected_fixed = sorted(random.Random(plan['seed'] + 191).sample(manifest['train_names'], fixed_count))
    need(fixed == expected_fixed, 'Fixed Train subset differs from original recipe')
    initial_path = parent / 'initial_val/result.json'
    initial = read(initial_path)
    images = initial_path.parent / 'images.jsonl'
    need(initial.get('step') == 0 and initial.get('split') == 'val' and
         initial.get('variant') == 'original' and initial.get('image_count') == 450 and
         initial.get('requested_names') == manifest['val_names'] and
         initial.get('execution_complete') is True and initial.get('journal_verified') is True and
         initial.get('test_accessed') is False and initial.get('images_sha256') == sha(images),
         'Parent initial Val evidence incomplete or changed')
    names = []
    with images.open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            need(row.get('index') == index, 'Parent initial Val journal order differs')
            names.append(row['image'])
    need(names == manifest['val_names'], 'Parent initial Val coverage differs')
    files = [plain / name for name in ('runner.pyz', 'summary.json', 'supervisor.json',
             'manifest.json', 'lineage.json', 'diagnostic_updates.jsonl', 'comparison.json')]
    files += [initial_path, images]
    info = dict(root=str(plain), rows=rows, fingerprints={str(p): sha(p) for p in files},
                initial_val=dict(path=str(initial_path), sha256=sha(initial_path),
                    images_path=str(images), images_sha256=sha(images),
                    phase=initial.get('phase'), complete=initial.get('complete'),
                    full_validation_evidence=initial.get('full_validation_evidence'),
                    inherited_reference_only=True, promoted_by_recovery=False))
    return (*result, info)
