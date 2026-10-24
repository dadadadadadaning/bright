"""Stdlib-only, read-only gates for the known calibration804 failure.

No replay evidence, checkpoint deserialization, Torch import or evidence promotion.
The parent recipe is retained; the worker declares its execution change separately.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import random

import helpers

PARENT = helpers.PARENT
IDENTITY = helpers.IDENTITY
PARENT_FORMAT = 'BRIGHT_SEVERITY_MEMORY_V1'
CHECKPOINT_SHA256 = '23766189b371dff7c62eb373baaf78c0e216721e13be23623255e4771955a747'
TARGET_IMAGE = 'la_palma-volcano_00000270'
RESTORED_STEP = 800
LAST_COMPLETE_STEP = 803
require, read, sha = helpers.require, helpers.read, helpers.sha


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _counts(value, shape, label):
    require(isinstance(value, list) and len(value) == shape[0], 'Invalid '+label+' shape')
    if len(shape) > 1:
        for row in value:
            _counts(row, shape[1:], label)
    else:
        require(all(type(n) is int and n >= 0 for n in value), 'Invalid '+label+' counts')
    return value


def _zero_matrix():
    return [[0] * 4 for _ in range(4)]


def _add_matrix(total, value):
    for i in range(4):
        for j in range(4):
            total[i][j] += value[i][j]


def _evaluation(parent, folder, step, names):
    """Rebuild ordered Val450 confusion/transition/loss evidence without NumPy."""
    report_path = parent / folder / 'result.json'
    images_path = report_path.with_name('images.jsonl')
    report = read(report_path)
    expected = dict(format=PARENT_FORMAT+'_EVALUATION', step=step, split='val', seed=3407,
                    variant='calibration', image_count=450, requested_names=names,
                    execution_complete=True, journal_verified=True, whole_val450_processed=True,
                    smoke=False, test_accessed=False, head_state_unchanged=True, rng_restored=True)
    for key, value in expected.items():
        require(report.get(key) == value, 'Parent '+folder+' evidence differs: '+key)
    # Both evaluations preceded the parent's failure, so the final run-wide
    # integrity gate never promoted them. Preserve these original flags verbatim.
    pending = dict(phase='evaluation_complete_integrity_pending', complete=False,
                   full_validation_evidence=False, final_integrity_pending=True)
    for key, value in pending.items():
        require(report.get(key) == value, 'Parent '+folder+' pending flag differs: '+key)
    require(not report_path.with_name('failure.json').exists(), 'Parent '+folder+' has an evaluation failure')
    image_digest = sha(images_path)
    require(report.get('images_sha256') == image_digest, 'Parent '+folder+' journal hash differs')
    rows = _rows(images_path)
    require(len(rows) == len(names), 'Parent '+folder+' Val coverage differs')
    keys = ('grid_probq', 'calibration')
    totals = {key: _zero_matrix() for key in keys}
    losses = {key: 0.0 for key in keys}
    events = {}
    transitions = [[row[:] for row in _zero_matrix()] for _ in range(4)]
    for index, (row, name) in enumerate(zip(rows, names)):
        require(row.get('index') == index and row.get('image') == name,
                'Parent '+folder+' journal order differs')
        require(row.get('test_accessed') is False and isinstance(row.get('event'), str),
                'Parent '+folder+' journal scope differs')
        require(set(row.get('confusion', {})) == set(keys) and
                set(row.get('original_loss', {})) == set(keys), 'Parent '+folder+' journal paths differ')
        event = events.setdefault(row['event'], {key: _zero_matrix() for key in keys})
        for key in keys:
            cm = _counts(row['confusion'][key], (4, 4), 'confusion')
            _add_matrix(totals[key], cm)
            _add_matrix(event[key], cm)
            loss = row['original_loss'][key]
            require(_finite(loss), 'Parent '+folder+' nonfinite loss')
            losses[key] += loss
        paired = _counts(row.get('paired_counts_grid_to_variant'), (4, 4, 4), 'paired transition')
        for gt in range(4):
            for predicted in range(4):
                require(sum(paired[gt][predicted]) == row['confusion']['grid_probq'][gt][predicted] and
                        sum(paired[gt][old][predicted] for old in range(4)) == row['confusion']['calibration'][gt][predicted],
                        'Parent '+folder+' paired/confusion counts differ')
            _add_matrix(transitions[gt], paired[gt])
    require(set(report.get('metrics', {})) == set(keys), 'Parent '+folder+' aggregate paths differ')
    for key in keys:
        require(report['metrics'][key].get('confusion') == totals[key], 'Parent '+folder+' aggregate confusion differs')
        mean_loss = report['metrics'][key].get('mean_original_loss')
        require(_finite(mean_loss) and math.isclose(mean_loss, losses[key]/len(names), rel_tol=1e-12, abs_tol=1e-12),
                'Parent '+folder+' aggregate loss differs')
    recorded_events = report.get('per_event', {})
    require(set(recorded_events) == set(events), 'Parent '+folder+' event coverage differs')
    for event, tables in events.items():
        require(set(recorded_events[event]) == set(keys) and all(
            recorded_events[event][key].get('confusion') == tables[key] for key in keys),
            'Parent '+folder+' event confusion differs')
    require(report.get('paired_counts_grid_to_variant') == transitions and
            report.get('paired_axes') == ['GT', 'grid_probq', 'calibration'],
            'Parent '+folder+' aggregate paired counts differ')
    paths = [report_path, images_path]
    sidecar = report_path.with_suffix('.json.sha256')
    if sidecar.exists():
        require(sidecar.read_text().split() == [sha(report_path), 'result.json'],
                'Parent '+folder+' result sidecar differs')
        paths.append(sidecar)
    reference = dict(path=str(report_path), sha256=sha(report_path), images_path=str(images_path),
                     images_sha256=image_digest, step=step, split='val', variant='calibration',
                     image_count=450, execution_complete=report['execution_complete'],
                     journal_verified=report['journal_verified'], whole_val450_processed=report['whole_val450_processed'],
                     **{key: report[key] for key in pending}, inherited_reference_only=True,
                     promoted_by_recovery=False)
    return reference, {str(path): sha(path) for path in paths}


def check_inputs(package, parent):
    """Return exactly exp, plan, manifest, rows, checkpoint, fingerprints, info."""
    package, parent = Path(package).resolve(), Path(parent).resolve()
    require(parent == Path(PARENT).resolve(), 'Requires the fixed failed calibration parent')
    for name, digest in IDENTITY.items():
        require(sha(package / name) == digest, 'Unsupported installed version: '+name)
    plan, manifest, status = [read(parent/name) for name in ('plan.json', 'manifest.json', 'summary.json')]
    require(manifest.get('plan') == plan, 'Parent manifest/plan mismatch')
    expected = dict(format=PARENT_FORMAT, variant='calibration', effective_mode='calibration', seed=3407,
                    endpoint_step=2500, read_weight=0, extension_files=IDENTITY,
                    scheduler_horizon=5000, maximum_lr=.0001, weight_decay=.005,
                    warmup_steps=500, grad_clip=1., optimizer='AdamW', support_arm='d',
                    post500_lr_multiplier=.2, fixed_train_count=450, test_accessed=False,
                    fresh_initialization=True, old_head_weights_reused=False,
                    historical_trajectory_claim=False, added_loss=None,
                    original_loss='exact runtime.loss (NLL plus Lovasz), unchanged')
    for key, value in expected.items():
        require(key in plan and plan[key] == value, 'Unexpected parent recipe: '+key)
    for key, value in dict(format=PARENT_FORMAT, variant='calibration', effective_mode='calibration',
                           seed=3407, phase='failed', complete=False, training_complete=False,
                           full_validation_evidence=False, smoke=False, test_accessed=False,
                           last_complete_step=LAST_COMPLETE_STEP, endpoint_step=2500,
                           failure_phase='update_804', current_image=TARGET_IMAGE).items():
        require(status.get(key) == value, 'Requires the known calibration804 failure: '+key)
    first = read(parent/'first_failure.json')
    require(first.get('phase') == 'update_804' and first.get('image') == TARGET_IMAGE and
            first.get('last_complete_step') == LAST_COMPLETE_STEP and
            first.get('tensors_modified') is False and first.get('automatic_retry') is False,
            'First failure does not match calibration804')
    require('device-side assert' in str(first.get('error')) and
            'device-side assert' in str(status.get('error')), 'Recorded failure is not the known CUDA assertion')
    require(manifest.get('format') == PARENT_FORMAT and manifest.get('seed') == 3407 and
            manifest.get('actual_training_names') == manifest.get('train_names') and
            manifest.get('fresh_initialization') is True and manifest.get('test_accessed') is False and
            manifest.get('support_arm') == 'd', 'Parent manifest scope differs')
    train_names, val_names = manifest['train_names'], manifest['val_names']
    require(isinstance(train_names, list) and isinstance(val_names, list) and
            all(isinstance(name, str) for name in train_names+val_names) and
            len(train_names) == len(set(train_names)) == 2918 and
            len(val_names) == len(set(val_names)) == 450, 'Train/Val scope differs')
    require(not set(train_names) & set(val_names) and TARGET_IMAGE in train_names, 'Train/Val overlap or failure image outside Train')
    fixed = sorted(random.Random(3407+191).sample(train_names, 450))
    require(manifest.get('fixed_train_names') == fixed, 'Fixed Train subset differs from original recipe')
    checkpoint = parent/'checkpoints/checkpoint_000800.pt'
    sidecar = read(checkpoint.with_suffix('.pt.json'))
    digest = sha(checkpoint)
    require(digest == CHECKPOINT_SHA256, 'Pinned checkpoint800 SHA differs')
    for record, label in ((sidecar, 'sidecar'), (status.get('last_checkpoint') or {}, 'summary')):
        require(record.get('step') == RESTORED_STEP and record.get('sha256') == digest and
                record.get('roundtrip_verified') is True and record.get('variant') == 'calibration' and
                Path(record.get('path', '')).resolve() == checkpoint,
                'Checkpoint800 '+label+' mismatch')
    rows = _rows(parent/'train.jsonl')
    require(len(rows) == LAST_COMPLETE_STEP, 'Expected exactly803 completed journal rows')
    for step, row in enumerate(rows, 1):
        require(type(row.get('step')) is int and row['step'] == step and row.get('optimizer_step') == step,
                'Nonconsecutive journal/optimizer step')
        require(row.get('epoch') == 0 and row.get('epoch_seed') == 3407 and
                row.get('variant') == 'calibration' and row.get('image') in train_names,
                'Journal epoch/variant/Train membership mismatch')
        require(all(_finite(row.get(key)) for key in ('original_loss', 'training_loss', 'grad_norm_preclip', 'lr')) and
                row['training_loss'] == row['original_loss'] and row.get('read_loss') == 0 and
                row.get('read_loss_evaluated') is False, 'Journal objective/scalars differ')
    paths = [parent/name for name in ('plan.json', 'manifest.json', 'summary.json', 'train.jsonl', 'first_failure.json')]
    paths += [checkpoint, checkpoint.with_suffix('.pt.json')]
    source = plan.get('source') or {}
    ledgers = source.get('ledgers') or {}
    require(isinstance(ledgers, dict) and ledgers, 'Missing source ledger')
    for path, expected_digest in ledgers.items():
        require(sha(path) == expected_digest, 'Source ledger changed: '+path)
        paths.append(Path(path))
    paths += [package/name for name in IDENTITY]
    spec = importlib.util.spec_from_file_location('_verified_calibration_parent_experiment', package/'experiment.py')
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    bridge = exp.load_file(Path(plan['addon_dir'])/'runtime_bridge.py', 'calibration_preflight_bridge')
    exp.verify_plan(plan, bridge)
    inherited, inherited_hashes = {}, {}
    for folder, step in (('initial_val', 0), ('val_step_000500', 500)):
        reference, fingerprints = _evaluation(parent, folder, step, val_names)
        inherited[folder] = reference
        inherited_hashes.update(fingerprints)
    info = dict(root=str(parent), fingerprints=inherited_hashes, inherited_evaluations=inherited,
                inherited_evaluations_reference_only=True, prior_replay_required=False,
                restored_step=RESTORED_STEP, discarded_unsaved_parent_steps=[801, 803],
                historical_failure_resolved=False, exact_historical_replay=False,
                test_accessed=False)
    return exp, plan, manifest, rows, checkpoint, {str(path): sha(path) for path in paths}, info
