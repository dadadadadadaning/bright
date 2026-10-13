"""Read-only evidence checks for the earlier R0/RT/RF endpoint.

Finding a process/checkpoint is not completion. Failed or intermediate runs
remain inspectable; no ledger is edited and no failed run is resumed here.
"""
from __future__ import annotations

import json
from pathlib import Path

from common import digest
from base_runtime import source_protocol

OLD_FORMAT = 'SRAC_REGION_MEMORY_JOINT_TRAIN_V1'
OLD_ARMS = ('r0', 'rt', 'rf')


def _read(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value


def _train_root(root):
    root = Path(root).expanduser().resolve()
    if root.name == 'summary.json':
        root = root.parent
    if (root / 'train/summary.json').is_file():
        return root / 'train'
    if (root / 'summary.json').is_file():
        return root
    raise FileNotFoundError('No R0/RT/RF training summary under ' + str(root))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _all_true_fields(value):
    """Integrity reports must contain successful boolean evidence, not empties."""
    if not isinstance(value, dict) or not value:
        return False
    if 'passed' in value:
        return value['passed'] is True
    leaves = []
    def walk(item):
        if isinstance(item, dict):
            for entry in item.values():
                walk(entry)
        elif isinstance(item, bool):
            leaves.append(item)
    walk(value)
    return bool(leaves) and all(leaves)


def inspect_baseline(args, root):
    train_root = _train_root(root)
    summary_path = train_root / 'summary.json'
    final_path = train_root / 'final_val/result.json'
    manifest_path = train_root / 'manifest.json'
    summary, final, manifest = (_read(p) for p in (summary_path, final_path, manifest_path))
    _require(summary.get('format') == OLD_FORMAT, 'Not the earlier three-arm joint experiment')
    _require(summary.get('complete') is True and summary.get('full_validation_evidence') is True,
             'Three-arm training has no complete formal endpoint')
    _require(summary.get('training_complete') is True and summary.get('smoke') is False,
             'Smoke or unfinished training cannot satisfy the baseline prerequisite')
    _require(summary.get('test_accessed') is False, 'Missing explicit no-Test record')
    _require(summary.get('updates_per_arm') == {a: 5000 for a in OLD_ARMS},
             'All R0/RT/RF arms must have exactly 5000 updates')
    _require(final.get('image_count') == 450 and final.get('full_validation_evidence') is True
             and final.get('primary_endpoint') is True and final.get('final_integrity_pending') is False,
             'The saved final Val450 is not the verified primary endpoint')
    _require(final.get('smoke') is False and final.get('test_accessed') is False,
             'Final result has incompatible smoke/Test identity')
    _require(final.get('saved_predictions_reopened') is True,
             'Final prediction evidence was not independently reopened')
    _require(_all_true_fields(summary.get('integrity')) and _all_true_fields(final.get('final_integrity')),
             'Frozen source/bank/runtime integrity did not pass')
    guard = summary.get('guard_report', {})
    _require(guard.get('restored') is True and not guard.get('failures') and not guard.get('restore_conflicts'),
             'Baseline pooling guard did not complete cleanly')
    metrics = final.get('metrics', {})
    _require(all(a in metrics and isinstance(metrics[a].get('foreground_miou'), (int, float))
                 and 0 <= metrics[a]['foreground_miou'] <= 1 for a in OLD_ARMS),
             'Final R0/RT/RF metrics are missing or invalid')
    _require(all(summary.get('metrics', {}).get(a) == metrics[a] for a in OLD_ARMS),
             'Summary and final metrics disagree')
    protocol = manifest.get('protocol_info', {})
    current_protocol = source_protocol(args)
    bank_path = Path(args.v3_root).resolve() / 'keys/paired_bank.pt'
    _require(protocol.get('source_protocol_sha256') == digest(current_protocol),
             'Baseline uses a different source E0 protocol')
    _require(protocol.get('paired_bank_sha256') == digest(bank_path),
             'Baseline uses a different FP32 parent bank')
    checkpoint_path = Path(summary.get('last_checkpoint') or '__missing_checkpoint__')
    if not checkpoint_path.is_absolute():
        checkpoint_path = train_root / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    _require(checkpoint_path.is_file(), 'Saved baseline last checkpoint is missing')
    checkpoint_meta_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.json')
    checkpoint = _read(checkpoint_meta_path)
    _require(checkpoint.get('format') == OLD_FORMAT and checkpoint.get('step') == 5000
             and checkpoint.get('mode') == 'train' and checkpoint.get('training_complete') is True,
             'Baseline checkpoint is not the 5000-step trained endpoint')
    _require(checkpoint.get('sha256') == digest(checkpoint_path), 'Baseline checkpoint hash mismatch')
    _require(summary.get('checkpoint') == checkpoint, 'Baseline checkpoint manifest binding differs')
    _require(checkpoint.get('protocol_info') == protocol,
             'Baseline checkpoint and training protocol differ')
    _require(checkpoint.get('source_hash') == manifest.get('source_hash'),
             'Baseline checkpoint and training code identities differ')
    files = {str(p.resolve()): digest(p) for p in
             (summary_path, final_path, manifest_path, checkpoint_path, checkpoint_meta_path)}
    input_path = train_root / 'input_fingerprints.json'
    _require(input_path.is_file() and digest(input_path) == summary.get('input_fingerprints_sha256'),
             'Baseline input provenance file is missing or changed')
    files[str(input_path.resolve())] = digest(input_path)
    # A caller may pass the pipeline root or its actual train subdirectory.
    # In either case inspect the associated state if it exists.
    state_path = (train_root.parent if train_root.name == 'train' else train_root) / 'workflow_state.json'
    if state_path.is_file():
        state = _read(state_path)
        _require(state.get('complete') is True and state.get('full_validation_evidence') is True,
                 'Three-arm pipeline state is still partial or failed')
        files[str(state_path.resolve())] = digest(state_path)
    return dict(passed=True, root=str(train_root.parent if train_root.name == 'train' else train_root),
                train_root=str(train_root), format=OLD_FORMAT, source_protocol_sha256=digest(current_protocol),
                paired_bank_sha256=digest(bank_path), files=files,
                updates_per_arm=summary['updates_per_arm'], metrics={a: metrics[a] for a in OLD_ARMS},
                interpretation='Completed prior three-arm endpoint; not a claim that Memory improved.')


def _candidate_paths(args):
    explicit = getattr(args, 'baseline_root', None)
    if explicit:
        return [Path(explicit).expanduser().resolve()]
    base = Path(args.v3_root).resolve().parent.parent
    candidates = set()
    if base.is_dir():
        for family in base.iterdir():
            if not family.is_dir() or not any(word in family.name.upper() for word in ('JOINT', 'RECOVERY')):
                continue
            for path in family.glob('run*/**/summary.json'):
                try:
                    data = _read(path)
                except (OSError, ValueError):
                    continue
                updates = data.get('updates_per_arm', {})
                if isinstance(updates, dict) and set(OLD_ARMS) <= set(updates):
                    candidates.add(path.parent)
    return sorted(candidates, key=lambda p: str(p), reverse=True)


def find_baselines(args):
    rows = []
    for root in _candidate_paths(args):
        row = dict(root=str(root), accepted=False)
        try:
            train = _train_root(root)
            summary = _read(train / 'summary.json')
            row.update(train_root=str(train), **{k: summary.get(k) for k in
                       ('complete', 'full_validation_evidence', 'updates_per_arm', 'phase', 'error', 'smoke')})
            evidence = inspect_baseline(args, root)
            row.update(accepted=True, evidence=evidence, reason='完整末步证据核对通过')
        except (OSError, ValueError, KeyError, TypeError) as error:
            row['reason'] = str(error)
        rows.append(row)
    return rows


def validate_baseline(args):
    rows = find_baselines(args)
    valid = [r for r in rows if r['accepted']]
    if valid:
        return valid[0]['evidence']
    reasons = '; '.join(r['root'] + ': ' + r['reason'] for r in rows[:3])
    raise ValueError('尚未找到通过最终核对的 R0/RT/RF 三组结果。先运行 find-baseline 查看；'
                     '可先运行 check 做新工具真实 20 步检查。正式训练需 --baseline-root 指向完成目录。'
                     + (' Details: ' + reasons if reasons else ' No candidate summary found.'))
