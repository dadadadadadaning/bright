#!/usr/bin/env python3
"""Restore original report bytes from the user's evidence collector, never weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

NEW = Path('/root/shared-nvme/BRIGHT-master/New')
ROOT = Path('/root/shared-nvme/BRIGHT-checkpoints/SRAC_SOFT_MEMORY_Q_REBASED_V1/run_20261014_202057_full_fEYBZv7D')
BACKUP = NEW / 'evidence_collect_20261016_032748_d1wmpcny'
FORMAT = 'SRAC_SOFT_MEMORY_Q_REBASED_V1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe(path):
    path = Path(os.path.abspath(os.path.expanduser(str(path))))
    if any(part.lower() in {'pfllib-master', 'pfllip-master', 'conda-envs'} for part in path.parts):
        raise ValueError('Unrelated protected path: ' + str(path))
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Symlink not accepted: ' + str(path))
    return path


def read(path):
    value = json.loads(safe(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value


def restore(backup, root, step=500, apply=False):
    backup, root = safe(backup), safe(root)
    inventory = read(backup / 'inventory.json')
    if inventory.get('format') != 'BRIGHT_JOINT_EVIDENCE_COLLECT_V1' or inventory.get('root') != str(root):
        raise ValueError('Backup is not bound to this original ABCD run')
    choices = {
        'train_summary': ('original/train/summary.json', root / 'train/summary.json'),
        'train_manifest': ('original/train/manifest.json', root / 'train/manifest.json'),
        'workflow_state': ('original/workflow_state.json', root / 'workflow_state.json'),
        'external_source': ('original/external.source.json', Path(str(root) + '.source.json')),
    }
    folder = 'final_val' if step == 5000 else 'val_step_%06d' % step
    choices['val_' + str(step)] = ('original/train/' + folder + '/result.json', root / 'train' / folder / 'result.json')
    planned, unavailable = [], []
    for key, (rel, destination) in choices.items():
        row = inventory.get('files', {}).get(key, {})
        if row.get('status') != 'copied':
            unavailable.append(key)
            continue
        expected = 'snapshots/' + rel
        if (row.get('kind') != 'report' or row.get('source') != str(destination)
                or row.get('snapshot') != expected or row.get('source_changed_during_read') is not False
                or row.get('json_status') != 'parsed_object' or not isinstance(row.get('source_before'), dict)
                or row.get('source_before') != row.get('source_after')):
            raise ValueError('Invalid backup identity or unstable capture: ' + key)
        source = safe(backup / expected)
        data = source.read_bytes()
        if sha(data) != row.get('snapshot_sha256') or len(data) != row.get('snapshot_bytes'):
            raise ValueError('Snapshot checksum/size mismatch: ' + key)
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError('Invalid original JSON: ' + key)
        if key in ('train_summary', 'workflow_state'):
            if (value.get('format') != FORMAT or value.get('complete') is not True
                    or value.get('full_validation_evidence') is not True or value.get('smoke') is not False):
                raise ValueError('Backup is not a completed formal report: ' + key)
        if key == 'train_manifest' and value.get('format') != FORMAT:
            raise ValueError('Wrong manifest format')
        if key.startswith('val_') and (value.get('image_count') != 450 or value.get('execution_complete') is not True):
            raise ValueError('Backup is not a full Val450 result: ' + key)
        destination = safe(destination)
        if destination.exists():
            if not destination.is_file() or destination.read_bytes() != data:
                raise ValueError('Existing original differs; will NOT overwrite: ' + str(destination))
            action = 'already_identical'
        else:
            action = 'restore_missing'
        planned.append((key, destination, data, action))
    # Validate all candidates before restoring any of them.
    restored = []
    for key, destination, data, action in planned:
        if apply and action == 'restore_missing':
            destination.parent.mkdir(parents=True, exist_ok=True)
            safe(destination)
            fd, temp = tempfile.mkstemp(prefix='.report_restore_', dir=destination.parent)
            try:
                with os.fdopen(fd, 'wb') as f:
                    f.write(data); f.flush(); os.fsync(f.fileno())
                os.link(temp, destination)  # Exclusive, never replaces existing destination.
            finally:
                Path(temp).unlink(missing_ok=True)
            if sha(destination.read_bytes()) != sha(data):
                raise ValueError('Post-copy mismatch: ' + str(destination))
            restored.append(str(destination))
    filename = 'checkpoint_last.pt' if step == 5000 else 'checkpoint_%06d.pt' % step
    required = [root / 'workflow_state.json', root / 'train/summary.json', root / 'train/manifest.json',
                root / 'train/input_fingerprints.json', root / 'train/checkpoints' / filename,
                root / 'train/checkpoints' / (filename + '.json'), Path(str(root) + '.source.json'),
                root / 'train' / ('final_val' if step == 5000 else 'val_step_%06d' % step) / 'result.json']
    missing = [str(p) for p in required if not safe(p).is_file() or p.stat().st_size == 0]
    return dict(format='SLOT_AUDIT_ORIGINAL_REPORT_RESTORE_V1', restore_requested=apply,
                training=False, cuda_used=False, test_accessed=False,
                restored=restored, backup_unavailable_keys=unavailable,
                plan=[dict(key=k, destination=str(p), action=a) for k, p, _, a in planned],
                missing_core_files=missing, core_files_present=not missing,
                ready_for_cuda=False,
                note='Only original report bytes restored. Model/bank hashes and Q dependencies still require the original audit checks.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backup', default=str(BACKUP))
    p.add_argument('--train-root', default=str(ROOT))
    p.add_argument('--step', type=int, choices=(500,1500,5000), default=500)
    p.add_argument('--restore', action='store_true')
    a = p.parse_args()
    try:
        report = restore(a.backup, a.train_root, a.step, a.restore)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print('恢复未完成：' + type(exc).__name__ + ': ' + str(exc))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['missing_core_files']:
        print('报告处理结束；仍缺上述原始文件。未启动训练或诊断。')
        return 3
    print('关键文件存在；下一步执行 Q 来源恢复与原审计校验。本步骤尚未验证模型可运行。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
