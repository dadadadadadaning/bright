#!/usr/bin/env python3
"""Restore only byte-identical Q-source dependencies; no Torch, training or rewrites."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

DEFAULT_BASE = Path('/root/shared-nvme/BRIGHT-checkpoints')
DEFAULT_NEW = Path('/root/shared-nvme/BRIGHT-master/New')
DEFAULT_RUN = DEFAULT_BASE / 'SRAC_SOFT_MEMORY_Q_REBASED_V1/run_20261014_202057_full_fEYBZv7D'
FORMAT = 'SRAC_SOFT_MEMORY_Q_REBASED_V1'
BAD_PARTS = {'pfllib-master', 'pfllip-master', 'conda-envs', 'dataset', 'datasets', '.git'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_path(raw, roots=None):
    path = Path(os.path.abspath(os.path.expanduser(str(raw))))
    if any(part.lower() in BAD_PARTS for part in path.parts):
        raise ValueError('Protected/unrelated directory: ' + str(path))
    if roots is not None and not any(path.is_relative_to(root) for root in roots):
        raise ValueError('Path outside permitted roots: ' + str(path))
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Symlinks are not accepted: ' + str(path))
    return path


def read_json(path):
    data = json.loads(checked_path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return data


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dependency_plan(args):
    base = checked_path(args.checkpoint_root)
    new = checked_path(args.new_root)
    root = checked_path(args.train_root, [base])
    train = root / 'train'
    summary = read_json(train / 'summary.json')
    require(summary.get('format') == FORMAT and summary.get('complete') is True
            and summary.get('full_validation_evidence') is True
            and summary.get('smoke') is False, 'Original training is not formally complete')
    ledger_path = checked_path(train / 'input_fingerprints.json')
    require(digest(ledger_path) == summary.get('input_fingerprints_sha256'),
            'Original input ledger checksum mismatch')
    ledger = read_json(ledger_path)
    require(bool(ledger) and all(isinstance(k, str) and valid_sha(v) for k, v in ledger.items()),
            'Invalid original input ledger')
    manifest = read_json(train / 'manifest.json')
    filename = 'checkpoint_last.pt' if args.step == 5000 else 'checkpoint_%06d.pt' % args.step
    checkpoint = checked_path(train / 'checkpoints' / filename)
    meta = read_json(checkpoint.with_suffix('.pt.json'))
    require(meta.get('format') == FORMAT and meta.get('step') == args.step
            and meta.get('file') == filename and digest(checkpoint) == meta.get('sha256'),
            'Selected checkpoint/sidecar binding mismatch')
    info = manifest.get('protocol_info')
    require(manifest.get('format') == FORMAT and isinstance(info, dict)
            and info == meta.get('protocol_info'), 'Manifest/checkpoint protocol binding mismatch')
    protocol = checked_path(info['source_protocol'], [base, new])
    expected_protocol = info.get('source_protocol_sha256')
    require(valid_sha(expected_protocol) and ledger.get(str(protocol)) == expected_protocol
            and digest(protocol) == expected_protocol, 'Source protocol checksum mismatch')
    source = read_json(protocol)
    require(source.get('format') == 'SOFT_MEMORY_Q_SOURCE_V1'
            and source.get('test_accessed') is False
            and source.get('historical_e0_source_reverified') is False, 'Invalid Q source protocol')
    require(all(valid_sha(source.get(k)) for k in ('q_checkpoint_sha256', 'a1_sha256',
            'e0_metadata_protocol_sha256', 'evidence_sha256')), 'Incomplete source protocol SHA256 pins')
    q = checked_path(source['q_checkpoint'], [base])
    require(q.name == 'last.pth' and q.parent.name == 'train_baseline'
            and q.parent.parent.name == 'formal', 'Unexpected formal Q checkpoint path')
    required = []
    def add(name, path, pinned=None, restore=False):
        path = checked_path(path, [base] if restore else None)
        expected = ledger.get(str(path))
        require(valid_sha(expected), 'Missing bound original SHA256: ' + str(path))
        if pinned is not None:
            require(expected == pinned, 'Protocol and input ledger disagree: ' + str(path))
        required.append(dict(name=name, path=str(path), expected_sha256=expected,
                             restore_allowed=restore))
    add('q_checkpoint', q, source.get('q_checkpoint_sha256'), True)
    add('q_train_summary', q.parent / 'summary.json', restore=True)
    add('q_train_manifest', q.parent / 'manifest.json', restore=True)
    add('a1_checkpoint', source['a1_checkpoint'], source.get('a1_sha256'))
    add('e0_metadata_protocol', source['e0_metadata_protocol'],
        source.get('e0_metadata_protocol_sha256'))
    level = source.get('evidence_level')
    if level == 'training_final_val450':
        add('q_train_validation', q.parent / 'validation.jsonl', source.get('evidence_sha256'), True)
    elif level == 'independent_val450':
        val = q.parent.parent / 'val_baseline'
        add('q_val_summary', val / 'summary.json', source.get('evidence_sha256'), True)
        add('q_val_result', val / 'result.json', restore=True)
    else:
        raise ValueError('Unknown Q evidence level')
    roots = [base, new / 'source_restore_inputs'] + [checked_path(p, [base, new]) for p in args.search_root]
    roots = list(dict.fromkeys(checked_path(p, [base, new]) for p in roots))
    return required, roots, base, dict(source_protocol=str(protocol),
                source_protocol_sha256=expected_protocol, input_ledger=str(ledger_path),
                input_ledger_sha256=summary['input_fingerprints_sha256'], step=args.step)


def candidates(roots, names):
    if not names:
        return []
    found = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        require(root.is_dir(), 'Search root must be a directory: ' + str(root))
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d.lower() not in BAD_PARTS
                       and not (Path(directory) / d).is_symlink()]
            for name in set(files) & names:
                path = Path(directory) / name
                if path.is_symlink() or not path.is_file() or str(path) in seen:
                    continue
                seen.add(str(path)); found.append(path)
    preferred = 'restored_20261014_134825_xno2hi6o'
    return sorted(found, key=lambda p: (preferred not in p.parts, str(p)))


def atomic_copy(source, destination, expected, base):
    source = checked_path(source)
    destination = checked_path(destination, [base])
    require(not destination.exists(), 'Destination already exists: ' + str(destination))
    require(digest(source) == expected, 'Restore candidate changed: ' + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    checked_path(destination, [base])
    descriptor, temporary = tempfile.mkstemp(prefix='.q_source_restore_', dir=destination.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, 'wb') as target, source.open('rb') as original:
            shutil.copyfileobj(original, target, length=1024 * 1024)
            target.flush(); os.fsync(target.fileno())
        require(digest(temporary) == expected, 'Copied bytes do not match original SHA256')
        checked_path(destination, [base])
        os.link(temporary, destination)  # Atomic publication; never replaces an existing path.
    finally:
        temporary.unlink(missing_ok=True)


def execute(args):
    report = dict(format='SLOT_AUDIT_Q_SOURCE_RESTORE_V1', restore_requested=args.restore,
                  complete=False, training=False, cuda_used=False, test_accessed=False,
                  restored=[], dependencies=[], error=None)
    try:
        rows, roots, base, binding = dependency_plan(args)
        report.update(binding)
        report['dependencies'] = rows
        missing = []
        for row in rows:
            path = checked_path(row['path'])
            if path.is_file():
                row['actual_sha256'] = digest(path)
                row['status'] = 'verified' if row['actual_sha256'] == row['expected_sha256'] else 'conflict'
            elif path.exists():
                row['status'] = 'conflict'
            else:
                row['status'] = 'missing'
                if row['restore_allowed']:
                    missing.append(row)
        cache = {}
        for path in candidates(roots, {Path(row['path']).name for row in missing}):
            relevant = [r for r in missing if Path(r['path']).name == path.name and 'candidate' not in r]
            if not relevant:
                continue
            try:
                checked_path(path, roots)
                cache[str(path)] = digest(path)
            except (OSError, ValueError):
                continue
            for row in relevant:
                if cache[str(path)] == row['expected_sha256']:
                    row.update(candidate=str(path), status='recoverable')
        blocked = [r for r in rows if r['status'] not in ('verified', 'recoverable')]
        if blocked:
            report['error'] = 'Dependencies missing or conflicting; no restore performed'
            return report, 2
        if not args.restore and missing:
            report['error'] = 'Read-only plan: matching backups found; use --restore to copy original bytes'
            return report, 2
        if args.restore:
            # Recheck the whole plan before copying any file.
            for row in rows:
                path = checked_path(row['path'])
                if row['status'] == 'verified':
                    require(path.is_file() and digest(path) == row['expected_sha256'], 'Source changed during plan')
                else:
                    require(not path.exists() and digest(checked_path(row['candidate'])) == row['expected_sha256'],
                            'Restore plan changed; no overwrite permitted')
            for row in missing:
                atomic_copy(row['candidate'], row['path'], row['expected_sha256'], base)
                row['status'] = 'restored'
                report['restored'].append(row['path'])
        require(all(digest(checked_path(r['path'])) == r['expected_sha256'] for r in rows),
                'Final source verification failed')
        require(digest(report['source_protocol']) == report['source_protocol_sha256'], 'Source protocol changed')
        require(digest(report['input_ledger']) == report['input_ledger_sha256'], 'Input ledger changed')
        report['complete'] = True
        return report, 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        return report, 2


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--restore', action='store_true')
    p.add_argument('--train-root', default=str(DEFAULT_RUN))
    p.add_argument('--checkpoint-root', default=str(DEFAULT_BASE))
    p.add_argument('--new-root', default=str(DEFAULT_NEW))
    p.add_argument('--search-root', action='append', default=[])
    p.add_argument('--step', type=int, choices=(500, 1500, 5000), default=500)
    p.add_argument('--output', help='Optional new JSON report file under New; existing files are never overwritten')
    return p


def main():
    args = parser().parse_args()
    report, code = execute(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = checked_path(args.output, [checked_path(args.new_root)])
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            stream.write(text + '\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
