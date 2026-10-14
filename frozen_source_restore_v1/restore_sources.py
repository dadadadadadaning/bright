#!/usr/bin/env python3
"""Audit/restore frozen E0 inputs by their original SHA256. Stdlib only.

Never edits the protocol, overwrites a target, loads a checkpoint, or trains.
Archives are streamed as data; their contents are never extracted wholesale.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import tempfile
import zipfile


DEFAULT_PROTOCOL = '/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_NIGHT_V1/run_20261012_161704_5dd0ec34/fresh_e0/protocol.json'
DEFAULT_BACKUP = '/root/shared-nvme/BRIGHT-master/New/source_restore_inputs'
CHUNK = 1024 * 1024
FORMAT = 'BRIGHT_FROZEN_SOURCE_RESTORE_V1'


def stream_hash(handle):
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(CHUNK), b''):
        digest.update(chunk)
    return digest.hexdigest()


def file_hash(path):
    with open(path, 'rb') as handle:
        return stream_hash(handle)


def safe_target(path, allowed):
    """Reject traversal and symlinks, including dangling symlinks in parents."""
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Target must be an absolute path without ..: ' + str(path))
    try:
        relative = path.relative_to(allowed)
    except ValueError:
        raise ValueError('Target is outside allowed-root: ' + str(path))
    current = allowed
    # An explicitly selected mount root may itself be a symlink on cloud hosts.
    # Only links below that trusted root are disallowed.
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Target path contains a symlink: ' + str(current))
    return path


def archive_name_safe(name):
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and '..' not in path.parts and '\\' not in name and not re.match(r'^[A-Za-z]:', name)


@contextlib.contextmanager
def candidate_stream(candidate):
    path = Path(candidate['path'])
    if path.is_symlink() or not path.is_file():
        raise ValueError('Backup source is not a regular file: ' + str(path))
    if candidate['kind'] == 'file':
        with path.open('rb') as handle:
            yield handle
    elif candidate['kind'] == 'zip':
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            info = members[candidate['index']]
            mode = (info.external_attr >> 16) & 0xFFFF
            if info.filename != candidate['member'] or not archive_name_safe(info.filename) or info.is_dir() or (stat.S_IFMT(mode) and not stat.S_ISREG(mode)):
                raise ValueError('ZIP member is changed or not a safe regular file')
            with archive.open(info) as handle:
                yield handle
    else:
        with tarfile.open(path, 'r:*') as archive:
            info = archive.getmembers()[candidate['index']]
            if info.name != candidate['member'] or not info.isfile() or not archive_name_safe(info.name):
                raise ValueError('TAR member is changed or not a safe regular file')
            with archive.extractfile(info) as handle:
                yield handle


def source_label(candidate):
    return candidate['path'] + ('::' + candidate['member'] if 'member' in candidate else '')


def collect_candidates(backups, names, issues):
    candidates = {name: [] for name in names}
    seen_files = set()

    def add(candidate, basename):
        if basename not in candidates:
            return
        try:
            with candidate_stream(candidate) as handle:
                candidate['sha256'] = stream_hash(handle)
            candidates[basename].append(candidate)
        except (OSError, ValueError, RuntimeError, EOFError, zipfile.BadZipFile, tarfile.TarError, IndexError) as exc:
            issues.append({'source': source_label(candidate), 'error': str(exc)})

    def inspect(path):
        path = Path(path)
        if str(path) in seen_files:
            return
        seen_files.add(str(path))
        if path.is_symlink():
            issues.append({'source': str(path), 'error': 'Backup symlink skipped'})
            return
        if not path.is_file():
            issues.append({'source': str(path), 'error': 'Backup is not a regular file'})
            return
        add({'kind': 'file', 'path': str(path)}, path.name)
        lower = path.name.lower()
        try:
            if lower.endswith('.zip'):
                with zipfile.ZipFile(path) as archive:
                    for index, info in enumerate(archive.infolist()):
                        basename = PurePosixPath(info.filename).name
                        if basename not in names or info.is_dir():
                            continue
                        add({'kind': 'zip', 'path': str(path), 'member': info.filename, 'index': index}, basename)
            elif lower.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz')):
                with tarfile.open(path, 'r:*') as archive:
                    for index, info in enumerate(archive.getmembers()):
                        basename = PurePosixPath(info.name).name
                        if basename not in names or info.isdir():
                            continue
                        add({'kind': 'tar', 'path': str(path), 'member': info.name, 'index': index}, basename)
        except (OSError, ValueError, EOFError, zipfile.BadZipFile, tarfile.TarError) as exc:
            issues.append({'source': str(path), 'error': str(exc)})

    for backup in backups:
        root = Path(os.path.abspath(backup))
        if not root.exists() or root.is_symlink():
            issues.append({'source': str(root), 'error': 'Backup path missing or symlink; upload original local backup here'})
            continue
        if root.is_file():
            inspect(root)
            continue
        def onerror(exc):
            issues.append({'source': str(exc.filename), 'error': str(exc)})
        for folder, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
            for dirname in list(dirs):
                child = Path(folder) / dirname
                if child.is_symlink():
                    dirs.remove(dirname)
                    issues.append({'source': str(child), 'error': 'Backup directory symlink skipped'})
            for name in sorted(files):
                inspect(Path(folder) / name)
    return candidates


def exclusive_install(temp_path, target, expected):
    """Hardlink publication is atomic and refuses existing targets.

    Some shared mounts reject hardlinks. O_EXCL is the compatible fallback;
    only an inode created by this invocation is ever removed on failure.
    """
    mode = 'hardlink_no_replace'
    created_stat = None
    try:
        try:
            os.link(temp_path, target)
            created_stat = os.lstat(target)
        except OSError as exc:
            unsupported = {errno.EXDEV, errno.EPERM, errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}
            if exc.errno not in unsupported:
                raise
            mode = 'exclusive_create_copy'
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
            fd = os.open(target, flags, 0o600)
            created_stat = os.fstat(fd)
            with os.fdopen(fd, 'wb') as output, open(temp_path, 'rb') as source:
                for chunk in iter(lambda: source.read(CHUNK), b''):
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        if file_hash(target) != expected:
            raise ValueError('Installed target SHA256 does not match protocol')
        return mode
    except BaseException:
        if created_stat is not None:
            try:
                now = os.lstat(target)
                if (now.st_dev, now.st_ino) == (created_stat.st_dev, created_stat.st_ino):
                    os.unlink(target)
            except FileNotFoundError:
                pass
        raise


def restore_candidate(candidate, target, expected, allowed):
    safe_target(target, allowed)
    target.parent.mkdir(parents=True, exist_ok=True)
    safe_target(target, allowed)
    if os.path.lexists(target):
        raise FileExistsError('Refusing to overwrite existing target: ' + str(target))
    fd, temporary = tempfile.mkstemp(prefix='.bright_restore_', dir=target.parent)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, 'wb') as output, candidate_stream(candidate) as source:
            for chunk in iter(lambda: source.read(CHUNK), b''):
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != expected:
            raise ValueError('Backup changed after search or SHA256 mismatch; nothing installed')
        safe_target(target, allowed)
        return exclusive_install(temporary, target, expected)
    finally:
        Path(temporary).unlink(missing_ok=True)


def inspect_inputs(protocol, allowed):
    if not isinstance(protocol, dict):
        raise ValueError('Protocol must be a JSON object')
    paths, fingerprints = protocol.get('input_paths'), protocol.get('input_fingerprints')
    if protocol.get('format') != 'SRAC_HARDQUERY_PROTOCOL_V1' or protocol.get('complete') is not True or protocol.get('scope') != 'all':
        raise ValueError('Expected complete E0 protocol: format=SRAC_HARDQUERY_PROTOCOL_V1, scope=all, complete=true')
    if not isinstance(paths, dict) or not paths or not isinstance(fingerprints, dict) or set(paths) != set(fingerprints):
        raise ValueError('input_paths and input_fingerprints must be nonempty dictionaries with identical keys')
    rows = []
    for name, original in paths.items():
        row = {'name': name, 'path': original, 'expected_sha256': fingerprints[name]}
        rows.append(row)
        try:
            if not isinstance(original, str) or not isinstance(fingerprints[name], str) or not re.fullmatch('[0-9a-fA-F]{64}', fingerprints[name]):
                raise ValueError('Invalid path or SHA256 in protocol')
            row['expected_sha256'] = fingerprints[name].lower()
            target = safe_target(original, allowed)
            if not os.path.lexists(target):
                row['status'] = 'missing'
                continue
            if not target.is_file():
                raise ValueError('Existing target is not a regular file')
            row['actual_sha256'] = file_hash(target)
            row['status'] = 'verified' if row['actual_sha256'] == row['expected_sha256'] else 'conflict'
            if row['status'] == 'conflict':
                row['error'] = 'Existing file has wrong SHA256; will not overwrite'
        except (OSError, ValueError, TypeError) as exc:
            row.update(status='conflict', error=str(exc))
    return rows


def run(args):
    report = {'format': FORMAT, 'action': args.action, 'complete': False,
              'protocol': str(Path(args.protocol).absolute()), 'files': [], 'issues': [],
              'training': False, 'cuda_used': False, 'protocol_modified': False}
    protocol_path = Path(args.protocol)
    original_bytes = None
    report_writable = False
    try:
        allowed = Path(os.path.abspath(args.allowed_root))
        if not allowed.is_dir():
            raise ValueError('allowed-root must be an existing directory')
        report['allowed_root'] = str(allowed)
        report['canonical_allowed_root'] = str(allowed.resolve())
        if args.report:
            report_path = Path(args.report).absolute()
            if os.path.lexists(report_path):
                raise FileExistsError('Report must be a NEW file; refusing to overwrite: ' + str(report_path))
        original_bytes = protocol_path.read_bytes()
        report['protocol_sha256_before'] = hashlib.sha256(original_bytes).hexdigest()
        protocol = json.loads(original_bytes)
        report['files'] = inspect_inputs(protocol, allowed)
        if args.report:
            canonical_report = report_path.resolve()
            protected = [protocol_path, *[Path(row['path']) for row in report['files'] if isinstance(row['path'], str)]]
            for path in protected:
                canonical_source = path.resolve()
                if canonical_report == canonical_source or canonical_source in canonical_report.parents:
                    raise ValueError('Report must not occupy or create a directory at a frozen input path: ' + str(path))
            report_writable = True
        missing = [row for row in report['files'] if row['status'] == 'missing']
        if args.action == 'restore' and missing:
            backups = args.backup or [DEFAULT_BACKUP]
            report['backups'] = backups
            candidates = collect_candidates(backups, {Path(row['path']).name for row in missing}, report['issues'])
            for row in missing:
                found = candidates[Path(row['path']).name]
                row['candidates'] = [{'source': source_label(candidate), 'sha256': candidate['sha256']} for candidate in found]
                matches = [candidate for candidate in found if candidate['sha256'] == row['expected_sha256']]
                if not matches:
                    row['error'] = 'No original SHA256 match; upload the original local backup (file/directory/ZIP/TAR)'
                    continue
                candidate = matches[0]
                row['source'] = source_label(candidate)
                try:
                    row['install_mode'] = restore_candidate(candidate, Path(row['path']), row['expected_sha256'], allowed)
                    row['actual_sha256'] = file_hash(row['path'])
                    row['status'] = 'restored'
                except (OSError, ValueError, RuntimeError, EOFError, zipfile.BadZipFile, tarfile.TarError, IndexError) as exc:
                    row.update(status='conflict', error=str(exc))
        # Recheck every input, including files that existed before this invocation.
        final = {row['name']: row for row in inspect_inputs(protocol, allowed)}
        for row in report['files']:
            check = final[row['name']]
            if check['status'] != 'verified':
                row.update(check)
            else:
                row['actual_sha256'] = check['actual_sha256']
        report['complete'] = bool(report['files']) and all(row['status'] in ('verified', 'restored') for row in report['files'])
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        report['error'] = str(exc)
    finally:
        if original_bytes is not None:
            try:
                after = protocol_path.read_bytes()
                report['protocol_sha256_after'] = hashlib.sha256(after).hexdigest()
                report['protocol_unchanged'] = after == original_bytes
                if not report['protocol_unchanged']:
                    report['complete'] = False
                    report['issues'].append({'source': str(protocol_path), 'error': 'Protocol changed during operation; evidence is invalid'})
            except OSError as exc:
                report['complete'] = False
                report['protocol_unchanged'] = False
                report['issues'].append({'source': str(protocol_path), 'error': str(exc)})
    report['counts'] = dict(Counter(row['status'] for row in report['files']))
    payload = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.report and report_writable:
        try:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            with open(args.report, 'x', encoding='utf-8') as handle:
                handle.write(payload)
        except (OSError, ValueError) as exc:
            report['complete'] = False
            report['error'] = 'Could not write new report: ' + str(exc)
            payload = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    print(payload, end='')
    print('全部原输入校验通过；未运行训练。' if report['complete'] else '尚未恢复完整依赖；请将缺失文件的原始本地备份上传后重试，勿重建或伪造 manifest。', file=sys.stderr)
    return 0 if report['complete'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit', 'restore'))
    parser.add_argument('--protocol', default=DEFAULT_PROTOCOL)
    parser.add_argument('--backup', action='append', help='Repeat for backup directories, ZIP, TAR or TAR.GZ files')
    parser.add_argument('--allowed-root', default='/root/shared-nvme')
    parser.add_argument('--report', help='Optional NEW JSON report path; never overwritten')
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
