#!/usr/bin/env python3
"""Verify and install the offline BRIGHT paired FP32 memory experiment without overwriting.

This installer uses only the Python standard library. It does not download,
import package code, or run a model. Failed copies remain marked and intact.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile


PACKAGE = "damageformer_srac_memory_rebuild_v3"
PREFIX = PACKAGE + "/"
INSTALLER = "INSTALL_MEMORY_REBUILD_V3.py"
MANIFEST = "PACKAGE_SHA256.json"
DEFAULT_ARCHIVE = "/root/shared-nvme/BRIGHT-master/New/bright_memory_rebuild_v3.zip"
DEFAULT_DEST = "/root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/" + PACKAGE
INSTALL_MARKER = ".install_in_progress"
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
REQUIRED_FILES = {
    "__init__.py", "shared.py", "paired.py", "core.py", "build_keys.py",
    "diagnose.py", "report.py", "run_pipeline.py", "run.sh",
    "RELEASE_SHA256.json", "README.md",
}


class InstallError(RuntimeError):
    """The archive or destination violates the non-overwriting contract."""


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InstallError("Duplicate checksum manifest key: " + str(key))
        result[key] = value
    return result


def _safe_name(name):
    if not isinstance(name, str):
        raise InstallError("Archive paths must be strings")
    path = PurePosixPath(name)
    if (not name or "\\" in name or "\x00" in name or path.is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))
            or path.as_posix() != name):
        raise InstallError("Unsafe archive path: " + repr(name))
    if name in (INSTALLER, MANIFEST):
        return
    if not name.startswith(PREFIX):
        raise InstallError("Archive file is outside the paired FP32 experiment directory: " + name)
    relative = name[len(PREFIX):]
    if ("__pycache__" in relative.split("/")
            or not all(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", part)
                for part in relative.split("/"))
            or PurePosixPath(relative).suffix not in (".py", ".sh", ".md", ".json")):
        raise InstallError("Unexpected tool file: " + name)


def verify_archive(archive):
    """Return verified relative payload bytes; perform no destination writes."""
    try:
        with zipfile.ZipFile(archive) as zipped:
            members = zipped.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                raise InstallError("Archive has duplicate member names")
            if not members or sum(member.file_size for member in members) > MAX_ARCHIVE_BYTES:
                raise InstallError("Archive is empty or exceeds the size limit")
            for member in members:
                _safe_name(member.filename)
                file_type = stat.S_IFMT(member.external_attr >> 16)
                if (member.orig_filename != member.filename or member.is_dir()
                        or file_type not in (0, stat.S_IFREG) or member.flag_bits & 1
                        or member.file_size > MAX_FILE_BYTES):
                    raise InstallError("Unsafe ZIP member type, size or encryption: " + member.filename)
            if MANIFEST not in names:
                raise InstallError("Archive lacks " + MANIFEST)
            try:
                hashes = json.loads(zipped.read(MANIFEST).decode("utf-8"),
                                    object_pairs_hook=_unique_json_object)
            except (UnicodeDecodeError, ValueError) as error:
                raise InstallError("Invalid checksum manifest: " + str(error)) from error
            if not isinstance(hashes, dict) or not hashes or INSTALLER not in hashes:
                raise InstallError("Manifest must map every payload file and installer to SHA256")
            for name, expected in hashes.items():
                _safe_name(name)
                if name == MANIFEST or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                    raise InstallError("Invalid SHA256 manifest entry: " + name)
            if set(names) != {MANIFEST, *hashes}:
                raise InstallError("ZIP members do not exactly match the checksum whitelist")
            payload = {}
            for name, expected in hashes.items():
                data = zipped.read(name)
                if _sha256(data) != expected:
                    raise InstallError("SHA256 mismatch: " + name)
                if name.startswith(PREFIX):
                    payload[name[len(PREFIX):]] = data
            if not REQUIRED_FILES.issubset(payload):
                raise InstallError("Payload lacks required paired FP32 experiment/release files")
            # A path cannot simultaneously denote a file and a directory.
            for name in payload:
                if any(str(parent) in payload for parent in PurePosixPath(name).parents if str(parent) != "."):
                    raise InstallError("Payload has a file/directory path conflict: " + name)
            return payload
    except (OSError, zipfile.BadZipFile, RuntimeError, EOFError) as error:
        if isinstance(error, InstallError):
            raise
        raise InstallError("Cannot verify archive: " + str(error)) from error


def _directory_flags():
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _open_parent(target):
    """Open each absolute parent component without following a symlink."""
    target = Path(os.path.abspath(os.path.expanduser(str(target))))
    if target.name in ("", ".", ".."):
        raise InstallError("Destination must name a new directory")
    descriptors = []
    links = []
    descriptor = os.open("/", _directory_flags())
    descriptors.append(descriptor)
    try:
        for part in target.parent.parts[1:]:
            child = os.open(part, _directory_flags(), dir_fd=descriptor)
            descriptors.append(child)
            links.append((descriptor, part, child))
            descriptor = child
        _check_links(links)
        return target, descriptor, descriptors, links
    except BaseException:
        for opened in reversed(descriptors):
            os.close(opened)
        raise


def _check_links(links):
    for parent_fd, name, child_fd in links:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(child_fd)
        if (not stat.S_ISDIR(current.st_mode)
                or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)):
            raise InstallError("Directory changed during installation; preserving existing/partial files: " + name)


def _write_exclusive_at(directory_fd, name, data):
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=directory_fd)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o755 if name.endswith(".sh") else 0o644)


def _sync_directory(directory_fd):
    """Accept explicitly unsupported directory sync; keep file sync strict."""
    try:
        os.fsync(directory_fd)
    except OSError as error:
        unsupported = {errno.EINVAL, errno.EOPNOTSUPP,
                       getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)}
        if error.errno not in unsupported:
            raise


def _regular_file_matches(directory_fd, name, data):
    before_path = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(before_path.st_mode):
        raise InstallError("Destination contains a non-regular file: " + name)
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if ((before.st_dev, before.st_ino) != (before_path.st_dev, before_path.st_ino)
                or before.st_size != len(data) or _sha256(stream.read()) != _sha256(data)):
            raise InstallError("Existing/installed file differs; preserving it: " + name)
        after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            raise InstallError("File changed during verification; preserving it: " + name)


def _check_cache(directory_fd, name, descriptors, links):
    """Ignore bytecode contents, but never accept linked or special entries."""
    child = os.open(name, _directory_flags(), dir_fd=directory_fd)
    descriptors.append(child)
    links.append((directory_fd, name, child))
    for item in os.listdir(child):
        info = os.stat(item, dir_fd=child, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise InstallError("Unexpected non-regular entry in __pycache__: " + item)


def _verify_tree(directory_fd, payload, descriptors, links, marker=None):
    files = {}
    directories = set()
    for path, data in payload.items():
        parts = path.split("/")
        if len(parts) == 1:
            files[path] = data
        else:
            directories.add(parts[0])
    if marker is not None:
        files[INSTALL_MARKER] = marker
    actual = set(os.listdir(directory_fd))
    if "__pycache__" in actual:
        _check_cache(directory_fd, "__pycache__", descriptors, links)
        actual.remove("__pycache__")
    if actual != set(files) | directories:
        raise InstallError("Destination has a different file/directory set; preserving it")
    for name, data in files.items():
        _regular_file_matches(directory_fd, name, data)
    for name in sorted(directories):
        child = os.open(name, _directory_flags(), dir_fd=directory_fd)
        descriptors.append(child)
        links.append((directory_fd, name, child))
        subset = {path[len(name) + 1:]: data for path, data in payload.items() if path.startswith(name + "/")}
        _verify_tree(child, subset, descriptors, links)
    _check_links(links)


def install(archive, dest=DEFAULT_DEST):
    """Install exclusively, or accept a byte-identical existing tree read-only."""
    payload = verify_archive(Path(archive).expanduser())
    descriptors = []
    root_fd = None
    existing = True
    marker_removed = False
    try:
        target, parent_fd, descriptors, links = _open_parent(dest)
        existing = False
        try:
            os.mkdir(target.name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            existing = True
        root_fd = os.open(target.name, _directory_flags(), dir_fd=parent_fd)
        descriptors.append(root_fd)
        links.append((parent_fd, target.name, root_fd))
        _check_links(links)
        if existing:
            if INSTALL_MARKER in os.listdir(root_fd):
                raise InstallError("Incomplete-install marker exists; preserving the partial destination: " + str(target))
            _verify_tree(root_fd, payload, descriptors, links)
            return "already_identical", target
        marker = b"Offline exclusive installation is incomplete. Do not run these tools.\n"
        _write_exclusive_at(root_fd, INSTALL_MARKER, marker)
        opened = {"": root_fd}
        for path, data in sorted(payload.items()):
            parts = path.split("/")
            current_path = ""
            directory_fd = root_fd
            for part in parts[:-1]:
                next_path = current_path + ("/" if current_path else "") + part
                if next_path not in opened:
                    _check_links(links)
                    os.mkdir(part, 0o755, dir_fd=directory_fd)
                    child = os.open(part, _directory_flags(), dir_fd=directory_fd)
                    descriptors.append(child)
                    links.append((directory_fd, part, child))
                    opened[next_path] = child
                directory_fd = opened[next_path]
                current_path = next_path
            _check_links(links)
            _write_exclusive_at(directory_fd, parts[-1], data)
        _verify_tree(root_fd, payload, descriptors, links, marker=marker)
        _check_links(links)
        os.fchmod(root_fd, 0o755)
        os.unlink(INSTALL_MARKER, dir_fd=root_fd)
        marker_removed = True
        _sync_directory(root_fd)
        _check_links(links)
        return "installed_exclusive", target
    except (OSError, InstallError) as error:
        if root_fd is not None and not existing and marker_removed:
            try:
                _write_exclusive_at(root_fd, INSTALL_MARKER,
                                    b"Installation failed after final verification. Do not run these tools.\n")
            except OSError:
                pass
        raise InstallError("Installation refused or incomplete; preserving existing/partial files: " + str(error)) from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--dest", default=DEFAULT_DEST)
    arguments = parser.parse_args(argv)
    try:
        status, target = install(arguments.archive, arguments.dest)
    except InstallError as error:
        parser.exit(1, "Installation refused: " + str(error) + "\n")
    print(status + ": " + str(target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
