#!/usr/bin/env python3
"""Verify a BRIGHT code archive, then install only its new experiment folder.

Standard-library-only installer for Linux servers. This also works when the
archive's installer is executed with ``python -c``: argparse reads sys.argv.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import zipfile


PACKAGE = "damageformer_srac_memory_key_compare_v2"
PREFIX = "bda_benchmark/experiments/" + PACKAGE + "/"
MANIFEST = "PACKAGE_SHA256.json"
INSTALLER = "INSTALL_SERVER.py"
DEFAULT_ARCHIVE = "/root/shared-nvme/bright_memory_key_compare_v2.zip"
DEFAULT_REPO = "/root/shared-nvme/BRIGHT-master"
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
INSTALL_MARKER = ".install_in_progress"
UNSUPPORTED_RENAME_ERRNOS = {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP,
                             getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)}


class InstallError(RuntimeError):
    """An archive or destination failed the non-overwriting install contract."""


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
    path = PurePosixPath(name)
    if (not name or "\\" in name or "\x00" in name or path.is_absolute()
            or any(part in (".", "..") for part in name.split("/"))
            or path.as_posix() != name):
        raise InstallError("Unsafe archive path: " + repr(name))
    if name in (INSTALLER, MANIFEST):
        return
    if not name.startswith(PREFIX):
        raise InstallError("Archive file is outside the new experiment: " + name)
    filename = name[len(PREFIX):]
    if (not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", filename)
            or PurePosixPath(filename).suffix not in (".py", ".sh", ".md")):
        raise InstallError("Unexpected experiment file: " + name)


def read_verified_archive(archive):
    """Verify every ZIP member before inspecting or writing the destination."""
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
                        or file_type not in (0, stat.S_IFREG)
                        or member.flag_bits & 1 or member.file_size > MAX_FILE_BYTES):
                    raise InstallError("Unsafe ZIP member type, size or encryption: " + member.filename)
            if MANIFEST not in names:
                raise InstallError("Archive lacks " + MANIFEST)
            try:
                hashes = json.loads(zipped.read(MANIFEST).decode("utf-8"),
                                    object_pairs_hook=_unique_json_object)
            except (UnicodeDecodeError, ValueError) as error:
                raise InstallError("Invalid checksum manifest: " + str(error)) from error
            if not isinstance(hashes, dict) or not hashes or INSTALLER not in hashes:
                raise InstallError("Checksum manifest must map payload paths and the installer to SHA256")
            for name, expected in hashes.items():
                if not isinstance(name, str):
                    raise InstallError("Checksum manifest paths must be strings")
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
            if not {"__init__.py", "run.sh"}.issubset(payload):
                raise InstallError("Experiment payload lacks __init__.py or run.sh")
            return payload
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        if isinstance(error, InstallError):
            raise
        raise InstallError("Cannot verify archive: " + str(error)) from error


def _destination_identical(target, payload):
    if target.is_symlink() or not target.is_dir():
        raise InstallError("Existing destination is not a regular directory; preserving it: " + str(target))
    if INSTALL_MARKER not in payload and os.path.lexists(target / INSTALL_MARKER):
        raise InstallError("Existing destination has an incomplete-install marker; preserving it: " + str(target))
    actual = set()
    for item in target.iterdir():
        if item.name == "__pycache__" and item.is_dir() and not item.is_symlink():
            continue
        if item.is_symlink() or not item.is_file():
            raise InstallError("Existing destination contains an unexpected entry; preserving it: " + str(item))
        actual.add(item.name)
    if actual != set(payload):
        raise InstallError("Existing destination has a different file set; preserving it: " + str(target))
    for name, expected in payload.items():
        path = target / name
        if path.stat().st_size != len(expected) or _sha256(path.read_bytes()) != _sha256(expected):
            raise InstallError("Existing destination differs; preserving it: " + str(path))
    return True


def _rename_noreplace_function():
    """Prefer Linux's atomic no-overwrite rename; report unsupported operations."""
    if not sys.platform.startswith("linux"):
        raise InstallError("Safe atomic installation requires a Linux server with renameat2")
    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, "renameat2", None)
    if rename is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable")
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    # Retain the library through the closure for implementations with unloading.
    def move(source, target):
        _ = library
        result = rename(-100, os.fsencode(source), -100, os.fsencode(target), 1)
        if result != 0:
            number = ctypes.get_errno()
            if number in (errno.EEXIST, errno.ENOTEMPTY):
                raise InstallError("Destination appeared during installation; preserving it: " + str(target))
            raise OSError(number, os.strerror(number), str(target))
    return move


def _write_exclusive_at(directory_fd, name, data):
    """Create a new regular file in the opened directory, never truncate one."""
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=directory_fd)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o755 if name.endswith(".sh") else 0o644)


def _verify_exclusive_directory(target, directory_fd, payload):
    """Verify through the held directory descriptor, rejecting path replacement."""
    opened = os.fstat(directory_fd)
    current = os.stat(target, follow_symlinks=False)
    if (not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)):
        raise InstallError("Destination changed during installation; preserving partial files: " + str(target))
    names = set(os.listdir(directory_fd))
    if "__pycache__" in names:
        cache = os.stat("__pycache__", dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(cache.st_mode):
            names.remove("__pycache__")
    if names != set(payload):
        raise InstallError("Destination file set changed during installation; preserving partial files: " + str(target))
    for name, expected in payload.items():
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if (not stat.S_ISREG(before.st_mode) or before.st_size != len(expected)
                    or _sha256(stream.read()) != _sha256(expected)):
                raise InstallError("Installed file checksum differs; preserving partial files: " + name)
            after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
                raise InstallError("Installed file changed during verification; preserving partial files: " + name)


def _install_exclusive(target, payload):
    """Compatibility path: exclusive creation, visible marker, no partial cleanup.

    This path is intentionally not atomic. A failed copy remains in place with
    its marker so another installer cannot mistake it for a complete package.
    """
    try:
        os.mkdir(target, 0o700)
    except FileExistsError as error:
        raise InstallError("Destination appeared during installation; preserving it: " + str(target)) from error
    directory_fd = None
    try:
        directory_fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        marker = b"Non-atomic exclusive installation is in progress; do not run this experiment.\n"
        _write_exclusive_at(directory_fd, INSTALL_MARKER, marker)
        for name, data in payload.items():
            _write_exclusive_at(directory_fd, name, data)
        _verify_exclusive_directory(target, directory_fd, {**payload, INSTALL_MARKER: marker})
        os.fchmod(directory_fd, 0o755)
        os.unlink(INSTALL_MARKER, dir_fd=directory_fd)
    except (OSError, InstallError) as error:
        raise InstallError("Compatibility installation incomplete; preserving the partial destination and any marker: "
                           + str(target) + "; " + str(error)) from error
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def install(archive, repo):
    payload = read_verified_archive(Path(archive).expanduser())
    try:
        repository = Path(repo).expanduser().resolve(strict=True)
        parent = (repository / "bda_benchmark" / "experiments").resolve(strict=True)
    except OSError as error:
        raise InstallError("Repository experiment directory is unavailable: " + str(error)) from error
    if not repository.is_dir() or not parent.is_dir() or repository not in parent.parents:
        raise InstallError("Destination must be an existing experiment directory inside the selected repository")
    target = parent / PACKAGE
    if os.path.lexists(target):
        _destination_identical(target, payload)
        return "already_identical", target
    temporary = Path(tempfile.mkdtemp(prefix="." + PACKAGE + ".install-", dir=parent))
    try:
        for name, data in payload.items():
            path = temporary / name
            with path.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            path.chmod(0o755 if path.suffix == ".sh" else 0o644)
        temporary.chmod(0o755)
        _destination_identical(temporary, payload)
        try:
            _rename_noreplace_function()(temporary, target)
        except OSError as error:
            if error.errno not in UNSUPPORTED_RENAME_ERRNOS:
                raise InstallError("Atomic installation failed: " + str(error)) from error
            _install_exclusive(target, payload)
            return "installed_exclusive", target
        return "installed", target
    finally:
        # This random private folder is the only object cleanup may remove.
        if temporary.exists():
            shutil.rmtree(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    args = parser.parse_args(argv)
    try:
        status, target = install(args.archive, args.repo)
    except (InstallError, OSError) as error:
        print("Installation refused: " + str(error), file=sys.stderr)
        return 1
    print(("Existing experiment is byte-identical; no files changed: " if status == "already_identical"
           else "Installed verified new experiment: ") + str(target))
    if status == "installed_exclusive":
        print("Filesystem compatibility mode used exclusive file creation and checksum verification; installation was not atomic.")
    print("Only code was installed. No training, dataset access, checkpoint changes or Test evaluation ran.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
