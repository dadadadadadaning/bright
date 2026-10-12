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


PACKAGE = "damageformer_srac_memory_key_compare_v1"
PREFIX = "bda_benchmark/experiments/" + PACKAGE + "/"
MANIFEST = "PACKAGE_SHA256.json"
INSTALLER = "INSTALL_SERVER.py"
DEFAULT_ARCHIVE = "/root/shared-nvme/bright_memory_key_compare_v1.zip"
DEFAULT_REPO = "/root/shared-nvme/BRIGHT-master"
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024


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
    """Use Linux's atomic no-overwrite directory rename; never emulate overwrite."""
    if not sys.platform.startswith("linux"):
        raise InstallError("Safe atomic installation requires a Linux server with renameat2")
    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, "renameat2", None)
    if rename is None:
        raise InstallError("renameat2 is unavailable; no files were installed")
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
            raise InstallError("Atomic installation failed: " + os.strerror(number))
    return move


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
    rename = _rename_noreplace_function()
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
        rename(temporary, target)
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
    print(("Installed verified new experiment: " if status == "installed"
           else "Existing experiment is byte-identical; no files changed: ") + str(target))
    print("Only code was installed. No training, dataset access, checkpoint changes or Test evaluation ran.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
