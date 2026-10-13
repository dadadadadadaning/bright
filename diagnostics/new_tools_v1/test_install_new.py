"""Meaningful standard-library-only tests for the offline bundle installer."""
from __future__ import annotations

import hashlib
import errno
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile


SOURCE = Path(__file__).with_name("INSTALL_NEW.py")
SPEC = importlib.util.spec_from_file_location("new_installer_under_test", SOURCE)
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.archive = self.base / "tools.zip"
        self.target = self.base / installer.PACKAGE
        self.payload = {
            "RUN_TOOLS.py": b"print('launcher')\n",
            "RUN_MEMORY_REPLAY.py": b"print('memory')\n",
            "AUDIT_MEMORY_REPLAY_V2.py": b"print('audit')\n",
            "RELEASE_SHA256.json": b"{}\n",
            "prt/run.sh": b"#!/bin/sh\nexit 0\n",
            "region/night/README.md": b"Offline instructions.\n",
        }
        self.write_archive()

    def tearDown(self):
        self.temporary.cleanup()

    def write_archive(self, mutations=None, extra=None, duplicate=None,
                      manifest=None, special=None):
        files = {installer.PREFIX + name: data for name, data in self.payload.items()}
        files[installer.INSTALLER] = SOURCE.read_bytes()
        hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
        if mutations:
            files.update(mutations)
        if manifest is None:
            manifest = json.dumps(hashes).encode()
        files[installer.MANIFEST] = manifest
        if extra:
            files.update(extra)
        with zipfile.ZipFile(self.archive, "w", zipfile.ZIP_DEFLATED) as zipped:
            for name, data in files.items():
                info = zipfile.ZipInfo(name)
                info.external_attr = ((special if special is not None and name == installer.PREFIX + "RUN_TOOLS.py"
                                       else stat.S_IFREG | 0o644) << 16)
                zipped.writestr(info, data)
            if duplicate:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    zipped.writestr(duplicate, files[duplicate])

    def test_nested_install_and_read_only_repeat_accept_bytecode(self):
        status, target = installer.install(self.archive, self.target)
        self.assertEqual(status, "installed_exclusive")
        self.assertEqual(target, self.target)
        self.assertFalse((target / installer.INSTALL_MARKER).exists())
        for name, expected in self.payload.items():
            self.assertEqual((target / name).read_bytes(), expected)
        self.assertEqual((target / "prt/run.sh").stat().st_mode & 0o777, 0o755)
        cache = target / "region/night/__pycache__"
        cache.mkdir()
        (cache / "module.cpython-310.pyc").write_bytes(b"cached")
        before = (target / "RUN_TOOLS.py").stat().st_mtime_ns
        self.assertEqual(installer.install(self.archive, target)[0], "already_identical")
        self.assertEqual((target / "RUN_TOOLS.py").stat().st_mtime_ns, before)

    def test_modified_existing_file_preserved(self):
        installer.install(self.archive, self.target)
        path = self.target / "prt/run.sh"
        path.write_bytes(b"user modification")
        with self.assertRaises(installer.InstallError):
            installer.install(self.archive, self.target)
        self.assertEqual(path.read_bytes(), b"user modification")

    def test_existing_extra_file_or_empty_directory_refused(self):
        installer.install(self.archive, self.target)
        (self.target / "extra").mkdir()
        with self.assertRaises(installer.InstallError):
            installer.install(self.archive, self.target)
        self.assertTrue((self.target / "extra").is_dir())

    def test_partial_write_retains_marker_and_cannot_be_reinstalled(self):
        original = installer._write_exclusive_at

        def fail(directory_fd, name, data):
            if name == "RUN_TOOLS.py":
                raise OSError("simulated storage failure")
            return original(directory_fd, name, data)

        with mock.patch.object(installer, "_write_exclusive_at", side_effect=fail):
            with self.assertRaises(installer.InstallError):
                installer.install(self.archive, self.target)
        self.assertTrue((self.target / installer.INSTALL_MARKER).is_file())
        original_file = self.target / "AUDIT_MEMORY_REPLAY_V2.py"
        self.assertTrue(original_file.is_file())
        before = original_file.read_bytes()
        with self.assertRaises(installer.InstallError):
            installer.install(self.archive, self.target)
        self.assertEqual(original_file.read_bytes(), before)

    def test_unsafe_paths_rejected_before_destination_write(self):
        for name in ("/absolute.py", installer.PREFIX + "../outside.py",
                     installer.PREFIX + "a\\b.py", installer.PREFIX + "a//b.py",
                     installer.PREFIX + "./b.py", "unlisted.py"):
            with self.subTest(name=name):
                self.write_archive(extra={name: b"bad"})
                with self.assertRaises(installer.InstallError):
                    installer.install(self.archive, self.target)
                self.assertFalse(self.target.exists())

    def test_duplicate_zip_and_duplicate_manifest_keys_rejected(self):
        self.write_archive(duplicate=installer.PREFIX + "RUN_TOOLS.py")
        with self.assertRaises(installer.InstallError):
            installer.verify_archive(self.archive)
        self.write_archive(manifest=b'{"a":"1","a":"2"}')
        with self.assertRaises(installer.InstallError):
            installer.verify_archive(self.archive)

    def test_checksum_mismatch_and_exact_whitelist_enforced(self):
        self.write_archive(mutations={installer.PREFIX + "RUN_TOOLS.py": b"changed"})
        with self.assertRaises(installer.InstallError):
            installer.verify_archive(self.archive)
        self.write_archive(extra={installer.PREFIX + "extra.py": b"extra"})
        with self.assertRaises(installer.InstallError):
            installer.verify_archive(self.archive)
        del self.payload["RUN_MEMORY_REPLAY.py"]
        self.write_archive()
        with self.assertRaises(installer.InstallError):
            installer.verify_archive(self.archive)

    def test_symlink_member_and_size_limit_rejected(self):
        self.write_archive(special=stat.S_IFLNK | 0o777)
        with self.assertRaises(installer.InstallError):
            installer.verify_archive(self.archive)
        self.write_archive()
        with mock.patch.object(installer, "MAX_FILE_BYTES", 20):
            with self.assertRaises(installer.InstallError):
                installer.verify_archive(self.archive)
        with mock.patch.object(installer, "MAX_ARCHIVE_BYTES", 20):
            with self.assertRaises(installer.InstallError):
                installer.verify_archive(self.archive)

    def test_existing_target_and_parent_symlinks_not_followed(self):
        outside = self.base / "outside"
        outside.mkdir()
        sentinel = outside / "sentinel.py"
        sentinel.write_bytes(b"keep")
        self.target.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(installer.InstallError):
            installer.install(self.archive, self.target)
        self.assertEqual(list(outside.iterdir()), [sentinel])
        self.target.unlink()
        linked_parent = self.base / "linked"
        linked_parent.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(installer.InstallError):
            installer.install(self.archive, linked_parent / installer.PACKAGE)
        self.assertEqual(list(outside.iterdir()), [sentinel])

    def test_nested_directory_symlink_race_never_writes_outside(self):
        outside = self.base / "outside"
        outside.mkdir()
        sentinel = outside / "sentinel.py"
        sentinel.write_bytes(b"keep")
        original = installer._write_exclusive_at
        moved = self.base / "moved-prt"

        def replace_directory(directory_fd, name, data):
            if name == "run.sh":
                (self.target / "prt").rename(moved)
                (self.target / "prt").symlink_to(outside, target_is_directory=True)
            return original(directory_fd, name, data)

        with mock.patch.object(installer, "_write_exclusive_at", side_effect=replace_directory):
            with self.assertRaises(installer.InstallError):
                installer.install(self.archive, self.target)
        self.assertEqual(list(outside.iterdir()), [sentinel])
        self.assertTrue((self.target / installer.INSTALL_MARKER).is_file())
        self.assertEqual((moved / "run.sh").read_bytes(), self.payload["prt/run.sh"])

    def test_existing_payload_symlink_and_cache_symlink_refused(self):
        installer.install(self.archive, self.target)
        path = self.target / "RUN_TOOLS.py"
        path.unlink()
        path.symlink_to(self.archive)
        with self.assertRaises(installer.InstallError):
            installer.install(self.archive, self.target)
        path.unlink()
        path.write_bytes(self.payload["RUN_TOOLS.py"])
        (self.target / "__pycache__").symlink_to(self.base, target_is_directory=True)
        with self.assertRaises(installer.InstallError):
            installer.install(self.archive, self.target)

    def test_manifest_includes_installer_hash(self):
        files = {installer.PREFIX + name: data for name, data in self.payload.items()}
        hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
        self.write_archive(manifest=json.dumps(hashes).encode())
        with self.assertRaises(installer.InstallError):
            installer.verify_archive(self.archive)

    def test_unsupported_directory_sync_does_not_block_verified_install(self):
        original = installer.os.fsync

        def unsupported(descriptor):
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise OSError(errno.EINVAL, "directory fsync is unsupported")
            return original(descriptor)

        with mock.patch.object(installer.os, "fsync", side_effect=unsupported):
            self.assertEqual(installer.install(self.archive, self.target)[0], "installed_exclusive")
        self.assertFalse((self.target / installer.INSTALL_MARKER).exists())
        self.assertEqual(installer.install(self.archive, self.target)[0], "already_identical")

    def test_directory_sync_io_error_remains_failure_and_restores_marker(self):
        original = installer.os.fsync

        def fail(descriptor):
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise OSError(errno.EIO, "directory sync IO error")
            return original(descriptor)

        with mock.patch.object(installer.os, "fsync", side_effect=fail):
            with self.assertRaises(installer.InstallError):
                installer.install(self.archive, self.target)
        self.assertTrue((self.target / installer.INSTALL_MARKER).is_file())
        self.assertEqual((self.target / "RUN_TOOLS.py").read_bytes(), self.payload["RUN_TOOLS.py"])
        with self.assertRaises(installer.InstallError):
            installer.install(self.archive, self.target)


if __name__ == "__main__":
    unittest.main()
