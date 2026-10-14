"""Filesystem recovery tests; synthetic files only, no BRIGHT/CUDA."""
import hashlib
import errno
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile

SCRIPT = Path(__file__).with_name('restore_sources.py')
SPEC = importlib.util.spec_from_file_location('restore_sources_under_test', SCRIPT)
REPAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPAIR)


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.allowed = self.root / 'server'
        self.allowed.mkdir()
        self.paths = {'joint_manifest.json': self.allowed / 'snapshot/joint/manifest.json',
                      'j1': self.allowed / 'snapshot/joint/j1_last.pth'}
        self.data = {'joint_manifest.json': b'{"original": true}\n', 'j1': b'original-weight-bytes'}
        self.protocol = self.allowed / 'protocol.json'
        self.record = {'format': 'SRAC_HARDQUERY_PROTOCOL_V1', 'scope': 'all', 'complete': True,
                       'input_paths': {k: str(v) for k,v in self.paths.items()},
                       'input_fingerprints': {k: hashlib.sha256(v).hexdigest() for k,v in self.data.items()}}
        self.write_protocol()
        self.report = self.root / 'result.json'
        self.backup = self.root / 'backup'
        self.backup.mkdir()

    def write_protocol(self):
        self.protocol.write_text(json.dumps(self.record))

    def call(self, action='restore', backups=None):
        command = [sys.executable, str(SCRIPT), action, '--protocol', str(self.protocol),
                   '--allowed-root', str(self.allowed), '--report', str(self.report)]
        if action == 'restore':
            for path in backups if backups is not None else [self.backup]:
                command.extend(['--backup', str(path)])
        result = subprocess.run(command, text=True, capture_output=True)
        return result

    def populate_raw(self):
        for key,path in self.paths.items():
            folder = self.backup / key.replace('.', '_')
            folder.mkdir()
            (folder / path.name).write_bytes(self.data[key])

    def assert_restored(self):
        for name,path in self.paths.items():
            self.assertEqual(path.read_bytes(), self.data[name])
        report=json.loads(self.report.read_text())
        self.assertTrue(report['complete'])
        self.assertTrue(report['protocol_unchanged'])

    def test_audit_reports_missing_without_creating_targets(self):
        result=self.call('audit')
        self.assertEqual(result.returncode,2,result.stderr)
        self.assertFalse(any(p.exists() for p in self.paths.values()))
        self.assertFalse(json.loads(self.report.read_text())['complete'])

    def test_report_cannot_occupy_frozen_input_or_create_it_as_directory(self):
        self.populate_raw()
        alias = self.root / 'server_alias'
        alias.symlink_to(self.allowed, target_is_directory=True)
        target = self.paths['joint_manifest.json']
        for action in ('audit', 'restore'):
            for report in (target, target / 'report.json', alias / target.relative_to(self.allowed)):
                with self.subTest(action=action, report=str(report)):
                    self.report = report
                    result = self.call(action)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn('Report must not occupy', result.stdout)
                    self.assertFalse(any(p.exists() for p in self.paths.values()))

    def test_restore_raw_exact_bytes_preserves_protocol(self):
        original=self.protocol.read_bytes()
        self.populate_raw()
        result=self.call()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assert_restored()
        self.assertEqual(self.protocol.read_bytes(),original)

    def test_nested_zip_discovery_and_prefix_independence(self):
        zpath=self.backup/'snapshot_backup.zip'
        with zipfile.ZipFile(zpath,'w') as z:
            for name,path in self.paths.items():
                z.writestr('backup-root/arbitrary-folder/'+path.name,self.data[name])
            z.writestr('../../escape.txt',b'forbidden')
        result=self.call()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assert_restored()
        self.assertFalse((self.root/'escape.txt').exists())

    def test_tar_gz_regular_members(self):
        archive=self.backup/'backup.tar.gz'
        with tarfile.open(archive,'w:gz') as t:
            for name,path in self.paths.items():
                payload=self.data[name]
                entry=tarfile.TarInfo('old-run/'+path.name)
                entry.size=len(payload)
                t.addfile(entry,io.BytesIO(payload))
        result=self.call()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assert_restored()

    def test_wrong_candidate_hash_never_restores(self):
        for path in self.paths.values():
            (self.backup/path.name).write_bytes(b'wrong bytes')
        result=self.call()
        self.assertEqual(result.returncode,2)
        self.assertFalse(any(p.exists() for p in self.paths.values()))

    def test_existing_wrong_file_is_never_overwritten(self):
        self.populate_raw()
        path=self.paths['j1']
        path.parent.mkdir(parents=True)
        path.write_bytes(b'existing unrelated weights')
        result=self.call()
        self.assertEqual(result.returncode,2)
        self.assertEqual(path.read_bytes(),b'existing unrelated weights')

    def test_all_present_needs_no_backup(self):
        for name,path in self.paths.items():
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(self.data[name])
        result=self.call(backups=[self.root/'nonexistent-backup'])
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assert_restored()

    def test_incomplete_protocol_cannot_authorize_restore(self):
        self.populate_raw()
        self.record['complete']=False
        self.write_protocol()
        result=self.call()
        self.assertEqual(result.returncode,2)
        self.assertFalse(any(p.exists() for p in self.paths.values()))

    def test_destination_outside_allowed_root_is_rejected(self):
        self.populate_raw()
        destination=self.root/'outside/manifest.json'
        self.record['input_paths']['joint_manifest.json']=str(destination)
        self.write_protocol()
        result=self.call()
        self.assertEqual(result.returncode,2)
        self.assertFalse(destination.exists())

    def test_broken_symlink_not_replaced(self):
        self.populate_raw()
        path=self.paths['j1']
        path.parent.mkdir(parents=True)
        path.symlink_to(self.allowed/'absent-target')
        result=self.call()
        self.assertEqual(result.returncode,2)
        self.assertTrue(path.is_symlink())
        self.assertFalse((self.allowed/'absent-target').exists())

    def test_symlinked_parent_cannot_escape_allowed_root(self):
        self.populate_raw()
        outside=self.root/'outside'
        outside.mkdir()
        (self.allowed/'snapshot').symlink_to(outside,target_is_directory=True)
        result=self.call()
        self.assertEqual(result.returncode,2)
        self.assertFalse((outside/'joint/manifest.json').exists())

    def test_wrong_same_name_candidate_does_not_hide_correct_copy(self):
        self.populate_raw()
        (self.backup/'manifest.json').write_bytes(b'unrelated manifest')
        result=self.call()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assert_restored()

    def test_zip_symlink_does_not_restore_target(self):
        archive=self.backup/'link.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for name,path in self.paths.items():
                item=zipfile.ZipInfo(path.name)
                item.create_system=3
                item.external_attr=(0o120777<<16)
                z.writestr(item,self.data[name])
        result=self.call()
        self.assertEqual(result.returncode,2)
        self.assertFalse(any(p.exists() for p in self.paths.values()))

    def test_existing_report_is_preserved(self):
        self.populate_raw()
        self.report.write_text('preserve report')
        result=self.call()
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(self.report.read_text(),'preserve report')

    def test_explicit_allowed_root_may_be_platform_symlink(self):
        physical=self.root/'physical-storage'
        self.allowed.rename(physical)
        self.allowed.symlink_to(physical,target_is_directory=True)
        self.populate_raw()
        result=self.call()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assert_restored()

    def test_shared_filesystem_link_fallback_preserves_bytes(self):
        temporary=self.root/'staged'
        temporary.write_bytes(b'verified staged bytes')
        expected=hashlib.sha256(temporary.read_bytes()).hexdigest()
        destination=self.allowed/'restored'
        with mock.patch.object(REPAIR.os,'link',side_effect=OSError(errno.EINVAL,'not supported')):
            mode=REPAIR.exclusive_install(temporary,destination,expected)
        self.assertEqual(mode,'exclusive_create_copy')
        self.assertEqual(destination.read_bytes(),temporary.read_bytes())

    def test_fallback_never_overwrites_concurrently_existing_target(self):
        temporary=self.root/'staged'
        temporary.write_bytes(b'source')
        destination=self.allowed/'existing'
        destination.write_bytes(b'keep existing')
        with mock.patch.object(REPAIR.os,'link',side_effect=OSError(errno.EINVAL,'not supported')):
            with self.assertRaises(FileExistsError):
                REPAIR.exclusive_install(temporary,destination,hashlib.sha256(b'source').hexdigest())
        self.assertEqual(destination.read_bytes(),b'keep existing')

    def test_bad_install_checksum_removes_only_created_target(self):
        temporary=self.root/'staged'
        temporary.write_bytes(b'wrong source')
        destination=self.allowed/'new-file'
        with self.assertRaises(ValueError):
            REPAIR.exclusive_install(temporary,destination,hashlib.sha256(b'expected').hexdigest())
        self.assertFalse(destination.exists())
        self.assertEqual(temporary.read_bytes(),b'wrong source')

    def test_backup_modified_after_search_rejected_before_install(self):
        source=self.backup/'j1_last.pth'
        source.write_bytes(b'changed since matching')
        candidate={'kind':'file','path':str(source),'sha256':self.record['input_fingerprints']['j1']}
        with self.assertRaisesRegex(ValueError,'changed|mismatch'):
            REPAIR.restore_candidate(candidate,self.paths['j1'],self.record['input_fingerprints']['j1'],self.allowed)
        self.assertFalse(self.paths['j1'].exists())


if __name__=='__main__':
    unittest.main()
