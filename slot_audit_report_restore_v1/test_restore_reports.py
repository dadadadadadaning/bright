import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from restore_reports import restore, FORMAT


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.base = Path(self.t.name)
        self.root = self.base / 'run'
        self.backup = self.base / 'backup'
        self.backup.mkdir()
        self.summary = dict(format=FORMAT, complete=True, full_validation_evidence=True, smoke=False)
        self.data = (json.dumps(self.summary, indent=3) + '\n').encode()
        self.source = self.backup / 'snapshots/original/train/summary.json'
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(self.data)
        self.destination = self.root / 'train/summary.json'
        self.inventory = dict(format='BRIGHT_JOINT_EVIDENCE_COLLECT_V1', root=str(self.root), files={
            'train_summary': dict(source=str(self.destination), kind='report', status='copied',
                snapshot='snapshots/original/train/summary.json', snapshot_sha256=hashlib.sha256(self.data).hexdigest(),
                snapshot_bytes=len(self.data), source_changed_during_read=False, json_status='parsed_object',
                source_before={'size': len(self.data)}, source_after={'size': len(self.data)})})
        self.write_inventory()

    def write_inventory(self):
        (self.backup / 'inventory.json').write_text(json.dumps(self.inventory))

    def test_restore_original_bytes_and_report_missing_weights(self):
        r = restore(self.backup, self.root, apply=True)
        self.assertEqual(self.destination.read_bytes(), self.data)
        self.assertIn(str(self.root / 'train/checkpoints/checkpoint_000500.pt'), r['missing_core_files'])
        self.assertFalse(r['ready_for_cuda'])
        self.assertFalse(r['training'])

    def test_readonly_does_not_create_original_directory(self):
        restore(self.backup, self.root)
        self.assertFalse(self.root.exists())

    def test_idempotent_for_identical_original(self):
        restore(self.backup, self.root, apply=True)
        r = restore(self.backup, self.root, apply=True)
        self.assertEqual(r['restored'], [])

    def test_conflicting_original_is_not_overwritten(self):
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(b'other original')
        with self.assertRaises(ValueError): restore(self.backup, self.root, apply=True)
        self.assertEqual(self.destination.read_bytes(), b'other original')

    def test_tampered_backup_rejected(self):
        self.source.write_bytes(b'{}')
        with self.assertRaises(ValueError): restore(self.backup, self.root, apply=True)
        self.assertFalse(self.root.exists())

    def test_other_run_rejected(self):
        self.inventory['root'] = '/different/run'
        self.write_inventory()
        with self.assertRaises(ValueError): restore(self.backup, self.root, apply=True)

    def test_partial_summary_is_not_promoted(self):
        data = json.dumps({**self.summary, 'complete': False}).encode()
        self.source.write_bytes(data)
        row = self.inventory['files']['train_summary']
        row.update(snapshot_sha256=hashlib.sha256(data).hexdigest(), snapshot_bytes=len(data))
        self.write_inventory()
        with self.assertRaises(ValueError): restore(self.backup, self.root, apply=True)
        self.assertFalse(self.root.exists())

    def test_symlink_source_rejected(self):
        dest = self.base / 'other.json'
        self.source.rename(dest)
        self.source.symlink_to(dest)
        with self.assertRaises(ValueError): restore(self.backup, self.root, apply=True)

    def test_capture_changed_is_rejected(self):
        self.inventory['files']['train_summary']['source_after'] = {'size': 1}
        self.write_inventory()
        with self.assertRaises(ValueError): restore(self.backup, self.root, apply=True)
        self.assertFalse(self.root.exists())

    def test_inventory_cannot_redirect_source(self):
        self.inventory['files']['train_summary']['snapshot'] = '../../other.json'
        self.write_inventory()
        with self.assertRaises(ValueError): restore(self.backup, self.root, apply=True)
        self.assertFalse(self.root.exists())


if __name__ == '__main__': unittest.main()
