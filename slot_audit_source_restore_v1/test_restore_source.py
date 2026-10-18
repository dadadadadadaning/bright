import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import restore_source as r


class RestoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / 'checkpoints'
        self.new = self.root / 'New'
        self.train_root = self.base / 'joint/run'
        self.train = self.train_root / 'train'
        self.qdir = self.base / 'sar/original/sar_q/formal/train_baseline'
        self.backup = self.base / 'sar/restored_20261014_134825_xno2hi6o/sar_q/formal/train_baseline'
        self.protocol = self.base / 'joint/source.json'
        self.new.mkdir()
        self.qdir.mkdir(parents=True)
        self.family = [self.qdir / f for f in ('last.pth', 'summary.json', 'manifest.json', 'validation.jsonl')]
        for i, path in enumerate(self.family):
            path.write_bytes(('original-%d' % i).encode())
        self.a1 = self.base / 'a1.pth'; self.a1.write_bytes(b'a1-original')
        self.e0 = self.base / 'e0.json'; self.e0.write_bytes(b'{"metadata":true}')
        self.source = dict(format='SOFT_MEMORY_Q_SOURCE_V1', test_accessed=False,
            historical_e0_source_reverified=False, q_checkpoint=str(self.family[0]),
            q_checkpoint_sha256=r.digest(self.family[0]), a1_checkpoint=str(self.a1),
            a1_sha256=r.digest(self.a1), e0_metadata_protocol=str(self.e0),
            e0_metadata_protocol_sha256=r.digest(self.e0), evidence_level='training_final_val450',
            evidence_sha256=r.digest(self.family[3]))
        self._bind()
        self.backup.mkdir(parents=True)
        for path in self.family:
            (self.backup / path.name).write_bytes(path.read_bytes())
        self.args = argparse.Namespace(train_root=str(self.train_root), checkpoint_root=str(self.base),
             new_root=str(self.new), step=500, search_root=[], restore=False, output=None)

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding='utf-8')

    def _bind(self):
        self.write(self.protocol, self.source)
        self.ledger = {str(p): r.digest(p) for p in self.family + [self.a1, self.e0, self.protocol]}
        ledger_path = self.train / 'input_fingerprints.json'
        self.write(ledger_path, self.ledger)
        self.write(self.train / 'summary.json', dict(format=r.FORMAT, complete=True,
            full_validation_evidence=True, smoke=False, input_fingerprints_sha256=r.digest(ledger_path)))
        info = dict(source_protocol=str(self.protocol), source_protocol_sha256=r.digest(self.protocol))
        self.write(self.train / 'manifest.json', dict(format=r.FORMAT, protocol_info=info))
        checkpoint = self.train / 'checkpoints/checkpoint_000500.pt'
        checkpoint.parent.mkdir(parents=True, exist_ok=True); checkpoint.write_bytes(b'joint-head-checkpoint')
        self.write(checkpoint.with_suffix('.pt.json'), dict(format=r.FORMAT, step=500,
            file=checkpoint.name, sha256=r.digest(checkpoint), protocol_info=info))

    def remove_family(self):
        for path in self.family:
            path.unlink()

    def run_restore(self):
        self.args.restore = True
        return r.execute(self.args)

    def test_complete_readonly(self):
        report, code = r.execute(self.args)
        self.assertEqual(code, 0); self.assertTrue(report['complete']); self.assertEqual(report['restored'], [])

    def test_readonly_plan_never_restores(self):
        self.remove_family()
        report, code = r.execute(self.args)
        self.assertEqual(code, 2)
        self.assertTrue(all(not p.exists() for p in self.family))
        self.assertEqual(sum(d['status'] == 'recoverable' for d in report['dependencies']), 4)

    def test_restore_exact_bytes_and_protocol_unchanged(self):
        protocol = self.protocol.read_bytes()
        self.remove_family()
        report, code = self.run_restore()
        self.assertEqual(code, 0); self.assertEqual(len(report['restored']), 4)
        self.assertEqual(self.protocol.read_bytes(), protocol)
        for p in self.family:
            self.assertEqual(r.digest(p), self.ledger[str(p)])
            self.assertFalse(p.is_symlink())

    def test_second_restore_is_idempotent(self):
        self.remove_family(); self.assertEqual(self.run_restore()[1], 0)
        report, code = self.run_restore()
        self.assertEqual(code, 0); self.assertEqual(report['restored'], [])

    def test_existing_wrong_file_aborts_every_copy(self):
        self.remove_family(); self.family[0].write_bytes(b'wrong')
        report, code = self.run_restore()
        self.assertEqual(code, 2); self.assertEqual(report['restored'], [])
        self.assertEqual(self.family[0].read_bytes(), b'wrong')
        self.assertFalse(self.family[1].exists())

    def test_wrong_hash_same_name_is_not_a_backup(self):
        self.remove_family(); (self.backup / 'last.pth').write_bytes(b'other-checkpoint')
        report, code = self.run_restore()
        self.assertEqual(code, 2); self.assertTrue(all(not p.exists() for p in self.family))
        q = next(d for d in report['dependencies'] if d['name'] == 'q_checkpoint')
        self.assertEqual(q['status'], 'missing')

    def test_missing_one_companion_prevents_partial_restore(self):
        self.remove_family(); (self.backup / 'manifest.json').unlink()
        self.assertEqual(self.run_restore()[1], 2)
        self.assertTrue(all(not p.exists() for p in self.family))

    def test_missing_a1_is_reported_not_restored(self):
        self.remove_family(); (self.backup / 'a1.pth').write_bytes(self.a1.read_bytes()); self.a1.unlink()
        report, code = self.run_restore()
        self.assertEqual(code, 2); self.assertFalse(self.a1.exists())
        self.assertTrue(all(not p.exists() for p in self.family))

    def test_corrupt_ledger_is_rejected(self):
        self.remove_family(); self.write(self.train / 'input_fingerprints.json', {})
        report, code = self.run_restore()
        self.assertEqual(code, 2); self.assertIn('ledger checksum', report['error'])

    def test_changed_source_protocol_is_rejected(self):
        self.remove_family(); self.protocol.write_text('{}')
        report, code = self.run_restore()
        self.assertEqual(code, 2); self.assertIn('protocol checksum', report['error'])

    def test_protocol_conflicting_with_bound_ledger_rejected(self):
        self.source['q_checkpoint_sha256'] = '0' * 64; self._bind(); self.remove_family()
        report, code = self.run_restore()
        self.assertEqual(code, 2); self.assertIn('disagree', report['error'])

    def test_no_symlink_destination_or_source(self):
        self.remove_family()
        self.family[0].symlink_to(self.backup / 'last.pth')
        self.assertEqual(self.run_restore()[1], 2)
        self.family[0].unlink()
        (self.backup / 'last.pth').rename(self.backup / 'actual.pth')
        (self.backup / 'last.pth').symlink_to(self.backup / 'actual.pth')
        self.assertEqual(self.run_restore()[1], 2)
        self.assertTrue(all(not p.exists() for p in self.family))

    def test_unrelated_search_root_rejected(self):
        self.args.search_root = [str(self.root / 'outside')]
        self.assertEqual(self.run_restore()[1], 2)
        self.args.search_root = [str(self.new / 'PFLlib-master')]
        self.assertEqual(self.run_restore()[1], 2)

    def test_copy_error_cleans_temporary_file(self):
        target = self.qdir / 'new.pth'
        with patch.object(r.shutil, 'copyfileobj', side_effect=OSError('copy failed')):
            with self.assertRaises(OSError):
                r.atomic_copy(self.family[0], target, r.digest(self.family[0]), self.base)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.qdir.glob('.q_source_restore_*')), [])

    def test_publish_race_never_overwrites_other_file(self):
        target = self.qdir / 'race.pth'
        link = r.os.link
        def concurrent_create(src, dst):
            Path(dst).write_bytes(b'concurrent-original'); return link(src, dst)
        with patch.object(r.os, 'link', side_effect=concurrent_create):
            with self.assertRaises(FileExistsError):
                r.atomic_copy(self.family[0], target, r.digest(self.family[0]), self.base)
        self.assertEqual(target.read_bytes(), b'concurrent-original')
        self.assertEqual(list(self.qdir.glob('.q_source_restore_*')), [])

    def test_independent_evidence_recovers_both_val_files(self):
        val = self.qdir.parent / 'val_baseline'
        val.mkdir()
        for filename in ('summary.json', 'result.json'):
            p = val / filename; p.write_bytes(filename.encode()); self.family.append(p)
        self.source.update(evidence_level='independent_val450', evidence_sha256=r.digest(val / 'summary.json'))
        self._bind()
        vback = self.backup.parent / 'val_baseline'; vback.mkdir()
        for p in self.family[-2:]:
            (vback / p.name).write_bytes(p.read_bytes())
        self.remove_family()
        report, code = self.run_restore()
        self.assertEqual(code, 0)
        self.assertEqual(len(report['restored']), 5)
        self.assertFalse((self.qdir / 'validation.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
