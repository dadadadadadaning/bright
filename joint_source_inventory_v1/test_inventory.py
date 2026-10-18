import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('inventory', Path(__file__).with_name('inspect_surviving_sources.py'))
inv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inv)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name) / 'checkpoints'
        self.new = Path(self.tmp.name) / 'New'
        self.base.mkdir()
        self.new.mkdir()

    def put(self, p, obj):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj))

    def test_missing_old_protocol_does_not_block_other_q(self):
        root = self.base / 'restored' / 'formal' / 'train_baseline'
        root.mkdir(parents=True)
        (root / 'last.pth').write_bytes(b'not a real tensor')
        self.put(root / 'summary.json', dict(complete=True, steps=2000, smoke=False))
        report = inv.collect(self.base, self.new)
        self.assertEqual(len(report['formal_q_candidates']), 1)
        self.assertFalse(report['source_verified'])
        self.assertFalse(report['weights_loaded'])
        self.assertEqual(report['formal_q_candidates'][0]['evidence_files'][1]['status'], 'missing')

    def test_smoke_and_best_not_formal_candidates(self):
        for rel in ('x/smoke/train_baseline/last.pth', 'x/formal/train_baseline/best.pth'):
            p = self.base / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'x')
        report = inv.collect(self.base, self.new)
        self.assertEqual(report['formal_q_candidates'], [])
        self.assertEqual(len(report['weight_candidates']), 2)

    def test_bad_json_and_protected_directory(self):
        (self.new / 'protocol.json').write_text('{')
        self.put(self.new / 'PFLlib-master' / 'protocol.json', {'format': 'SRAC_HARDQUERY_PROTOCOL_V1'})
        report = inv.collect(self.base, self.new)
        self.assertEqual(len(report['warnings']), 1)
        self.assertEqual(report['split_metadata_candidates'], [])

    def test_backup_metadata_and_missing_references(self):
        self.put(self.new / 'backup' / 'external_source.json', dict(format='SOFT_MEMORY_Q_SOURCE_V1', q_checkpoint=str(self.base / 'gone.pth')))
        self.put(self.new / 'backup' / 'protocol.json', dict(format='SRAC_HARDQUERY_PROTOCOL_V1', scope='all', arguments={'dataset': str(self.base)}, input_paths={'train_list': str(self.base / 'train.txt'), 'val_list': 'relative.txt'}))
        report = inv.collect(self.base, self.new)
        self.assertEqual(report['source_protocol_candidates'][0]['references']['q_checkpoint']['status'], 'missing')
        row = report['split_metadata_candidates'][0]
        self.assertEqual(row['dataset']['status'], 'directory')
        self.assertEqual(row['val_list']['status'], 'relative_path_unresolved')

    def test_symlink_not_traversed_and_bank_not_weight(self):
        (self.base / 'bank.pt').write_bytes(b'x')
        (self.new / 'linked').symlink_to(self.base, target_is_directory=True)
        report = inv.collect(self.base, self.new)
        self.assertEqual(len(report['bank_candidates']), 1)
        self.assertEqual(report['weight_candidates'], [])
        self.assertIsNone(report['bank_candidates'][0]['reported_format'])

    def test_missing_root_reported_without_claim_of_no_backups(self):
        report = inv.collect(self.base / 'absent', self.new)
        self.assertEqual(report['scan_roots'][0]['status'], 'missing')
        self.assertIn('本次扫描范围', inv.render(report))

    def test_common_a1_listed_even_with_many_bigger_weights(self):
        (self.base / 'best_val_miou.pth').write_bytes(b'x')
        for i in range(41):
            (self.base / ('head%d.pt' % i)).write_bytes(b'xxx')
        report = inv.collect(self.base, self.new)
        self.assertEqual(len(report['a1_name_candidates']), 1)
        self.assertIn('best_val_miou.pth', inv.render(report))


if __name__ == '__main__':
    unittest.main()
