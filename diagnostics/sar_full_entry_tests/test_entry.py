import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ENTRY = Path(os.environ.get('SAR_FULL_ENTRY_PATH', str(Path(__file__).resolve().parents[2] / 'RUN_SAR_FULL_CHAIN_V1.py')))


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.entry = self.root / ENTRY.name
        self.entry.write_bytes(ENTRY.read_bytes())
        spec = importlib.util.spec_from_file_location('_sar_full_entry_test', self.entry)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_embedded_release_extracts_complete_original_tools(self):
        tool = self.module.extract(self.root)
        manifest = json.loads((tool/'RELEASE_SHA256.json').read_text())
        for name, sha in manifest.items():
            self.assertEqual(hashlib.sha256((tool/name).read_bytes()).hexdigest(), sha)
        for name in ('chain.py', 'SAR_FREEZE_WITNESS_V1.py', 'sar_q_matched_v1/pipeline.py',
                     'sar_soft_cross_v1/run.py', 'sar_soft_cross_v1/prerequisites.py'):
            self.assertTrue((tool/name).is_file())

    def test_corrupt_payload_rejected_without_extracting_files(self):
        self.module.PAYLOAD_SHA256 = '0' * 64
        before = set(self.root.iterdir())
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.module.extract(self.root)
        self.assertEqual(set(self.root.iterdir()), before)

    def test_unsafe_archive_path_is_rejected(self):
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w') as z:
            z.writestr('sar_full_chain_v1/../../escape.py', b'bad')
        raw = output.getvalue()
        self.module.PAYLOAD = base64.b85encode(raw).decode()
        self.module.PAYLOAD_SHA256 = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(ValueError, 'Invalid embedded'):
            self.module.extract(self.root)
        self.assertFalse((self.root.parent/'escape.py').exists())

    def test_plan_runs_offline_without_repo_or_cuda(self):
        result = subprocess.run([sys.executable, str(self.entry), '--plan', '--repo', '/nonexistent',
                                 '--protocol', '/nonexistent'], text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout[result.stdout.index('{'):])
        self.assertTrue(plan['no_training_started'])
        self.assertEqual(plan['target'], 'all')
        self.assertEqual(plan['q_updates_per_arm'], 2000)
        self.assertTrue(plan['wait_for_declared_memory'])

    def test_status_does_not_start_job_or_require_original_sources(self):
        root = self.root/'result'
        root.mkdir()
        (root/'launch.json').write_text(json.dumps(dict(q_root=str(root/'q'), soft_root=str(root/'soft'))))
        (root/'chain_state.json').write_text(json.dumps(dict(complete=False, phase='failed', failure='fixture')))
        result = subprocess.run([sys.executable, str(self.entry), '--status', '--root', str(root)],
                                 text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('WORKER_PROCESS_ALIVE=False', result.stdout)
        self.assertIn('fixture', result.stdout)
        self.assertFalse((root/'q').exists())

    def test_extract_only_does_not_start_job(self):
        result = subprocess.run([sys.executable, str(self.entry), '--extract-only'],
                                 text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('EXTRACT_ONLY', result.stdout)
        self.assertNotIn('"pid"', result.stdout)

    def test_bundled_original_clis_plan_without_new_training(self):
        tool = self.module.extract(self.root)
        q = subprocess.run([sys.executable, str(tool/'sar_q_matched_v1/pipeline.py'), 'plan'],
                           text=True, capture_output=True, timeout=20)
        self.assertEqual(q.returncode, 0, q.stderr)
        self.assertEqual(len(json.loads(q.stdout)['stages']), 21)
        soft = subprocess.run([sys.executable, str(tool/'sar_soft_cross_v1/run.py'), 'plan'],
                              text=True, capture_output=True, timeout=20)
        self.assertEqual(soft.returncode, 0, soft.stderr)
        self.assertEqual(len(json.loads(soft.stdout)['stages']), 11)


if __name__ == '__main__':
    unittest.main()
