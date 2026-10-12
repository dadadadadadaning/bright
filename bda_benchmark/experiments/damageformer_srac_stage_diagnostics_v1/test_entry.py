"""Entry contracts without BRIGHT images or checkpoint files."""
import tempfile
import unittest
import contextlib
import io
from pathlib import Path
from unittest import mock

import torch

from .shared import new_output, save_json
from . import feature_audit, report


class EntryContracts(unittest.TestCase):
    def test_pool_failure_snapshot_owns_cpu_storage(self):
        feature = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4).transpose(2, 3)
        q = torch.full((1, 2, 8), .5)
        ids = torch.zeros(2, 8, dtype=torch.long)
        head = torch.nn.Conv2d(3, 4, 1)
        saved = feature_audit.pool_failure_snapshot(feature, q, ids, 1, head)
        original = feature.clone()
        feature.zero_()
        q.zero_()
        ids.fill_(1)
        with torch.no_grad():
            head.weight.zero_()
        self.assertTrue(torch.equal(saved['tensors']['feature'], original))
        self.assertTrue(torch.equal(saved['tensors']['q'], torch.full((1, 2, 8), .5)))
        self.assertEqual(int(saved['tensors']['ids'].max()), 0)
        self.assertFalse(saved['layout']['feature']['contiguous'])
        self.assertTrue(all(v.device.type == 'cpu' and v.is_contiguous()
                            and not v.requires_grad for v in saved['tensors'].values()))
        self.assertNotIn('gt', saved['tensors'])

    def test_failure_snapshot_save_does_not_touch_cuda_or_overwrite(self):
        snapshot = feature_audit.pool_failure_snapshot(torch.ones(1, 3, 2, 2),
                    torch.ones(1, 1, 4), torch.zeros(1, 4, dtype=torch.long),
                    1, torch.nn.Conv2d(3, 4, 1))
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(torch.cuda, 'synchronize', side_effect=RuntimeError('poisoned CUDA')):
                record = feature_audit.save_failed_pool_inputs(Path(temp), snapshot, 'example')
            self.assertGreater(record['bytes'], 0)
            payload = torch.load(Path(temp)/record['file'], map_location='cpu', weights_only=True)
            self.assertEqual(payload['image'], 'example')
            self.assertTrue(torch.equal(payload['tensors']['q'], snapshot['tensors']['q']))
            with self.assertRaises(FileExistsError):
                feature_audit.save_failed_pool_inputs(Path(temp), snapshot, 'example')
            self.assertIsNone(feature_audit.save_failed_pool_inputs(Path(temp), None, 'example'))

    def test_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'new'
            new_output(path)
            save_json(path/'sentinel.json', {'keep': True})
            with self.assertRaises(FileExistsError):
                new_output(path)
            self.assertTrue((path/'sentinel.json').is_file())

    def test_required_inputs_and_no_training_flags(self):
        args = feature_audit.parser().parse_args([
            '--checkpoint', 'snapshot.pth', '--a1_checkpoint', 'a1.pth',
            '--dataset', 'images', '--train_list', 'train_set.txt',
            '--val_list', 'val_set.txt', '--output', 'unused', '--smoke'])
        self.assertTrue(args.smoke)
        self.assertFalse(args.embedding_interventions)
        self.assertNotIn('steps', vars(args))

    def test_failed_checks_not_silently_accepted(self):
        with self.assertRaisesRegex(RuntimeError, 'affine'):
            feature_audit.enforce_checks({'affine': {'passed': False, 'max_abs': .1}})

    def test_report_can_read_failed_run_without_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            save_json(Path(temp)/'summary.json', {'complete': False, 'gates_passed': False})
            with contextlib.redirect_stdout(io.StringIO()) as output:
                report.run(temp)
            self.assertIn('No completed audit.json', output.getvalue())

    def test_shell_has_no_implicit_pipeline(self):
        text = (Path(__file__).parent/'run.sh').read_text(encoding='utf-8')
        self.assertIn('smoke_feature|feature|smoke_memory|memory', text)
        self.assertIn('Output exists; preserve it', text)
        self.assertNotIn('nohup', text)

    def test_report_memory_schema_and_missing_denominators(self):
        from .memory_core import transition_report
        import torch
        cohort = dict(count=0, any_supported_pixels=0, support_mass_sum=0,
                      evidence_damaged=0, final_damaged=0, evidence_damaged_and_final_damaged=0)
        with tempfile.TemporaryDirectory() as temp:
            save_json(Path(temp)/'summary.json', dict(complete=True, gates_passed=True,
                      full_validation_evidence=False, smoke=True))
            save_json(Path(temp)/'audit.json', dict(metrics={'j1': dict(valid_miou=0.,
                      precision=[None]*4, recall=[None]*4, f1=[None]*4)},
                      transitions={'j1_to_j2': transition_report(torch.zeros(4, 4, 4, dtype=torch.long))},
                      difficult_cohorts={'j2': {'damaged_j1_wrong': cohort}}))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                report.run(temp)
            self.assertIn('SMOKE/SUBSET ONLY', output.getvalue())
            self.assertIn('fixed GT-Damaged/J1-wrong cohort=0', output.getvalue())
            self.assertNotIn('FAILED/INCOMPLETE gates', output.getvalue())

    def test_report_feature_reproduction_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            save_json(Path(temp)/'summary.json', dict(complete=False, gates_passed=False))
            save_json(Path(temp)/'audit.json', dict(reproduction=dict(passed=False, checks={
                'region': dict(passed=False, expected=.6, actual=.5, absolute_error=.1)})))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                report.run(temp)
            self.assertIn('FAILED/INCOMPLETE gates', output.getvalue())
            self.assertIn("'name': 'region'", output.getvalue())


if __name__ == '__main__':
    unittest.main()
