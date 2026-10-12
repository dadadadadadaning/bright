"""Failure diagnostics must survive rejection; no optimizer/model edits in the gate."""
import copy
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from . import train
from .reproduction import build_policy


class InitialGateContracts(unittest.TestCase):
    def setUp(self):
        metric = dict(valid_miou=.5, iou=[.9,.6,.4,.5],
                      precision=[.9,.7,.6,.7], recall=[.9,.7,.6,.7])
        self.expected = {n:copy.deepcopy(metric) for n in ('old_pixel','old_region','hard')}
        self.actual = {n:copy.deepcopy(metric) for n in ('old_pixel','old_region','source_hard')}
        self.bundle = {'summary':{'final_metrics':self.expected}}
        self.args = SimpleNamespace(smoke=False, suite='structure')

    def test_failed_gate_saves_all_metrics_and_has_zero_optimizer_steps(self):
        self.actual['old_region']['precision'][2] += .002
        self.actual['source_hard']['recall'][3] += .003
        result = {'metrics':self.actual, 'image_count':450}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(train,'assert_source') as source:
            out = Path(tmp)
            with redirect_stdout(StringIO()), self.assertRaises(ValueError):
                train.initial_gate(self.bundle,result,self.args,out,build_policy(self.expected))
            self.assertEqual(json.loads((out/'initial.json').read_text()),result)
            check = json.loads((out/'reproduction.json').read_text())
            self.assertFalse(check['passed'])
            self.assertTrue(check['source_integrity_pass'])
            failed = [row for row in check['checks'] if not row['passed']]
            self.assertEqual({row['baseline'] for row in failed},{'old_region','source_hard'})
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['phase'],'baseline_check_failed')
            self.assertEqual(summary['optimizer_steps'],0)
            self.assertFalse(summary['complete'])
            self.assertFalse(list(out.glob('*.pth')))
            source.assert_called_once_with(self.bundle)

    def test_source_integrity_failure_is_not_hidden_by_metric_pass(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                train,'assert_source',side_effect=RuntimeError('Source parameters changed')):
            with redirect_stdout(StringIO()), self.assertRaises(RuntimeError):
                train.initial_gate(self.bundle,{'metrics':self.actual},self.args,Path(tmp),build_policy(self.expected))
            check=json.loads((Path(tmp)/'reproduction.json').read_text())
            self.assertFalse(check['source_integrity_pass'])
            self.assertFalse(check['passed'])
            self.assertTrue(check['metric_checks_passed'])

    def test_nonfinite_initial_metrics_save_evidence_before_rejection(self):
        self.actual['old_region']['precision'][2] = float('nan')
        self.actual['old_region']['recall'][2] = float('inf')
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(train,'assert_source'):
            out = Path(tmp)
            with redirect_stdout(StringIO()), self.assertRaises(ValueError):
                train.initial_gate(self.bundle,{'metrics':self.actual},self.args,out,build_policy(self.expected))
            initial = json.loads((out/'initial.json').read_text())
            self.assertEqual(initial['metrics']['old_region']['precision'][2], 'NaN')
            self.assertEqual(initial['metrics']['old_region']['recall'][2], 'Infinity')
            check = json.loads((out/'reproduction.json').read_text())
            self.assertFalse(check['passed'])
            self.assertFalse(check['metric_checks_passed'])
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['phase'], 'baseline_check_failed')
            self.assertEqual(summary['optimizer_steps'], 0)

    def test_smoke_never_claims_full_val_reproduction(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(train,'assert_source'):
            args=SimpleNamespace(smoke=True,suite='structure')
            with redirect_stdout(StringIO()):
                result=train.initial_gate(self.bundle,{'metrics':self.actual},args,Path(tmp),None)
            self.assertIsNone(result['passed'])
            self.assertEqual(result['status'],'smoke_not_full_val')

    def test_history_is_hashed_without_overwriting_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'validation.jsonl'
            rows=[dict(step=s,image_count=450,test_accessed=False,
                       metrics=copy.deepcopy(self.expected)) for s in (500,1000,1500)]
            rows[0]['metrics']['old_region']['precision'][2] += 2e-5
            path.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
            before=path.read_bytes()
            bundle=dict(summary=dict(final_step=1500,final_metrics=self.expected),
                        provenance={'source_file_sha256':{}})
            args=SimpleNamespace(smoke=False,run_root=tmp)
            policy=train.reproduction_policy(bundle,args)
            self.assertEqual(before,path.read_bytes())
            self.assertEqual(bundle['provenance']['source_file_sha256'][str(path)],policy['history_sha256'])
            self.assertEqual(policy['history_count'],3)


if __name__ == '__main__':
    unittest.main()
