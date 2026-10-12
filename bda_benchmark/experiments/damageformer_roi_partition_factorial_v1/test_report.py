import json
import tempfile
import unittest
from pathlib import Path
from .report import contrasts, render


class ReportTests(unittest.TestCase):
    def test_factorial_math_and_missing(self):
        row = {'metrics': {k: {'valid_miou': v} for k, v in dict(g0=.5,g1=.52,s0=.53,s2=.56).items()}}
        result = contrasts(row)
        self.assertAlmostEqual(result['interaction (s2-s0)-(g1-g0)'], 1.)
        self.assertAlmostEqual(result['Q main effect'], 3.5)
        self.assertIsNone(result['hardening without split (h0-s0)'])
        del row['metrics']['g1']
        self.assertIsNone(contrasts(row)['interaction (s2-s0)-(g1-g0)'])

    def test_incomplete_summary_tail_and_missing_subsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'manifest.json').write_text(json.dumps({'smoke': False, 'arguments': {'steps': 5000, 'variants': ['g0']}}))
            (root/'summary.json').write_text('{"complete": false}')
            row = {'step': 500, 'image_count':450, 'metrics': {'g0': {'valid_miou': .5}}, 'routing': {'g0': {'hard_occupied': 23}}}
            (root/'validation.jsonl').write_text(json.dumps(row)+'\n{"step":')
            result = render(root)
            self.assertIn('summary.complete: False', result)
            self.assertIn('completed full-Val fixed-step evidence: False', result)
            self.assertIn('skipped incomplete/invalid line', result)
            self.assertIn('unavailable (not zero)', result)
            self.assertIn('"hard_occupied": 23', result)

    def test_complete_final_and_smoke_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {'smoke':False, 'arguments': {'steps':20, 'variants':['g0']}}
            (root/'manifest.json').write_text(json.dumps(manifest))
            (root/'summary.json').write_text('{"complete": true}')
            row = {'step':20, 'image_count':450, 'metrics':{'g0':{'valid_miou':.5}}}
            (root/'validation.jsonl').write_text(json.dumps(row))
            self.assertIn('completed full-Val fixed-step evidence: True', render(root))
            manifest['smoke'] = True
            (root/'manifest.json').write_text(json.dumps(manifest))
            self.assertIn('completed full-Val fixed-step evidence: False', render(root))

    def test_external_audit_and_same_step_not_best_contrast(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'manifest.json').write_text(json.dumps({'smoke': False, 'arguments': {'steps': 20, 'variants': ['g0','g1','s0','s2']}}))
            (root/'summary.json').write_text('{"complete": true}')
            values = [dict(g0=.5,g1=.6,s0=.7,s2=.8), dict(g0=.8,g1=.6,s0=.5,s2=.7)]
            rows = [dict(step=step,image_count=450,metrics={k:{'valid_miou':v} for k,v in value.items()}) for step,value in zip([10,20],values)]
            rows[-1]['diagnostics_file'] = 'audit_final.json'
            (root/'validation.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
            (root/'audit_final.json').write_text(json.dumps({'groups':{'empty':{'metrics':{'g0':{'count':0,'valid_miou':None}}}},'per_image':[{'image':'test'}]}))
            report = render(root)
            self.assertIn('interaction (s2-s0)-(g1-g0): +40.0000', report)
            self.assertIn('"valid_miou": null', render(root,details=True))
            self.assertIn('1 rows in audit_final.json', report)


if __name__ == '__main__':
    unittest.main()
