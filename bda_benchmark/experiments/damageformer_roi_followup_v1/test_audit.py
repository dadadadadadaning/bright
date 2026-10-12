"""Synthetic CPU entry-point smoke; no benchmark files or source writes."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from . import audit_routing
from .common import routes
from .report import render
from bda_benchmark.experiments.damageformer_roi_exchange_v1.core import Readout
from bda_benchmark.experiments.damageformer_roi_partition_factorial_v1.test_core import reference


class AuditEntryContracts(unittest.TestCase):
    def test_reproduction_differences_reports_classwise_delta(self):
        metric = dict(valid_miou=.5, iou=[.1,.2,.3,.4],
                      precision=[.2,.3,.4,.5], recall=[.3,.4,.5,.6])
        changed = dict(metric, recall=[.3,.4,.50002,.6])
        report = audit_routing.reproduction_differences(
            {'old_region':changed}, {'old_region':metric})
        self.assertAlmostEqual(report['old_region']['recall']['delta'][2], 2e-5)
        self.assertAlmostEqual(report['old_region']['recall']['max_abs'], 2e-5)

    def test_real_audit_entry_and_default_report(self):
        torch.set_num_threads(1); torch.manual_seed(3407)
        ref = reference()
        feature = torch.randn(2,256,*ref['shape'])
        heads = {n: Readout(16).eval().requires_grad_(False) for n in ('hard','exchange')}
        bundle = dict(front=torch.nn.Identity().eval(),heads=heads,device=torch.device('cpu'),
            manifest=dict(arguments=dict(epsilon=.05,threshold=.5)),provenance={},val=['a','b'])
        ref['features']={'roi':feature}
        ref['pixel']=torch.randn(2,4,14,20)
        a,b = routes(ref,bundle['manifest'],'hard','hard')
        ref['prob']=heads['hard'](feature,a,(14,20))
        batch=(torch.randn(2,3,14,20),torch.randn(2,3,14,20),None,
               torch.randint(0,4,(2,14,20)),['bata-explosion_1','haiti-earthquake_2'])
        states={n:{k:v.clone() for k,v in h.state_dict().items()} for n,h in heads.items()}
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)/'audit'
            argv=['audit','--checkpoint','unused','--run_root','unused','--dataset','unused',
                  '--train_list','unused','--val_list','unused','--output',str(out),
                  '--smoke','--device','cpu','--workers','0','--heads','hard','--combinations','EE','EH']
            # The requested single-head slice must also work with default report.
            bundle['heads']={'hard':heads['hard']}
            with patch('sys.argv',argv), patch.object(audit_routing,'load_source',return_value=bundle), \
                 patch.object(audit_routing,'capture',return_value=ref), \
                 patch.object(audit_routing,'parse_batch',side_effect=lambda b,d:b), \
                 patch.object(audit_routing,'loader',return_value=[batch]), \
                 patch.object(audit_routing,'assert_source'):
                audit_routing.main()
            summary=json.loads((out/'summary.json').read_text())
            self.assertTrue(summary['complete']); self.assertTrue(summary['smoke'])
            self.assertEqual(summary['image_count'],2)
            self.assertIn('hard_EH',summary['final_metrics'])
            self.assertIn('PRIMARY hard_EH minus hard_EE',render(out))
        for n,h in heads.items():
            for k,v in h.state_dict().items(): self.assertTrue(torch.equal(states[n][k],v))
            self.assertTrue(all(p.grad is None for p in h.parameters()))


if __name__ == '__main__': unittest.main()
