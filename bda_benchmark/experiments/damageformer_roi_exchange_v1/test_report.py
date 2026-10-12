"""Read-only reports must expose class tradeoffs and missing evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from .report import class_counts, contrast, compact_audit, render
from bda_benchmark.experiments.damageformer_srac_predsplit_v1.audit import metrics


def fixture():
    a = [[70,10,15,5],[5,80,10,5],[5,10,80,5],[5,5,10,80]]
    b = [[92,3,5,0],[10,78,8,4],[6,8,82,4],[8,6,8,78]]
    harm = [[0,0,0,0],[4,0,2,0],[0,2,0,0],[4,0,0,0]]
    rescue = [[0,7,10,5],[0,0,4,0],[0,4,0,0],[0,2,0,0]]
    pair = dict(count=400,base_correct=310,base_wrong=90,harmed=12,rescued=32,net=20,
                harm_rate=12/310,recovery_rate=32/90,damaged_to_intact_harm=2,
                intact_to_damaged_rescue=4,harm_gt_to_wrong=harm,rescue_gt_from_wrong=rescue)
    bucket = dict(metrics={'hard':metrics(torch.tensor(a)), 'exchange':metrics(torch.tensor(b))},
                  comparisons={'hard__exchange':pair})
    return dict(image_count=450,groups={'all':bucket},per_event={'event':bucket},
                per_image=[{'image':'event_1'}],protocol={})


class ReportContracts(unittest.TestCase):
    def test_counts_and_iou_do_not_hide_recall_cost(self):
        m = fixture()['groups']['all']['metrics']
        self.assertEqual(class_counts(m['hard'],1),(80,25,20))
        self.assertEqual(class_counts(m['exchange'],1),(78,17,22))
        result = contrast(m['hard'],m['exchange'])
        for name in ('Intact','Destroyed'):
            self.assertEqual(result[name]['tp_change'],-2)
            self.assertEqual(result[name]['fn_change'],2)
            self.assertGreater(result[name]['iou_pp'],0)
            self.assertLess(result[name]['recall_pp'],0)
            self.assertGreater(result[name]['precision_pp'],0)
        self.assertEqual(result['Damaged']['fp_change'],-14)
        self.assertEqual(result['Damaged']['tp_change'],2)

    def test_compact_shows_gt_fp_origins_and_other_class_harm(self):
        text = '\n'.join(compact_audit(fixture()))
        self.assertIn('PRIMARY PAIR: exchange minus hard (same step)',text)
        self.assertIn('Damaged FP change by true class: {"Background": -10, "Intact": -2, "Destroyed": -2}',text)
        self.assertIn('Intact: {"Background": {"harm": 4, "rescue": 0}',text)
        self.assertIn('Destroyed: {"Background": {"harm": 4, "rescue": 0}',text)
        self.assertIn('B confidence bins unavailable',text)
        self.assertIn('cannot derive bins from aggregate confusion',text)
        self.assertIn('NOT additive mIoU components',text)

    def test_bins_show_only_nonempty_and_absent_pair_is_not_zero(self):
        data = fixture()
        data['protocol']['b_bins'] = [i/10 for i in range(11)]
        data['groups']['b_bin_9__roi_false_negative'] = copy.deepcopy(data['groups']['all'])
        empty = copy.deepcopy(data['groups']['all'])
        empty['comparisons']['hard__exchange']['count'] = 0
        data['groups']['b_bin_0__roi_true_positive'] = empty
        text = '\n'.join(compact_audit(data))
        self.assertIn('B[0.9,1.0] roi_false_negative:',text)
        self.assertNotIn('B[0.0,0.1) roi_true_positive:',text)
        absent = '\n'.join(compact_audit(data,'s0__s2'))
        self.assertIn('Requested primary pair unavailable; not treated as zero.',absent)

    def test_render_complete_fixed_step_and_routing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = fixture()
            manifest = dict(format='ROI_EXCHANGE_V1',smoke=False,epsilon=.05,
                            checkpoint_step=3000,arguments={'steps':5000})
            summary = dict(complete=True,smoke=False,final_step=5000,best={'hard':.8,'exchange':.9},
                           best_steps={'hard':1000,'exchange':2000})
            row = dict(step=5000,image_count=450,metrics=data['groups']['all']['metrics'],
                       routing={'hard':{'allocated':2048,'soft_occupied':1275},
                                'exchange':{'allocated':2048,'soft_occupied':2048}})
            for name, value in [('manifest.json',manifest),('summary.json',summary),
                                ('audit_final.json',data),('initial.json',{'hard_s2_parity':{'passed':True}})]:
                (root/name).write_text(json.dumps(value),encoding='utf-8')
            (root/'validation.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
            text = render(root)
            self.assertIn('Primary completed full-Val endpoint available: True',text)
            self.assertIn('"soft_occupied": 1275',text)
            self.assertIn('"passed": true',text)
            self.assertIn('Secondary best (do not use different best steps as primary)',text)
            self.assertIn('Intact: {"tp_change": -2',text)

    def test_incomplete_smoke_and_broken_tail_warn(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'manifest.json').write_text(json.dumps(dict(format='ROI_EXCHANGE_V1',smoke=True,
                                                             arguments={'steps':20})))
            (root/'summary.json').write_text('{"complete": false, "smoke": true}')
            (root/'validation.jsonl').write_text('{"step":')
            text = render(root)
            self.assertIn('complete: False; smoke: True',text)
            self.assertIn('Primary completed full-Val endpoint available: False',text)
            self.assertIn('Incomplete run',text)
            self.assertIn('Smoke validates execution only',text)
            self.assertIn('skipped incomplete/invalid line',text)
            self.assertIn('Final audit unavailable',text)


if __name__ == '__main__':
    unittest.main()
