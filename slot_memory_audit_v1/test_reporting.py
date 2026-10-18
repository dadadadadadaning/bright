import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from reporting import render_report, write_tables


class ReportingTests(unittest.TestCase):
    def fixture(self):
        m = dict(foreground_miou=.62, iou=[.95, .75, .45, .66],
                 precision=[.96, .8, .7, .8], recall=[.97, .8, .6, .8], valid_pixels=100)
        paired = dict(rescue_by_gt_class=[0, 2, 3, 1], harm_by_gt_class=[1, 1, 2, 0],
                      intact_correct_to_damaged=1, damaged_background_to_correct=1,
                      damaged_correct_to_background=2)
        p = dict(metrics={name: copy.deepcopy(m) for name in ('grid_probq', 'b', 'd')},
                 paired={'d_vs_b': paired})
        return dict(complete=True, execution_complete=True, full_validation_evidence=True,
                    smoke=False, checkpoint_step=500, image_count=450, training=False,
                    test_accessed=False, final_integrity_pending=False,
                    regions=dict(strata={'all_labeled': {
                        'feature_cosine': dict(n=0, weight_sum=0., weighted_sum=0., mean=None),
                        'slot1_ess': dict(n=2, weight_sum=4., weighted_sum=12., mean=3.),
                        'reader_d_damaged_label_mass': dict(n=2, weight_sum=4., weighted_sum=2., mean=.5),
                    }}), pixels={'all_labeled': p}, inventory={}, replay={'passed': True},
                    per_event={'event': {'pixels': {'all_labeled': p}}})

    def test_render_preserves_missing_and_pixel_units(self):
        text = render_report(self.fixture())
        self.assertIn('mean=NA; n=0; weight_sum=0.000000', text)
        self.assertIn('受损救回=3; 受损误伤=2', text)
        self.assertIn('reader_d_damaged_label_mass: mean=0.500000', text)
        self.assertIn('62.0000', text)
        self.assertIn('不是建筑', text)

    def test_csv_all_fields_and_event_roundtrip(self):
        data = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            write_tables(directory, data)
            root = Path(directory)
            with (root/'regions_summary.csv').open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            missing = next(row for row in rows if row['metric'] == 'feature_cosine')
            self.assertEqual(missing['mean'], '')
            self.assertEqual(missing['n'], '0')
            self.assertEqual(len(rows), 3)
            self.assertEqual(json.loads((root/'per_event.json').read_text()), data['per_event'])

    def test_strict_nonfinite_rejected_before_files(self):
        data = self.fixture()
        data['inventory']['invalid'] = float('nan')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_tables(directory, data)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_partial_report_does_not_claim_completion(self):
        report = render_report({'complete': False, 'image_count': 0})
        self.assertIn('complete: false', report)
        self.assertIn('尚无配对计数', report)


if __name__ == '__main__':
    unittest.main()
