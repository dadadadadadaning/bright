import unittest
import torch
from .diagnostics import DiagnosticAccumulator


class DiagnosticsContracts(unittest.TestCase):
    def test_transitions_roi_groups_ignore_and_events(self):
        gt = torch.tensor([[[2, 2], [0, 255]]])
        base = torch.tensor([[[2, 1], [0, 3]]])
        other = torch.tensor([[[1, 2], [1, 0]]])
        b = torch.tensor([[[[.2, .8], [.9, .5]]]])
        a = DiagnosticAccumulator()
        a.update(gt, {'old_pixel': base, 's0': base, 's2': other}, b, ['haiti-earthquake_00001'])
        r = a.finalize()
        row = r['groups']['all']['comparisons']['s0__s2']
        self.assertEqual(row['damaged_to_intact_harm'], 1)
        self.assertEqual(row['intact_to_damaged_rescue'], 1)
        self.assertEqual(row['harmed'], 2)
        self.assertEqual(row['rescued'], 1)
        self.assertEqual(row['count'], 3)
        self.assertEqual(r['groups']['roi_false_negative']['metrics']['s2']['count'], 1)
        self.assertEqual(r['groups']['roi_false_positive']['metrics']['s2']['count'], 1)
        self.assertIsNone(r['groups']['roi_true_negative']['metrics']['s2']['valid_miou'])
        self.assertEqual(sum(x['metrics']['s2']['count'] for x in r['per_event'].values()), 3)
        self.assertEqual(r['image_count'], 1)

    def test_native_group_nearest_not_bilinear_and_no_mutation(self):
        gt = torch.ones(1, 4, 4, dtype=torch.long)
        pred = gt.clone()
        building = torch.tensor([[[[0., 1.], [1., 0.]]]], requires_grad=True)
        a = DiagnosticAccumulator()
        a.update(gt, {'s0': pred}, building, ['event_1'])
        r = a.finalize()
        self.assertEqual(r['groups']['roi_false_negative']['metrics']['s0']['count'], 8)
        self.assertEqual(r['groups']['roi_true_positive']['metrics']['s0']['count'], 8)
        self.assertIsNone(building.grad)
        self.assertEqual(int(gt.sum()), 16)


if __name__ == '__main__':
    unittest.main()
