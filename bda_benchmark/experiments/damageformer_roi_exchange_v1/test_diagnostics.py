import contextlib
import io
import unittest
import torch
from .diagnostics import DiagnosticAccumulator
from .inspect_results import print_audit


class DiagnosticTests(unittest.TestCase):
    def test_bins_endpoints_and_harm_rescue(self):
        y = torch.tensor([[[1,1,1,1,1,1,0,0,3,3,3]]])
        b = torch.tensor([[[[i/10 for i in range(11)]]]])
        base = y.clone(); base[0,0,8] = 1
        other = y.clone(); other[0,0,0] = 3
        acc = DiagnosticAccumulator(pairs=[('hard','exchange')])
        acc.update(y,dict(hard=base,exchange=other),b,['event_000'])
        result = acc.finalize()
        groups = result['groups']
        counts = [sum(groups[f'b_bin_{i}__roi_{r}']['metrics']['hard']['count'] for r in ('true_positive','false_negative','false_positive','true_negative')) for i in range(10)]
        self.assertEqual(counts,[1]*9+[2])
        pair = groups['all']['comparisons']['hard__exchange']
        self.assertEqual(pair['harmed'],1); self.assertEqual(pair['rescued'],1)
        self.assertEqual(pair['harm_gt_to_wrong'][1][3],1)
        self.assertEqual(pair['rescue_gt_from_wrong'][3][1],1)
        self.assertEqual(pair['base_correct']+pair['base_wrong'],11)
        self.assertEqual(groups['roi_false_negative']['metrics']['hard']['count'],5)
        self.assertEqual(groups['roi_false_positive']['metrics']['hard']['count'],2)

    def test_native_nearest_and_explicit_pairs_are_local(self):
        y = torch.ones(1,2,4,dtype=torch.long)
        b = torch.tensor([[[[.1,.9]]]])
        a = DiagnosticAccumulator(pairs=[('a','b')])
        a.update(y,dict(a=y,b=y),b,['event_000'])
        self.assertEqual(a.finalize()['groups']['roi_false_negative']['metrics']['a']['count'],4)
        other = DiagnosticAccumulator()
        other.update(y,dict(s0=y,s2=y),b,['event_000'])
        self.assertIn('s0__s2',other.finalize()['groups']['all']['comparisons'])
        self.assertNotIn('a__b',other.finalize()['groups']['all']['comparisons'])

    def test_bad_pairs_and_prediction_change(self):
        y=torch.ones(1,2,2,dtype=torch.long); b=y[:,None].float()
        with self.assertRaises(ValueError):
            DiagnosticAccumulator(pairs=[('missing','s0')]).update(y,dict(s0=y),b,['a'])
        a=DiagnosticAccumulator(); a.update(y,dict(s0=y),b,['a'])
        with self.assertRaises(ValueError):
            a.update(y,dict(s2=y),b,['a'])

    def test_inspect_legacy_does_not_invent_bins(self):
        y=torch.tensor([[[1,3]]]); b=torch.ones(1,1,1,2)
        a=DiagnosticAccumulator(); a.update(y,dict(s0=y,s2=y.flip(-1)),b,['a'])
        data=a.finalize(); data['protocol'].pop('b_bins')
        output=io.StringIO()
        with contextlib.redirect_stdout(output): print_audit(data)
        self.assertIn('NOT AVAILABLE',output.getvalue())
        self.assertIn('Destroyed',output.getvalue())
        self.assertIn('harm_gt_to_wrong',output.getvalue())
        self.assertIn('"FN":',output.getvalue())


if __name__=='__main__': unittest.main()
