import unittest
from .report import paired_bootstrap


class ReportContracts(unittest.TestCase):
    def rows(self):
        cm = [[100,0,0,0],[0,20,0,0],[0,0,10,0],[0,0,0,10]]
        return [dict(event='a' if i<2 else 'b',metrics={n:dict(confusion=cm) for n in ('x','y')}) for i in range(4)]

    def test_identical_predictions_zero_paired_interval(self):
        result = paired_bootstrap(self.rows(),'x__y',50)
        self.assertEqual(result['delta_pp'],0.)
        self.assertEqual(result['percentile_95_pp'],[0.,0.])
        self.assertEqual(result['event_count'],2)

    def test_support_mismatch_rejected(self):
        rows = self.rows()
        rows[0]['metrics']['y']={'confusion':[[99,0,0,0],[0,20,0,0],[0,0,10,0],[0,0,0,10]]}
        with self.assertRaises(ValueError): paired_bootstrap(rows,'x__y',20)


if __name__ == '__main__': unittest.main()
