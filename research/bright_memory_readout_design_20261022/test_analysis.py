import copy
import unittest
from analyze_results import metrics, analyze, CLASSES


class AnalysisTests(unittest.TestCase):
    def report(self):
        # Equal GT rows, one intact pixel corrected from background.
        a = [[10, 1, 0, 0], [2, 5, 1, 0], [1, 2, 3, 0], [0, 1, 1, 4]]
        b = copy.deepcopy(a)
        b[1][0] -= 1
        b[1][1] += 1
        return dict(complete=True, final_integrity_pending=False, phase="complete",
                    split="val", image_count=450, step=2500, class_names=CLASSES,
                    metrics={"b": {"confusion": a}, "d": {"confusion": b}})

    def test_binary_collapse_counts_and_delta(self):
        r = analyze(self.report(), "b", "d")
        self.assertEqual(r["metrics"]["b"]["building_false_negative"], 3)
        self.assertEqual(r["metrics"]["d"]["building_false_negative"], 2)
        self.assertEqual(r["net_confusion_count_changes"][1][0], -1)
        self.assertGreater(r["differences"]["building_iou_pp"], 0)

    def test_refuse_partial_pending_and_different_class_counts(self):
        for patch in ({"image_count": 97}, {"complete": False}, {"final_integrity_pending": True},
                      {"phase": "evaluation_complete_integrity_pending"}, {"smoke": True}):
            d = self.report(); d.update(patch)
            with self.assertRaises(ValueError): analyze(d, "b", "d")
        d = self.report(); d["metrics"]["d"]["confusion"][0][0] += 1
        with self.assertRaises(ValueError): analyze(d, "b", "d")

    def test_absent_classes_follow_original_fixed_denominator(self):
        result = metrics([[10, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(result["miou4_pct"], 25)
        self.assertEqual(result["oa_pct"], 100)


if __name__ == "__main__":
    unittest.main()
