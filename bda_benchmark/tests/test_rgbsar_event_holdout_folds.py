import unittest

from bda_benchmark.script.standard_ML.build_RGBSAR_event_holdout_folds import (
    assign_event_folds,
    disaster_type,
    event_name,
)


class RGBSAREventHoldoutTests(unittest.TestCase):
    def test_name_parsers(self):
        self.assertEqual(event_name("la_palma-volcano_00000003"), "la_palma-volcano")
        self.assertEqual(disaster_type("la_palma-volcano"), "volcano")

    def test_folds_are_event_disjoint_and_complete(self):
        counts = {
            "a-earthquake": 12,
            "b-earthquake": 8,
            "c-flood": 6,
            "d-wildfire": 4,
            "e-explosion": 2,
        }
        folds = assign_event_folds(counts, 3)
        flattened = [event for fold in folds for event in fold]
        self.assertEqual(set(flattened), set(counts))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertTrue(all(fold for fold in folds))

    def test_invalid_fold_count_is_rejected(self):
        with self.assertRaises(ValueError):
            assign_event_folds({"a": 1, "b": 1}, 1)


if __name__ == "__main__":
    unittest.main()
