import unittest

from bda_benchmark.script.standard_ML.summarize_H7_A2_damaged_errors import (
    damaged_row,
    rows_from_payload,
)


class H7A2DamagedErrorTests(unittest.TestCase):
    def test_target_damaged_row_is_used(self):
        matrix = [
            [10, 0, 0, 0],
            [0, 10, 0, 0],
            [5, 20, 60, 15],
            [0, 0, 0, 10],
        ]
        row = damaged_row("sample-flood", matrix)
        self.assertEqual(row["gt_damaged_pixels"], 100)
        self.assertEqual(row["pred_intact_pct"], 20.0)
        self.assertEqual(row["pred_destroyed_pct"], 15.0)
        self.assertEqual(row["error_rate_pct"], 40.0)

    def test_test_accessed_report_is_rejected(self):
        with self.assertRaises(ValueError):
            rows_from_payload({"test_accessed": True, "events": []})


if __name__ == "__main__":
    unittest.main()
