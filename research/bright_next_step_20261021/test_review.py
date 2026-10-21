import json
from pathlib import Path
import tempfile
import unittest

from review import collect, confusion_metrics, main, metric_values


class EvidenceTests(unittest.TestCase):
    def test_metrics_and_direction_use_GT_rows(self):
        cm = [[90, 5, 3, 2], [4, 80, 10, 6], [7, 20, 60, 13], [8, 3, 9, 80]]
        m = confusion_metrics(cm)
        self.assertEqual(m["damaged_to_intact_pixels"], 20)
        self.assertEqual(m["damaged_to_intact_pct_of_gt_class"], 20)
        self.assertEqual(m["intact_to_damaged_pct_of_gt_class"], 10)
        self.assertEqual(m["pixel_oa_pct"], 77.5)
        self.assertAlmostEqual(m["iou_pct"][2], 100 * 60 / 122)
        self.assertAlmostEqual(m["pixel_building_binary_f1_pct"], 200 * 281 / (562 + 10 + 19))

    def test_absent_class_policy_is_explicit(self):
        m = confusion_metrics([[10, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(m["iou_pct"], [100, None, None, None])
        self.assertEqual(m["miou4_pct"], 25)
        self.assertEqual(m["foreground_miou3_pct"], 0)

    def test_IoU_alone_cannot_invent_confusion(self):
        issues = []
        m = metric_values({"iou": [.969541, .761561, .431425, .600486]},
                          ["background", "intact", "damaged", "destroyed"], issues)
        self.assertAlmostEqual(m["miou4_pct"], 69.075325)
        self.assertNotIn("pixel_oa_pct", m)
        self.assertTrue(issues)
        with self.assertRaises(ValueError):
            metric_values({"iou": [96, 76, 43, 60]}, None, [])

    def test_refuse_silent_invalid_evidence(self):
        with self.assertRaises(ValueError):
            confusion_metrics([[1.1] * 4 for _ in range(4)])
        with self.assertRaises(ValueError):
            confusion_metrics([[0] * 4 for _ in range(4)])
        with self.assertRaises(ValueError):
            metric_values({"iou": [.9, .7, .4, .6]}, ["intact", "background", "damaged", "destroyed"], [])

    def test_collect_preserves_inputs_and_incomplete_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            folder = run / "train/val_step_001000"
            folder.mkdir(parents=True)
            result = {"split": "val", "step": 1000, "image_count": 450, "complete": False,
                      "final_integrity_pending": True, "class_names": ["background", "intact", "damaged", "destroyed"],
                      "metrics": {"control": {"confusion": [[5 if i == j else 1 for j in range(4)] for i in range(4)]}}}
            (folder / "result.json").write_text(json.dumps(result))
            (run / "train/manifest.json").write_text(json.dumps({"runtime_config": {"seed": 3407, "lr": 1e-4}}))
            (run / "train/summary.json").write_text(json.dumps({"complete": False, "phase": "failed", "last_complete_step": 1396}))
            before = {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()}
            report = collect([run])
            self.assertEqual(report["statuses"][0]["fields"]["last_complete_step"], 1396)
            row = report["metrics"][0]
            self.assertFalse(row["complete"])
            self.assertEqual(row["metadata"]["seed"], 3407)
            self.assertIn("report integrity pending", row["issues"])
            output = Path(tmp) / "review"
            self.assertEqual(main(["--root", str(run), "--output", str(output)]), 0)
            after = {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            with self.assertRaises(FileExistsError):
                main(["--root", str(run), "--output", str(output)])

    def test_corrupt_file_is_reported_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "result.json").write_text('{"metrics":')
            report = collect([root])
            self.assertEqual(len(report["read_errors"]), 1)
            self.assertEqual(report["metrics"], [])


if __name__ == "__main__":
    unittest.main()
