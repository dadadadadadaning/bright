"""CPU-only metadata contracts; no BRIGHT data, checkpoint tensors or CUDA."""
from __future__ import annotations

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("bright_joint_generalization_diagnostic_v1.py")
spec = importlib.util.spec_from_file_location("joint_generalization", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class GeneralizationMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def journal(self):
        path = self.root / "train.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for step in range(1, 5001):
                row = {"step": step, "image": "image_%04d" % ((step - 1) % 2918)}
                stream.write(json.dumps(row) + "\n")
        return path

    def test_fixed_subset_seen_before_first_checkpoint(self):
        picked, evidence = module.choose_seen_train(
            self.journal(), {"image_%04d" % x for x in range(2918)})
        self.assertEqual(len(picked), 450)
        self.assertEqual(picked[0], "image_0000")
        self.assertEqual(picked[-1], "image_0449")
        self.assertEqual(evidence["selected_exposure_counts"]["500"]["histogram"], {"1": 450})
        self.assertEqual(evidence["selected_exposure_counts"]["1500"]["histogram"], {"1": 450})
        self.assertEqual(evidence["selected_exposure_counts"]["5000"]["histogram"], {"2": 450})

    def test_duplicate_first500_is_rejected(self):
        journal = self.journal()
        lines = journal.read_text().splitlines()
        lines[1] = json.dumps({"step": 2, "image": "image_0000"})
        journal.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(ValueError, "500 distinct"):
            module.choose_seen_train(journal, {"image_%04d" % x for x in range(2918)})

    def test_val_tolerance_is_in_percentage_points(self):
        original = {path: {"iou": [.96, .76, .44, .66], "foreground_miou": .62}
                    for path in module.PATHS}
        current = {path: dict(row, iou=list(row["iou"])) for path, row in original.items()}
        current["d"]["iou"][2] += .00019  # 0.019 percentage points
        self.assertTrue(module.compare_iou(current, original)["passed"])
        current["d"]["iou"][2] += .00002
        self.assertFalse(module.compare_iou(current, original)["passed"])

    def test_val_class_presence_change_is_rejected(self):
        original = {path: {"iou": [.9, .7, None, .6], "foreground_miou": .433}
                    for path in module.PATHS}
        current = {path: dict(row, iou=list(row["iou"])) for path, row in original.items()}
        current["a"]["iou"][2] = 0.0
        with self.assertRaisesRegex(ValueError, "presence"):
            module.compare_iou(current, original)

    def test_confusion_differences_are_reported_separately(self):
        cm = [[0] * 4 for _ in range(4)]
        original = {path: {"iou": [.9, .7, .4, .6], "foreground_miou": .566,
                           "confusion": [row[:] for row in cm]} for path in module.PATHS}
        current = {path: {"iou": list(row["iou"]), "foreground_miou": row["foreground_miou"],
                          "confusion": [cells[:] for cells in cm]}
                   for path, row in original.items()}
        current["d"]["confusion"][0][1] = 2
        report = module.compare_iou(current, original)
        self.assertTrue(report["passed"])
        self.assertFalse(report["paths"]["d"]["confusion_exact"])
        self.assertEqual(report["paths"]["d"]["sum_abs_confusion_cell_delta"], 2)

    def test_print_report_includes_confusion_orientation(self):
        matrix = [[10, 1, 0, 0], [0, 9, 1, 0], [0, 2, 8, 0], [0, 0, 1, 9]]
        row = {"mean_image_loss": 0.2, "mean_image_nll": 0.15,
               "mean_image_lovasz": 0.05,
               "metrics": {"iou": [.9, .7, .5, .8],
                           "foreground_miou": 2/3, "confusion": matrix}}
        results = {"500": {split: {"metrics": {path: row for path in module.PATHS}}
                           for split in ("fixed_train", "val")}}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            module.print_report(results)
        self.assertIn("GT_rows=BG/Intact/Damaged/Destroyed", output.getvalue())
        self.assertIn("matrix=[[10,1,0,0],[0,9,1,0],[0,2,8,0],[0,0,1,9]]", output.getvalue())

    def test_checkpoints_are_predeclared(self):
        train = self.root / "train"
        self.assertEqual(module.checkpoint_path(train, 500).name, "checkpoint_000500.pt")
        self.assertEqual(module.checkpoint_path(train, 1500).name, "checkpoint_001500.pt")
        self.assertEqual(module.checkpoint_path(train, 5000).name, "checkpoint_last.pt")
        with self.assertRaisesRegex(ValueError, "fixed"):
            module.checkpoint_path(train, 1000)

    def test_plan_is_torch_free_and_does_not_create_output(self):
        root = self.root / "original"
        train = root / "train"
        train.mkdir(parents=True)
        tool = self.root / "tool"
        tool.mkdir()
        (tool / "run.py").write_text("", encoding="utf-8")
        (root / "workflow_state.json").write_text(json.dumps({"complete": True,
            "full_validation_evidence": True}))
        (train / "summary.json").write_text(json.dumps({"complete": True,
            "full_validation_evidence": True, "smoke": False}))
        names = ["image_%04d" % x for x in range(2918)]
        val = ["val_%04d" % x for x in range(450)]
        (train / "manifest.json").write_text(json.dumps({"source_hash": "source",
            "train_names": names, "val_names": val}))
        with (train / "train.jsonl").open("w") as stream:
            for step in range(1, 5001):
                stream.write(json.dumps({"step": step, "image": names[(step - 1) % len(names)]}) + "\n")
        for step in module.STEPS:
            checkpoint = module.checkpoint_path(train, step)
            checkpoint.parent.mkdir(exist_ok=True)
            checkpoint.write_bytes(b"checkpoint" + str(step).encode())
            checkpoint.with_suffix(checkpoint.suffix + ".json").write_text(json.dumps({
                "step": step, "mode": "train", "smoke": False,
                "training_complete": step == 5000, "sha256": module.file_sha256(checkpoint),
                "source_hash": "source", "protocol_info": {"x": 1}}))
            folder = train / ("final_val" if step == 5000 else "val_step_%06d" % step)
            folder.mkdir()
            (folder / "result.json").write_text(json.dumps({"image_count": 450,
                "execution_complete": True, "full_validation_evidence": step == 5000,
                "final_integrity_pending": step != 5000,
                "metrics": {path: {} for path in module.PATHS}}))
        static = module.validate_static(root, tool, count=450)
        self.assertEqual(static["selection"]["selected_count"], 450)
        self.assertEqual(len(static["checkpoint_checks"]), 3)
        self.assertFalse((root / "diagnosis").exists())


if __name__ == "__main__":
    unittest.main()
