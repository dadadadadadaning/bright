"""Exercise read-only checking of already saved failed-run evaluations."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import resume_train
from test_evidence import ARMS, make_report, sha256, write_json, write_report


class SavedEvaluationCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names = ["val_%04d" % i for i in range(450)]
        report = make_report(1000, "val", self.names)
        self.path, self.report = write_report(
            self.root / "val_step_001000", report, self.names)
        write_json(self.root / "manifest.json", dict(
            format=resume_train.FORMAT, arms=ARMS, val_names=self.names,
            fixed_train_names=["fixed_train"],
            expected_evaluations=[
                dict(step=1000, split="val", image_count=450, primary=False),
                dict(step=1000, split="fixed_train", image_count=1, primary=False),
                dict(step=2500, split="val", image_count=450, primary=True),
            ]))
        self.args = SimpleNamespace(root=str(self.root))

    def snapshot(self):
        return {str(p.relative_to(self.root)): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in self.root.rglob("*") if p.is_file()}

    def test_checks_available_report_without_training_or_promoting(self):
        before = self.snapshot()
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(resume_train.check_evaluation(self.args), 0)
        self.assertIn("READ-ONLY CHECK PASSED: 1 report(s)", stream.getvalue())
        self.assertIn('"objective_journal_verified": true', stream.getvalue())
        self.assertEqual(self.snapshot(), before)
        self.assertIs(json.loads(self.path.read_text())["complete"], False)

    def test_rehashed_changed_component_is_rejected_without_file_mutation(self):
        for metrics in [self.report["metrics"], *self.report["per_event"].values()]:
            metrics["e"]["mean_image_class_nll_contribution"][0] += .001
        write_json(self.path, self.report)
        self.path.with_name("result.json.sha256").write_text(
            sha256(self.path) + "  result.json\n")
        before = self.snapshot()
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            resume_train.check_evaluation(self.args)
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
