import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import collect_joint_evidence_v1 as collector


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def metrics():
    return {"foreground_miou": .5, "iou": [.9, .6, .3, .6],
            "confusion": [[9, 0, 0, 0], [0, 6, 1, 0], [0, 1, 3, 0], [0, 0, 0, 6]]}


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "source"
        self.diagnostic = self.base / "diagnostic"
        self.output = self.base / "output"
        (self.root / "train").mkdir(parents=True)
        self.diagnostic.mkdir()

    def collect(self):
        report, archive = collector.collect(self.root, self.diagnostic, self.output)
        inventory = json.loads((report.parent / "inventory.json").read_text())
        return report, archive, inventory

    def test_snapshots_unchanged_sources_and_partial_diagnostic(self):
        report_data = {"image_count": 450, "execution_complete": True,
                       "metrics": {arm: metrics() for arm in collector.PATHS},
                       "per_event": {"event-one": {arm: metrics() for arm in collector.PATHS}}}
        val = self.root / "train/val_step_000500/result.json"
        write_json(val, report_data)
        diagnostic_row = {"mean_image_loss": 1.25, "mean_image_nll": .75,
                          "mean_image_lovasz": .5, "metrics": metrics()}
        split = {"image_count": 450, "metrics": {arm: diagnostic_row for arm in collector.PATHS}}
        write_json(self.diagnostic / "summary.json", {
            "complete": False, "steps": [500, 1500, 5000], "results": {"500": {"fixed_train": split, "val": split}},
            "error": "Interrupted"})
        (self.diagnostic / "traceback.txt").write_text("KeyboardInterrupt\n")
        external = Path(str(self.root) + ".source.json")
        write_json(external, {"original": "unchanged"})
        originals = [val, self.diagnostic / "summary.json", self.diagnostic / "traceback.txt", external]
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in originals}
        # Forbidden files must neither be read nor included in the archive.
        (self.root / "Test").mkdir()
        (self.root / "Test/secret.json").write_text("DO NOT READ")
        (self.root / "train/checkpoint.pt").write_bytes(b"DO NOT LOAD")
        report, archive, inventory = self.collect()
        self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in originals})
        self.assertEqual(inventory["diagnostic_status"]["state"], "reported_incomplete")
        self.assertIn("1500/val/d", inventory["diagnostic_status"]["missing_metric_groups"])
        text = report.read_text()
        self.assertIn("1.2500 (0.7500+0.5000)", text)
        self.assertIn("confusion GT行/预测列", text)
        self.assertIn("不能单独诊断过拟合", text)
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            self.assertFalse(any("Test" in name or ".pt" in name for name in names))
            match = next(name for name in names if "val_step_000500/result.json" in name)
            self.assertEqual(bundle.read(match), val.read_bytes())
        entry = inventory["files"]["val_500"]
        self.assertEqual(entry["snapshot_sha256"], hashlib.sha256(val.read_bytes()).hexdigest())
        second, second_zip, _ = self.collect()
        self.assertNotEqual(report, second)
        self.assertTrue(archive.exists() and second_zip.exists())

    def test_missing_summary_is_unknown_not_no_run(self):
        Path(str(self.diagnostic) + ".log").write_text("DIAGNOSTIC step=500 split=val 10/450\n")
        report, _, inventory = self.collect()
        self.assertEqual(inventory["files"]["diagnostic_summary"]["status"], "missing")
        self.assertEqual(inventory["diagnostic_status"]["state"], "unavailable")
        self.assertEqual(inventory["files"]["diagnostic_log_tail"]["status"], "copied")
        self.assertIn("不等于任务从未运行", report.read_text())
        loss = inventory["training_windows"]["1-500"]["heads"]["d"]["loss"]
        self.assertEqual(loss["n"], 0)
        self.assertIsNone(loss["mean"])

    def test_duplicate_truncated_rows_and_nonfinite_fields(self):
        path = self.root / "train/train.jsonl"
        def row(step, loss, **kw):
            return {"step": step, "heads": {arm: {"loss": loss} for arm in collector.ARMS}, **kw}
        records = [row(1, 2., lr=.001, event="event-a", image="image-1"),
                   row(2, None), row(1, 999., lr=99.), row(3, float("inf")),
                   row(501, 1e308, image="image-2"), row(502, 1e308, image="image-2")]
        path.write_text("\n".join(json.dumps(row) for row in records) + '\n{"step":4,')
        record, windows = collector.training_windows(path)
        self.assertEqual(record["counts"]["duplicate_steps"], 1)
        self.assertEqual(record["counts"]["truncated_json_lines"], 1)
        first = windows["1-500"]
        self.assertEqual(first["observed_rows"], 3)
        self.assertEqual(first["heads"]["a"]["loss"], {
            "n": 1, "mean": 2., "median": 2., "min": 2., "max": 2.,
            "missing_or_null": 1, "invalid_or_nonfinite": 1})
        self.assertEqual(first["rows_without_event"], 2)
        self.assertEqual(first["event_counts"], {"event-a": 1})
        self.assertEqual(windows["501-1000"]["unique_images"], 1)
        self.assertEqual(windows["501-1000"]["heads"]["d"]["loss"]["mean"], 1e308)
        self.assertFalse(record["observed_all_steps_1_to_5000"])
        json.dumps(windows, allow_nan=False)

    def test_parse_error_oversize_and_symlink_are_honest(self):
        (self.root / "workflow_state.json").write_text('{"bad":')
        oversized = self.root / "train/manifest.json"
        oversized.write_bytes(b" " * 65)
        secret = self.root / "Test/secret.json"
        write_json(secret, {"forbidden": True})
        (self.root / "source.json").symlink_to(secret)
        with patch.object(collector, "MAX_REPORT_BYTES", 64):
            _, _, inventory = self.collect()
        files = inventory["files"]
        self.assertEqual(files["workflow_state"]["status"], "parse_error")
        self.assertIsNotNone(files["workflow_state"]["snapshot"])
        self.assertEqual(files["train_manifest"]["status"], "oversize")
        self.assertIsNone(files["train_manifest"]["snapshot"])
        self.assertEqual(files["source"]["status"], "read_error")
        self.assertIsNone(files["source"]["snapshot"])

    def test_reject_output_under_either_source(self):
        for output in (self.root, self.root / "nested", self.diagnostic / "nested"):
            with self.assertRaisesRegex(ValueError, "outside both source roots"):
                collector.collect(self.root, self.diagnostic, output)
        self.assertFalse((self.root / "nested").exists())

    def test_root_directory_alias_is_supported(self):
        alias = self.base / "source_alias"
        alias.symlink_to(self.root, target_is_directory=True)
        write_json(self.root / "workflow_state.json", {"complete": True})
        report, _ = collector.collect(alias, self.diagnostic, self.output)
        inventory = json.loads((report.parent / "inventory.json").read_text())
        self.assertEqual(inventory["files"]["workflow_state"]["status"], "copied")
        self.assertEqual(inventory["root"], str(self.root))

    def test_tail_and_stream_limits(self):
        log = Path(str(self.root) + ".log")
        log.write_bytes(b"PREFIX-" + b"X" * 100)
        (self.root / "train/train.jsonl").write_bytes(b'{"step":1}\n' + b"x" * 100)
        with patch.object(collector, "MAX_TAIL_BYTES", 8), patch.object(collector, "MAX_JOURNAL_BYTES", 40):
            report, _, inventory = self.collect()
        tail = inventory["files"]["root_log_tail"]
        self.assertEqual((report.parent / tail["snapshot"]).read_bytes(), b"X" * 8)
        self.assertTrue(tail["is_tail"])
        journal = inventory["files"]["train_journal"]
        self.assertTrue(journal["byte_limit_reached"])
        self.assertEqual(journal["observed_bytes"], 40)
        self.assertEqual(journal["counts"]["truncated_json_lines"], 1)


if __name__ == "__main__":
    unittest.main()
