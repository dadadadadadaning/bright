"""Synthetic integration tests; no server data, GPU or real results are used."""
from __future__ import annotations

from argparse import Namespace
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_audit_predictions import REPO_ROOT, ARMS, digest, write_fixture
from verify_sar import main, validate


def rebind(root, row):
    (root / "result.json").write_text(json.dumps(row), encoding="utf-8")
    summary = dict(row, result_file="result.json", result_sha256=digest(root / "result.json"))
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def arms(folder):
    result = {}
    for name, arm in (("baseline", "baseline"), ("sar", "sar_cue"),
                      ("rgb", "rgb_capacity"), ("shifted", "shifted_sar")):
        root = Path(folder) / name
        write_fixture(root)
        row = json.loads((root / "result.json").read_text())
        row.update(arm=arm, checkpoint_sha256="synthetic-" + arm,
                   q_a1_reproduction={"passed": True})
        rebind(root, row)
        result[name] = str(root)
    return Namespace(repo_root=str(REPO_ROOT), reports_only=False, output=None, **result)


class SARVerificationTests(unittest.TestCase):
    def test_all_four_real_archive_gates_can_complete_without_positive_effect(self):
        with tempfile.TemporaryDirectory() as folder:
            report = validate(arms(folder))
            self.assertTrue(report["complete"])
            self.assertEqual(report["direct_differences_pp"]["q_region"]
                             ["sar_minus_shifted_sar"]["foreground_miou_pp"], 0)

    def test_missing_control_is_pending_not_a_completed_mechanism_check(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arms(folder)
            args.rgb = None
            report = validate(args)
            self.assertFalse(report["complete"])
            self.assertEqual(report["status"]["rgb"]["state"], "not_supplied")

    def test_mismatched_source_is_rejected_even_with_consistent_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arms(folder)
            root = Path(args.shifted)
            row = json.loads((root / "result.json").read_text())
            row["source_sha256"] = "synthetic-other-source"
            rebind(root, row)
            report = validate(args)
            self.assertFalse(report["complete"])
            self.assertIn("shifted", report["match_errors"])

    def test_stored_score_cannot_disagree_with_unchanged_confusion(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arms(folder)
            root = Path(args.sar)
            row = json.loads((root / "result.json").read_text())
            row["metrics"]["q_region"]["valid_miou"] = 0.5
            rebind(root, row)
            report = validate(args)
            self.assertFalse(report["complete"])
            self.assertIn("disagrees with confusion", report["status"]["sar"]["error"])

    def test_changed_targets_are_rejected_despite_identical_confusion(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arms(folder)
            root = Path(args.shifted)
            archive = root / "predictions/0000.npz"
            with np.load(archive, allow_pickle=False) as source:
                arrays = {key: source[key].copy()[..., ::-1] for key in source.files}
            np.savez_compressed(archive, **arrays)
            journal = root / "images.jsonl"
            rows = [json.loads(line) for line in journal.read_text().splitlines()]
            rows[0]["prediction_sha256"] = digest(archive)
            journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            report = validate(args)
            self.assertFalse(report["complete"])
            self.assertIn("validation_targets", report["match_errors"])

    def test_reports_only_exits_pending_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arms(folder)
            output = Path(folder) / "external-report.json"
            argv = ["verify_sar.py", "--repo-root", args.repo_root, "--baseline", args.baseline,
                    "--sar", args.sar, "--rgb", args.rgb, "--shifted", args.shifted,
                    "--reports-only", "--output", str(output)]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 2)
            before = output.read_bytes()
            with patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
