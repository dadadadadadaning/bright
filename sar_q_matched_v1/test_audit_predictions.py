"""Synthetic archive tests; these fixtures are not BRIGHT experiment evidence."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_predictions import audit_evaluation, confusion_from_maps, delta_cells, main

ARMS = ("a1_pixel", "q_pixel_grid", "q_region_mean", "q_grid_probq", "q_region")
REPO_ROOT = Path(os.environ.get("BRIGHT_REPO_ROOT", "/workspace/bright")).resolve()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture(root: Path, *, wrong_image=False, wrong_global=False, wrong_event=False) -> None:
    """450 tiny synthetic maps exercise the unchanged original artifact gates."""
    (root / "predictions").mkdir(parents=True)
    gt = np.array([[[0, 1, 2], [3, 255, 0]]], dtype=np.int16)
    pred = np.array([[[0, 1, 2], [3, 0, 0]]], dtype=np.uint8)
    cm = np.diag([2, 1, 1, 1]).astype(np.int64)
    journal = []
    for index in range(450):
        artifact = root / "predictions" / f"{index:04d}.npz"
        np.savez_compressed(artifact, target=gt, **{"prediction_" + arm: pred for arm in ARMS})
        image_cms = {arm: cm.tolist() for arm in ARMS}
        if wrong_image and index == 7:
            image_cms["q_grid_probq"][1][1] -= 1
            image_cms["q_grid_probq"][1][2] += 1
        journal.append(dict(index=index, image=f"synthetic-myanmar_{index:08d}", event="synthetic-myanmar",
                            confusion=image_cms, prediction_file=f"predictions/{index:04d}.npz",
                            prediction_sha256=digest(artifact)))
    (root / "images.jsonl").write_text("\n".join(json.dumps(row) for row in journal) + "\n", encoding="utf-8")
    metrics = {arm: dict(confusion=(cm * 450).tolist(), iou=[1.0] * 4, valid_miou=1.0) for arm in ARMS}
    per_event = json.loads(json.dumps({"synthetic-myanmar": metrics}))
    if wrong_global:
        metrics["q_region"]["confusion"][0][0] += 1
    if wrong_event:
        per_event["synthetic-myanmar"]["q_region_mean"]["confusion"][3][3] -= 1
    report = dict(format="SRAC_Q_FACTORIAL_EVAL_V1", complete=True, full_validation_evidence=True,
                  image_count=450, reproduction=dict(passed=True), reproduction_passed=True,
                  native_a1_replay_ok=True, test_accessed=False, immutable=dict(synthetic=True),
                  protocol_sha256="synthetic-protocol", source_sha256="synthetic-code",
                  checkpoint_sha256="synthetic-checkpoint", arm="baseline", checkpoint_steps=2000,
                  numerical_profile="legacy", pool_mode="native", training_hyperparameters=dict(steps=2000),
                  metrics=metrics, per_event=per_event)
    (root / "result.json").write_text(json.dumps(report), encoding="utf-8")
    summary = dict(report, result_file="result.json", result_sha256=digest(root / "result.json"))
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


class ArchivedConfusionTests(unittest.TestCase):
    def test_target_255_is_excluded_and_other_ids_are_not_relabelled(self):
        gt = np.array([[[0, 1], [255, 3]]], dtype=np.int16)
        pred = np.array([[[0, 2], [1, 3]]], dtype=np.uint8)
        expected = np.zeros((4, 4), dtype=np.int64)
        expected[0, 0] = expected[1, 2] = expected[3, 3] = 1
        np.testing.assert_array_equal(confusion_from_maps(gt, pred), expected)
        with self.assertRaises(ValueError):
            confusion_from_maps(gt, np.array([[[0, 2], [255, 3]]], dtype=np.int16))

    def test_cells_report_signed_exact_deltas(self):
        actual = np.zeros((4, 4), dtype=np.int64)
        stored = actual.copy()
        actual[1, 2], stored[1, 1] = 3, 3
        cells = delta_cells(actual, stored)
        self.assertEqual([(v["gt_id"], v["prediction_id"], v["delta"]) for v in cells],
                         [(1, 1, -3), (1, 2, 3)])

    def test_matching_archives_are_read_only_and_single_image_is_limited(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "evaluation"
            write_fixture(root)
            before = {p: digest(p) for p in root.rglob("*") if p.is_file()}
            full = audit_evaluation(REPO_ROOT, root)
            self.assertTrue(full["passed"])
            self.assertTrue(full["complete_audit"])
            self.assertTrue(full["full_validation_evidence"])
            self.assertEqual(full["image_count"], 450)
            self.assertEqual(full["matrix_checks"], 450 * len(ARMS) + 4 * len(ARMS))
            self.assertEqual(len(full["ordered_targets_sha256"]), 64)
            single = audit_evaluation(REPO_ROOT, root, "synthetic-myanmar_00000007")
            self.assertTrue(single["passed"])
            self.assertFalse(single["complete_audit"])
            self.assertFalse(single["full_validation_evidence"])
            self.assertEqual(single["image_count"], 1)
            self.assertEqual(single["matrix_checks"], len(ARMS))
            self.assertIsNone(single["ordered_targets_sha256"])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--repo-root", str(REPO_ROOT), "--evaluation", str(root),
                                       "--image", "synthetic-myanmar_00000007"]), 0)
            self.assertEqual(before, {p: digest(p) for p in root.rglob("*") if p.is_file()})

    def test_journal_mismatch_exits_nonzero_and_does_not_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "evaluation"
            write_fixture(root, wrong_image=True)
            output = Path(folder) / "audit.json"
            before = (root / "images.jsonl").read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(["--repo-root", str(REPO_ROOT), "--evaluation", str(root),
                               "--image", "synthetic-myanmar_00000007", "--output", str(output)])
            self.assertEqual(status, 1)
            report = json.loads(output.read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["mismatches"][0]["scope"], "image")
            self.assertEqual([cell["delta"] for cell in report["mismatches"][0]["cells"]], [1, -1])
            self.assertEqual(before, (root / "images.jsonl").read_bytes())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--repo-root", str(REPO_ROOT), "--evaluation", str(root),
                                       "--output", str(output)]), 2)

    def test_global_and_per_event_mismatches_are_detected_for_every_arm(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "evaluation"
            write_fixture(root, wrong_global=True, wrong_event=True)
            report = audit_evaluation(REPO_ROOT, root)
            self.assertFalse(report["passed"])
            self.assertTrue(report["complete_audit"])
            self.assertFalse(report["full_validation_evidence"])
            observed = {(r["scope"], r["record"], r["arm"]) for r in report["mismatches"]}
            self.assertEqual(observed, {("global", "summary", "q_region"), ("global", "result", "q_region"),
                                       ("event", "summary", "q_region_mean"), ("event", "result", "q_region_mean")})

    def test_original_hash_gate_rejects_tampered_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "evaluation"
            write_fixture(root)
            archive = root / "predictions/0000.npz"
            archive.write_bytes(archive.read_bytes() + b"synthetic tampering")
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(["--repo-root", str(REPO_ROOT), "--evaluation", str(root)])
            self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
