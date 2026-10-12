"""Synthetic CPU tests for saved-label analysis; no BRIGHT data are opened."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bda_benchmark.experiments.damageformer_srac_region_val_formal_v1.core import ARMS, PAIRS, source_hash
from bda_benchmark.experiments.damageformer_srac_region_val_formal_v1.report import IMMUTABLE_KEYS
from .analyze import (analyze, artifact_path, breakdown, class_boundary, digest,
                      joint_counts, metrics, transition_summary)


def save(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def fixture(root):
    """450 tiny synthetic maps exercise the real formal count gate."""
    root.mkdir()
    (root / "predictions").mkdir()
    names = [f"toy-event_{i:08d}" for i in range(450)]
    protocol_path = root.parent / "protocol.json"
    protocol = dict(format="SRAC_HARDQUERY_PROTOCOL_V1", complete=True,
                    full_validation_evidence=True, image_count=450, scope="all",
                    profile="legacy", pool_mode="native", smoke=False,
                    reproduction={"passed": True}, selected_images=names,
                    input_files_unchanged=True, code_unchanged=True,
                    frozen_evidence_unchanged=True, numerical_unchanged=True)
    save(protocol_path, protocol)
    y = np.array([[[0, 1, 2, 3], [2, 2, 1, 255]]], dtype=np.int16)
    old = np.array([[[0, 1, 2, 3], [1, 2, 1, 0]]], dtype=np.uint8)
    new = np.array([[[2, 2, 1, 2], [2, 2, 1, 0]]], dtype=np.uint8)
    predictions = {arm: new if arm == "region" else old for arm in ARMS}
    cms = {}
    for arm, pred in predictions.items():
        cms[arm] = joint_counts(y, pred, pred).sum(2)
    rows = []
    for index, name in enumerate(names):
        path = root / "predictions" / f"{index:04d}.npz"
        np.savez_compressed(path, target=y, **{"prediction_" + arm: value for arm, value in predictions.items()})
        rows.append(dict(index=index, image=name, event="toy-event",
                         confusion={arm: cm.tolist() for arm, cm in cms.items()},
                         prediction_file=path.relative_to(root).as_posix(),
                         prediction_sha256=digest(path)))
    write_journal(root, rows)
    base = dict(format="SRAC_REGION_VAL_FORMAL_V1", complete=True,
                full_validation_evidence=True, image_count=450,
                test_accessed=False, immutable={key: True for key in IMMUTABLE_KEYS},
                native_a1_replay_ok=True, source_sha256=source_hash())
    summary = dict(base, reproduction_passed=True, result_file="result.json")
    metric_rows = {arm: metrics(cm*450) for arm, cm in cms.items()}
    transitions = {}
    for pair in PAIRS:
        a, b = pair.split("__")
        transitions[pair] = transition_summary(joint_counts(y, predictions[a], predictions[b])*450)
    result = dict(base, expected_image_count=450, training=False,
                  reproduction={"passed": True, "image_count": 450, "tolerance": 1e-5,
                                "checks": [{"passed": True}]},
                  metrics=metric_rows, transitions=transitions,
                  pool_guard={"backend": "native", "restored": True, "failures": []},
                  per_event={"toy-event": metric_rows}, profile="legacy", pool_mode="native",
                  protocol_path=str(protocol_path), protocol_sha256=digest(protocol_path))
    save(root / "summary.json", summary)
    save(root / "result.json", result)
    return rows, y, old, new


def write_journal(root, rows):
    (root / "images.jsonl").write_text("\n".join(json.dumps(row) for row in rows)+"\n", encoding="utf-8")


class CountsTests(unittest.TestCase):
    def test_harm_rescue_and_false_positive_have_fixed_eligible_denominators(self):
        y = np.array([[2, 2, 2, 2, 1, 1, 1, 0, 3, 255]])
        old = np.array([[2, 2, 1, 0, 1, 2, 0, 0, 2, 2]])
        new = np.array([[1, 2, 2, 2, 2, 1, 1, 2, 3, 0]])
        row = breakdown(joint_counts(y, old, new))
        d = row["per_class"]["Damaged"]
        self.assertEqual((d["gt_count"], d["harm"], d["rescue"], d["net_tp"]), (4, 1, 2, 1))
        self.assertEqual(d["harm_rate"], 0.5)
        self.assertEqual(d["rescue_rate"], 1.0)
        self.assertEqual(row["damaged_harm_destination"]["Intact"]["count"], 1)
        f = row["damaged_false_positive_by_true_class"]["Intact"]
        self.assertEqual((f["added"], f["removed"], f["net"]), (1, 1, 0))
        self.assertEqual((f["addition_eligible_reference_not_damaged"], f["removal_eligible_reference_damaged"]), (2, 1))
        self.assertEqual(f["addition_rate"], 0.5)
        self.assertEqual(f["removal_rate"], 1.0)
        self.assertEqual(row["valid_pixels"], 9)
        self.assertIsNone(breakdown(np.zeros((4, 4, 4), dtype=np.int64))["reference_error_rate"])

    def test_boundary_ignores_void_image_border_and_diagonal_only_changes(self):
        y = np.array([[1, 1, 255], [1, 1, 255]])
        self.assertFalse(class_boundary(y).any())
        y = np.array([[1, 255], [255, 2]])
        self.assertFalse(class_boundary(y).any())
        y = np.array([[1, 1, 2, 2, 255]])
        np.testing.assert_array_equal(class_boundary(y), [[False, True, True, False, False]])
        y = np.array([[0, 0], [2, 2]])
        self.assertTrue(class_boundary(y).all())

    def test_boundary_and_interior_partition_valid_counts(self):
        y = np.array([[0, 0, 255, 1], [0, 2, 255, 1], [3, 2, 2, 1]])
        old = np.where(y < 4, y, 0)
        new = np.zeros_like(old)
        edge = class_boundary(y)
        whole = joint_counts(y, old, new)
        split = joint_counts(y, old, new, edge) + joint_counts(y, old, new, (y < 4) & ~edge)
        np.testing.assert_array_equal(whole, split)
        with self.assertRaises(ValueError):
            joint_counts(y, old.astype(float), new)


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / "input"
        self.rows, self.y, self.old, self.new = fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def run_analysis(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return analyze(self.root, self.base / "analysis", **kwargs)

    def test_all_maps_counts_order_protocol_and_source_are_verified(self):
        original_hash = digest(self.root / "result.json")
        report = self.run_analysis()
        self.assertTrue(report["complete"])
        self.assertTrue(report["protocol"]["selected_images_order_verified"])
        self.assertEqual(report["verified_prediction_files"], 450)
        self.assertEqual(report["overall"]["per_class"]["Damaged"]["harm"], 450)
        self.assertEqual(report["ignored_pixels"], 450)
        self.assertEqual(digest(self.root / "result.json"), original_hash)
        self.assertTrue((self.base / "analysis" / "report.md").is_file())

    def test_corrupt_npz_checksum_is_rejected_before_report(self):
        path = self.root / self.rows[0]["prediction_file"]
        path.write_bytes(path.read_bytes()+b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_analysis()
        self.assertFalse((self.base / "analysis").exists())

    def test_rehashed_map_must_still_match_journal_confusion(self):
        path = self.root / self.rows[0]["prediction_file"]
        with np.load(path, allow_pickle=False) as archive:
            maps = {key: archive[key] for key in archive.files}
        maps["prediction_region"][0, 0, 0] = 1
        np.savez_compressed(path, **maps)
        self.rows[0]["prediction_sha256"] = digest(path)
        write_journal(self.root, self.rows)
        with self.assertRaisesRegex(ValueError, "Count differs"):
            self.run_analysis()

    def test_duplicate_or_out_of_order_journal_rejected(self):
        self.rows[0]["image"] = self.rows[1]["image"]
        write_journal(self.root, self.rows)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            self.run_analysis()
        self.rows[0]["image"] = "toy-event_00000000"
        self.rows[0], self.rows[1] = self.rows[1], self.rows[0]
        write_journal(self.root, self.rows)
        with self.assertRaisesRegex(ValueError, "out of order"):
            self.run_analysis()

    def test_protocol_hash_and_order_cannot_silently_change(self):
        p = self.base / "protocol.json"
        data = json.loads(p.read_text())
        data["selected_images"] = list(reversed(data["selected_images"]))
        save(p, data)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.run_analysis()
        result = json.loads((self.root / "result.json").read_text())
        result["protocol_sha256"] = digest(p)
        save(self.root / "result.json", result)
        with self.assertRaisesRegex(ValueError, "image order"):
            self.run_analysis()

    def test_global_transition_mismatch_and_nonformal_result_rejected(self):
        p = self.root / "result.json"
        original = json.loads(p.read_text())
        bad = json.loads(p.read_text())
        bad["transitions"]["region_mean__region"]["joint_gt_reference_candidate"][2][2][1] += 1
        save(p, bad)
        with self.assertRaisesRegex(ValueError, "Count differs"):
            self.run_analysis()
        original["full_validation_evidence"] = False
        save(p, original)
        with self.assertRaisesRegex(ValueError, "full Val450"):
            self.run_analysis()

    def test_paths_are_confined_and_output_cannot_overwrite_input(self):
        for unsafe in ("../predictions/0000.npz", "/predictions/0000.npz", "predictions/../0000.npz", "predictions\\0000.npz"):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                artifact_path(self.root, unsafe)
        with self.assertRaisesRegex(ValueError, "read-only"):
            analyze(self.root, self.root / "analysis")

    def test_missing_protocol_is_explicit_not_a_claimed_order_verification(self):
        (self.base / "protocol.json").unlink()
        report = self.run_analysis(boundary=False)
        self.assertFalse(report["protocol"]["available"])
        self.assertFalse(report["protocol"]["selected_images_order_verified"])
        self.assertEqual(report["boundary_strata"], {})


if __name__ == "__main__":
    unittest.main()
