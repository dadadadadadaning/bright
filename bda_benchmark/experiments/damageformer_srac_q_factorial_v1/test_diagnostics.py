"""CPU synthetic checks for GT-only Q diagnostics and formal evidence gates."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .audit_saved_q import audit, verified_images
from .diagnostics import (
    area_class_fractions, building_probability_stats, component_building_stats,
    component_proxy_stats, confusion, prediction_confusion_by_building_probability,
    probe_prediction, region_mixing_mass,
)


class QDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [0, 0, 3, 3],
            [0, 0, 3, 3],
        ], dtype=np.int16)

    def test_identity_q_roundtrip_preserves_area_grid(self):
        q = np.ones((1, 4), dtype=np.float32)
        ids = np.arange(4, dtype=np.int64)[None]
        grid, rebuilt, labels, mass = probe_prediction(self.gt, q, ids, (2, 2))
        np.testing.assert_array_equal(grid, self.gt)
        np.testing.assert_array_equal(rebuilt, self.gt)
        np.testing.assert_allclose(labels, np.eye(4)[[1, 2, 0, 3]])
        np.testing.assert_allclose(mass, np.ones(4))

    def test_q_merge_can_lose_damaged_after_grid_keeps_it(self):
        q = np.ones((1, 4), dtype=np.float32)
        ids = np.array([[0, 0, 1, 2]], dtype=np.int64)
        grid, rebuilt, _, _ = probe_prediction(self.gt, q, ids, (2, 2))
        self.assertEqual(int((grid == self.gt).sum()), 16)
        self.assertEqual(int((rebuilt == self.gt).sum()), 12)
        self.assertTrue((rebuilt[:2, 2:] == 1).all())

    def test_building_probability_uses_soft_grid_truth(self):
        fractions, _ = area_class_fractions(self.gt, (2, 2))
        b = np.array([[0.9, 0.1], [0.05, 0.8]])
        report = building_probability_stats(b, fractions)
        self.assertAlmostEqual(report["Damaged"]["mean_predicted_building_probability"], 0.1)
        self.assertEqual(report["Damaged"]["low_probability_mass_fraction"]["0.3"], 1.0)
        self.assertEqual(report["Background"]["low_probability_mass_fraction"]["0.3"], 1.0)

    def test_component_proxy_is_raster_not_building(self):
        component = component_proxy_stats(self.gt, self.gt, self.gt)
        self.assertEqual(component["Intact"][0]["connected_raster_components"], 1)
        self.assertEqual(component["Damaged"][0]["gt_pixels"], 4)

    def test_small_damaged_component_is_detected_as_low_building_probability(self):
        b = np.array([[0.9, 0.1], [0.05, 0.8]])
        report = component_building_stats(self.gt, b)
        damaged = report["Damaged"][0]
        self.assertEqual(damaged["connected_raster_components"], 1)
        self.assertEqual(damaged["gt_pixels_at_or_above_threshold"], 0)
        self.assertEqual(damaged["components_majority_below_threshold"], 1)

    def test_prediction_errors_keep_fixed_gt_denominator_across_b_bins(self):
        b = np.array([[0.9, 0.1], [0.05, 0.8]])
        pred = self.gt.copy()
        pred[:2, 2:] = 1  # low-B Damaged misses become Intact
        pred[:2, :2] = 2  # high-B Intact becomes Damaged
        bins = prediction_confusion_by_building_probability(self.gt, pred, b)
        np.testing.assert_array_equal(bins.sum(axis=0), confusion(self.gt, pred))
        self.assertEqual(int(bins[1, 2, 1]), 4)
        self.assertEqual(int(bins[4, 1, 2]), 4)

    def test_background_and_foreground_class_mixing_are_separate(self):
        labels = np.array([[0.5, 0.5, 0, 0],
                           [0, 0.5, 0.5, 0],
                           [0, 1, 0, 0]], dtype=np.float64)
        report = region_mixing_mass(labels, np.array([2., 3., 4.]))
        self.assertEqual(report["background_foreground_mixed_mass"], 2.)
        self.assertEqual(report["foreground_class_mixed_mass"], 3.)
        self.assertEqual(report["labelled_mass"], 9.)

    def test_noninteger_area_grid_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer downsample"):
            area_class_fractions(self.gt, (3, 3))


class SavedEvidenceTests(unittest.TestCase):
    def _fixture(self, root):
        (root / "predictions").mkdir()
        file = root / "predictions" / "0000.npz"
        gt = np.array([[[1, 1], [0, 0]]], dtype=np.int16)
        np.savez_compressed(file, target=gt,
                            q=np.ones((1, 1, 1), dtype=np.float32),
                            region_ids=np.zeros((1, 1), dtype=np.int32),
                            grid_shape=np.array([1, 1], dtype=np.int32))
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        (root / "images.jsonl").write_text(json.dumps(dict(
            index=0, image="dummy-event_0001", event="dummy-event",
            prediction_file="predictions/0000.npz", prediction_sha256=digest)) + "\n")
        (root / "summary.json").write_text(json.dumps(dict(
            format="SRAC_Q_MEMORY_SAMEPASS_V1",
            complete=True, image_count=1, full_validation_evidence=False,
            test_accessed=False, audit_file="audit.json")))
        (root / "audit.json").write_text(json.dumps(dict(
            format="SRAC_Q_MEMORY_SAMEPASS_V1",
            complete=True, image_count=1, full_validation_evidence=False,
            test_accessed=False, reproduction={"passed": None})))
        return file

    def test_partial_result_requires_explicit_smoke_and_marks_b_unavailable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            root.mkdir()
            self._fixture(root)
            with self.assertRaisesRegex(ValueError, "full Val450"):
                verified_images(root)
            report = audit(root, Path(temp) / "diagnostic", smoke=True)
            self.assertFalse(report["full_validation_evidence"])
            self.assertEqual(report["arms"]["old"]["building_probability"]["status"],
                             "unavailable_in_saved_artifact")

    def test_prediction_sha_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            root.mkdir()
            file = self._fixture(root)
            with file.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verified_images(root, smoke=True)

    def test_new_evaluation_result_sha_and_variant_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            root.mkdir()
            file = self._fixture(root)
            with np.load(file, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays.update(q_new=arrays["q"], region_ids_new=arrays["region_ids"],
                          grid_shape_new=arrays["grid_shape"],
                          building_grid=np.array([[[[0.2]]]], dtype=np.float32),
                          building_grid_new=np.array([[[[0.4]]]], dtype=np.float32))
            np.savez_compressed(file, **arrays)
            journal = json.loads((root / "images.jsonl").read_text())
            journal["prediction_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
            (root / "images.jsonl").write_text(json.dumps(journal) + "\n")
            source = json.loads((root / "audit.json").read_text())
            source["format"] = "SRAC_Q_FACTORIAL_EVAL_V1"
            (root / "result.json").write_text(json.dumps(source))
            report_sha = hashlib.sha256((root / "result.json").read_bytes()).hexdigest()
            summary = json.loads((root / "summary.json").read_text())
            summary.update(format="SRAC_Q_FACTORIAL_EVAL_V1",
                           result_file="result.json", result_sha256=report_sha)
            summary.pop("audit_file")
            (root / "summary.json").write_text(json.dumps(summary))
            result = audit(root, Path(temp) / "diagnostic", smoke=True)
            self.assertEqual(sorted(result["arms"]), ["new", "old"])
            self.assertEqual(result["arms"]["old"]["building_probability"]["Intact"]
                             ["low_probability_mass_fraction"]["0.3"], 1.0)
            summary["result_sha256"] = "0" * 64
            (root / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "report hash mismatch"):
                verified_images(root, smoke=True)


if __name__ == "__main__":
    unittest.main()
