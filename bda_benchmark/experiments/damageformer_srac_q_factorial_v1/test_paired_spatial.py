"""Small synthetic checks for read-only paired Q spatial attribution."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .paired_spatial import (
    _analyse_image, damaged_boundary_distance, diagnose, false_positive_counts, paired_counts,
    q_mixing_exposure,
)


class PairedSpatialTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [0, 0, 3, 3],
            [0, 0, 3, 3],
        ], dtype=np.int16)
        self.q = np.ones((1, 4), dtype=np.float32)
        self.ids = np.array([[0, 0, 1, 2]], dtype=np.int32)
        self.b = np.array([[0.9, 0.1], [0.05, 0.8]], dtype=np.float32)

    def test_four_neighbor_distance_includes_image_exterior(self):
        all_damaged = np.full((5, 5), 2, dtype=np.int16)
        distance = damaged_boundary_distance(all_damaged)
        self.assertEqual(int(distance[0, 0]), 0)
        self.assertEqual(int(distance[1, 1]), 1)
        self.assertEqual(int(distance[2, 2]), 2)
        single = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
        self.assertEqual(int(damaged_boundary_distance(single)[1, 1]), 0)
        self.assertEqual(int(damaged_boundary_distance(single)[0, 0]), -1)

    def test_q_loss_is_attributed_to_low_b_and_foreground_class_mixing(self):
        result = _analyse_image(self.gt, self.q, self.ids, (2, 2), self.b)
        paired = result["paired"]
        self.assertEqual(paired["gt_damaged_pixels"], 4)
        self.assertEqual(paired["lost_by_q"], 4)
        self.assertEqual(paired["gained_by_q"], 0)
        self.assertEqual(paired["lost_to_intact"], 4)
        self.assertEqual(result["by_building_probability"]["below_0.5"]["lost_by_q"], 4)
        self.assertEqual(result["by_boundary_distance"]["0-0"]["lost_by_q"], 4)
        self.assertEqual(result["by_foreground_class_mixing_exposure"]
                         ["[0.5,1)"]["lost_by_q"], 4)
        self.assertEqual(result["by_background_mixing_exposure"]
                         ["[0,1e-08)"]["lost_by_q"], 4)

    def test_soft_exposure_respects_q_weight_not_hard_owner(self):
        q = np.array([[0.75], [0.25]])
        ids = np.array([[0], [1]])
        labels = np.array([[0, 0.5, 0.5, 0], [0.5, 0.5, 0, 0]])
        exposed = q_mixing_exposure(q, ids, (1, 1), labels,
                                    np.array([1., 1.]), (2, 2))
        np.testing.assert_allclose(exposed["foreground_class_mixing"], 0.75)
        np.testing.assert_allclose(exposed["background_mixing"], 0.25)

    def test_paired_counts_include_gain_and_keep_denominator(self):
        counts = paired_counts(np.array([1, 0, 1, 0], dtype=bool),
                               np.array([0, 1, 1, 0], dtype=bool),
                               np.array([1, 2, 2, 0]),
                               np.array([1, 1, 1, 1], dtype=bool))
        self.assertEqual(counts["gt_damaged_pixels"], 4)
        self.assertEqual(counts["lost_by_q"], 1)
        self.assertEqual(counts["gained_by_q"], 1)
        self.assertEqual(counts["correct_in_both"], 1)
        self.assertEqual(counts["wrong_in_both"], 1)

    def test_false_positive_counts_separate_new_from_removed(self):
        gt = np.array([0, 1, 3, 1, 2, 255])
        grid = np.array([0, 2, 2, 1, 2, 2])
        q = np.array([2, 1, 2, 2, 0, 2])
        counts = false_positive_counts(gt, grid, q)
        self.assertEqual(counts["gt_non_damaged_pixels"], 4)
        self.assertEqual(counts["grid_false_positive"], 2)
        self.assertEqual(counts["q_false_positive"], 3)
        self.assertEqual(counts["introduced"], 2)
        self.assertEqual(counts["removed"], 1)
        self.assertEqual(counts["introduced_from_background"], 1)
        self.assertEqual(counts["introduced_from_intact"], 1)

    def _saved_fixture(self, root: Path):
        (root / "predictions").mkdir()
        artifact = root / "predictions" / "0000.npz"
        np.savez_compressed(
            artifact, target=self.gt[None], q=self.q[None],
            region_ids=self.ids, grid_shape=np.array([2, 2]),
            building_grid=self.b[None, None],
        )
        sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        journal = dict(index=0, image="example-event_0001", event="example-event",
                       prediction_file="predictions/0000.npz", prediction_sha256=sha)
        (root / "images.jsonl").write_text(json.dumps(journal) + "\n", encoding="utf-8")
        (root / "summary.json").write_text(json.dumps(dict(
            format="SRAC_Q_MEMORY_SAMEPASS_V1", complete=True, image_count=1,
            full_validation_evidence=False, test_accessed=False,
            audit_file="audit.json")), encoding="utf-8")
        (root / "audit.json").write_text(json.dumps(dict(
            format="SRAC_Q_MEMORY_SAMEPASS_V1", complete=True, image_count=1,
            full_validation_evidence=False, test_accessed=False,
            reproduction={"passed": None})), encoding="utf-8")
        return artifact

    def test_partial_requires_smoke_and_output_is_non_formal(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / "source", Path(temp) / "diagnostic"
            root.mkdir()
            self._saved_fixture(root)
            with self.assertRaisesRegex(ValueError, "full Val450"):
                diagnose(root, out)
            self.assertFalse(out.exists())
            report = diagnose(root, out, smoke=True)
            self.assertTrue(report["complete"])
            self.assertFalse(report["full_validation_evidence"])
            self.assertEqual(report["arms"]["old"]["paired"]["lost_by_q"], 4)
            self.assertEqual(report["arms"]["old"]["false_positive"]["net_change"], 0)
            self.assertEqual(len((out / "images.jsonl").read_text().splitlines()), 1)
            with self.assertRaisesRegex(FileExistsError, "fresh output"):
                diagnose(root, out, smoke=True)

    def test_missing_or_tampered_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            root.mkdir()
            artifact = self._saved_fixture(root)
            with artifact.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                diagnose(root, Path(temp) / "result", smoke=True)


if __name__ == "__main__":
    unittest.main()
