"""Small CPU contracts; no BRIGHT image, checkpoint, GPU, or Test labels."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .stats import (
    ExposureAccumulator,
    StratifiedAccumulator,
    classify_regions,
    focus,
    route_maps,
    soft_route_exposure,
)
from .run import _artifact, _checked_protocol, _checked_rows, _support_mismatch


class StratificationTests(unittest.TestCase):
    def test_topk_mismatch_report_separates_rank_and_set_changes(self):
        e0 = np.array([[1, 0, 2], [0, 1, 2], [0, 1, 2]])
        replay = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 2]])
        scores = np.array(
            [[0.9, 0.8, 0.7, 0.6], [0.9, 0.8, 0.7, 0.6], [0.9, 0.8, 0.7, 0.6]]
        )
        result = _support_mismatch(e0, replay, scores, np.ones(4, bool), 3)
        self.assertEqual(result["changed_slots"], 4)
        self.assertEqual(result["queries_with_rank_change"], 2)
        self.assertEqual(result["queries_with_set_change"], 1)
        self.assertEqual(len(result["examples"]), 2)
        self.assertEqual(result["examples"][0]["e0_ids_scores_under_replay_query"], [0.8, 0.9, 0.7])
        self.assertAlmostEqual(result["examples"][0]["replay_boundary_margin"], 0.1)

    def test_background_dilution_is_not_foreground_class_mixing(self):
        distribution = np.array(
            [
                [0.8, 0, 0.2, 0],  # Mostly BG but foreground is purely D.
                [0, 0.5, 0.5, 0],  # Pure FG occupancy but mixed I/D.
                [1, 0, 0, 0],  # No foreground class can be assigned.
                [0.19, 0, 0.729, 0.081],  # FG occupancy .81; D is .9 of FG.
            ]
        )
        row = classify_regions(distribution, np.ones(4))
        np.testing.assert_array_equal(row["foreground_bin"], [1, 2, 0, 2])
        np.testing.assert_array_equal(row["class_purity_bin"], [3, 1, 0, 2])
        self.assertAlmostEqual(row["class_purity"][0], 1.0)
        self.assertAlmostEqual(row["class_purity"][1], 0.5)

    def test_hard_owner_and_soft_exposure_are_different(self):
        q = np.array([[0.9, 0.4, 1.0, 0.5], [0.1, 0.6, 0.0, 0.5]])
        ids = np.array([[0, 0, 1, 0], [1, 1, 0, 1]], dtype=np.int64)
        maps = route_maps(q, ids, {"foreground_bin": np.array([2, 1])}, (2, 2), (4, 4))
        np.testing.assert_array_equal(
            maps["foreground_bin"],
            np.array(
                [
                    [2, 2, 1, 1],
                    [2, 2, 1, 1],
                    [1, 1, 2, 2],
                    [1, 1, 2, 2],
                ]
            ),
        )
        exposure = soft_route_exposure(q, ids, np.array([False, True]), (2, 2))
        np.testing.assert_allclose(exposure, [[0.1, 0.6], [1.0, 0.5]])

    def test_fixed_j1_rescue_harm_and_intact_false_positive_denominators(self):
        target = np.array([[1, 2, 2, 1, 1]], dtype=np.int64)
        j1 = np.array([[1, 1, 2, 2, 1]], dtype=np.int64)
        j2 = np.array([[2, 2, 1, 1, 1]], dtype=np.int64)
        j3 = np.array([[1, 1, 2, 2, 2]], dtype=np.int64)
        maps = {
            "foreground_bin": np.array([[2, 2, 1, 0, 2]]),
            "class_purity_bin": np.array([[1, 3, 2, 0, 3]]),
        }
        acc = StratifiedAccumulator()
        acc.add("event_a", target, j1, {"j2": j2, "j3": j3}, maps)
        acc.add("event_b", target, j1, {"j2": j2, "j3": j3}, maps)
        total = acc.report()
        row = total["overall"]["all"]["all"]["j2"]
        self.assertEqual(row["damaged_j1_wrong"], 2)
        self.assertEqual(row["damaged_rescue"], 2)
        self.assertEqual(row["damaged_harm"], 2)
        self.assertEqual(row["intact_j1_correct"], 4)
        self.assertEqual(row["intact_correct_to_damaged"], 2)
        self.assertEqual(row["damaged_false_positive_delta"], 0)
        self.assertEqual(
            total["by_event"]["event_a"]["all"]["all"]["j2"]["damaged_rescue"], 1
        )
        self.assertEqual(
            total["overall"]["foreground_fraction"]["fg_ge080"]["j2"]["damaged_rescue"],
            2,
        )
        self.assertEqual(
            total["overall"]["foreground_class_purity"]["class_lt080"]["j2"][
                "intact_correct_to_damaged"
            ],
            2,
        )
        self.assertEqual(
            focus(np.zeros((4, 4, 4), dtype=np.int64))["damaged_rescue_rate"], None
        )

    def test_soft_exposure_keeps_same_fixed_denominator(self):
        target = np.array([[1, 2, 2]])
        old = np.array([[1, 1, 2]])
        cand = np.array([[2, 2, 1]])
        exposure = {
            "background_foreground_mixed": np.array([[0.8, 0.2, 0.0]]),
            "foreground_class_mixed": np.array([[0.1, 0.9, 0.5]]),
        }
        acc = ExposureAccumulator(arms=("j2",))
        acc.add("event_a", target, old, {"j2": cand}, exposure)
        row = acc.report()["overall"]["j2"]
        self.assertEqual(
            row["intact_j1_correct"]["background_foreground_mixed"]["count"], 1
        )
        self.assertAlmostEqual(
            row["intact_correct_to_damaged"]["background_foreground_mixed"]["mean"], 0.8
        )
        self.assertAlmostEqual(
            row["damaged_rescued"]["foreground_class_mixed"]["mean"], 0.9
        )
        self.assertAlmostEqual(
            row["damaged_harmed"]["foreground_class_mixed"]["mean"], 0.5
        )

    def test_e0_gate_rejects_strict_and_failed_runs(self):
        names = [f"event_{i:03d}.tif" for i in range(450)]
        base = dict(
            format="SRAC_HARDQUERY_PROTOCOL_V1",
            complete=True,
            full_validation_evidence=True,
            image_count=450,
            reproduction={"passed": True},
            scope="all",
            profile="legacy",
            pool_mode="native",
            selected_images=names,
            input_files_unchanged=True,
            code_unchanged=True,
            frozen_evidence_unchanged=True,
            numerical_unchanged=True,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "protocol.json"
            path.write_text(json.dumps(base), encoding="utf-8")
            _checked_protocol(path)
            base["profile"] = "strict"
            path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaises(ValueError):
                _checked_protocol(path)
            base["profile"] = "legacy"
            base["reproduction"]["passed"] = False
            path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaises(ValueError):
                _checked_protocol(path)

    def test_prediction_sha_and_image_order_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifact = root / "prediction.npz"
            np.savez_compressed(artifact, target=np.array([[[1]]], dtype=np.int16))
            sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
            row = dict(
                image="first", prediction_file="prediction.npz", prediction_sha256=sha
            )
            (root / "images.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            _checked_rows(root / "protocol.json", ["first"])
            with _artifact(root, row) as value:
                self.assertEqual(int(value["target"][0, 0, 0]), 1)
            row["prediction_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                _artifact(root, row)
            row["prediction_sha256"] = sha
            row["prediction_file"] = "../prediction.npz"
            with self.assertRaises(ValueError):
                _artifact(root, row)
            with self.assertRaises(ValueError):
                _checked_rows(root / "protocol.json", ["wrong"])


if __name__ == "__main__":
    unittest.main()
