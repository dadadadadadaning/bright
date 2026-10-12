"""Synthetic paired transitions; no BRIGHT image, model, or Test split is read."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .paired_prediction import (
    ARMS, paired_report, summarize_cube, transition_cube,
)


def _cm(gt, prediction):
    valid = gt < 4
    return np.bincount(gt[valid] * 4 + prediction[valid], minlength=16).reshape(4, 4)


def _formal(arm, gt, prediction):
    confusion = (_cm(gt, prediction) * 450).tolist()
    metric = dict(valid_miou=0.5, iou=[0.9, 0.5, 0.4, 0.6], confusion=confusion)
    return dict(format="SRAC_Q_FACTORIAL_EVAL_V1", complete=True,
                full_validation_evidence=True, image_count=450,
                reproduction={"passed": True}, native_a1_replay_ok=True,
                test_accessed=False, immutable={"files": True},
                protocol_sha256="a" * 64, source_sha256="b" * 64,
                numerical_profile="legacy", pool_mode="native",
                checkpoint_sha256=("c" if arm == "baseline" else "d") * 64,
                arm=arm, checkpoint_steps=2000,
                training_hyperparameters=dict(steps=2000, eval_every=2000,
                                              batch=2, accum=8, workers=4,
                                              lr=1e-4, aux_weight=0.1,
                                              cue_gain=0.1, amp="bf16"),
                metrics={name: dict(metric) for name in
                         ("a1_pixel", "q_pixel_grid", "q_region_mean", *ARMS)},
                per_event={"synthetic": {name: dict(metric) for name in
                                         ("a1_pixel", "q_pixel_grid", "q_region_mean", *ARMS)}})


class PairedPredictionTests(unittest.TestCase):
    @staticmethod
    def sample():
        gt = np.array([[2, 2, 2, 1], [1, 0, 3, 255]], dtype=np.int16)
        old = np.array([[1, 2, 2, 1], [2, 0, 3, 3]], dtype=np.uint8)
        new = np.array([[2, 1, 0, 2], [1, 2, 3, 0]], dtype=np.uint8)
        return gt, old, new

    def test_transition_identity_and_false_positive_sources(self):
        gt, old, new = self.sample()
        cube = transition_cube(gt, old, new)
        row = summarize_cube(cube)
        damaged = row["classes"]["Damaged"]
        self.assertEqual((damaged["rescue"], damaged["harm"], damaged["correct_delta"]),
                         (1, 2, -1))
        self.assertEqual(damaged["rescue_from"]["Intact"], 1)
        self.assertEqual(damaged["harm_to"]["Background"], 1)
        self.assertEqual(damaged["harm_to"]["Intact"], 1)
        self.assertEqual(row["damaged_false_positive_by_gt"]["Intact"],
                         dict(added=1, removed=1, net_change=0))
        self.assertEqual(row["damaged_false_positive_by_gt"]["total"]["net_change"], 1)
        self.assertEqual(row["gt_damaged_prediction_destinations"]["Background"],
                         dict(baseline=0, candidate=1, delta=1))
        self.assertEqual(row["gt_damaged_prediction_destinations"]["Intact"],
                         dict(baseline=1, candidate=1, delta=0))
        self.assertEqual(int(cube.sum()), 7)  # Ignore=255 is excluded.
        np.testing.assert_array_equal(cube.sum(axis=2), _cm(gt, old))
        np.testing.assert_array_equal(cube.sum(axis=1), _cm(gt, new))

    def test_invalid_prediction_and_shape_are_rejected(self):
        gt, old, new = self.sample()
        bad = new.copy()
        bad[0, 0] = 4
        with self.assertRaises(ValueError):
            transition_cube(gt, old, bad)
        with self.assertRaises(ValueError):
            transition_cube(gt, old, new[:, :2])

    def test_formal_pair_checks_each_image_and_global_confusion(self):
        gt, old, new = self.sample()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.npz"
            second = root / "second.npz"
            np.savez_compressed(first, target=gt[None],
                                prediction_q_grid_probq=old[None], prediction_q_region=old[None])
            np.savez_compressed(second, target=gt[None],
                                prediction_q_grid_probq=new[None], prediction_q_region=new[None])
            base, candidate = _formal("baseline", gt, old), _formal("semantic_reconstruction", gt, new)

            def images(path, prediction):
                confusion = _cm(gt, prediction).tolist()
                return [(dict(index=index, image=f"synthetic_{index:04d}", event="synthetic",
                              confusion={arm: confusion for arm in ARMS}), path)
                        for index in range(450)]

            baseline_images, candidate_images = images(first, old), images(second, new)
            with patch("bda_benchmark.experiments.damageformer_srac_q_factorial_v1.paired_prediction.checked_eval",
                       side_effect=[(root / "baseline" / "summary.json", base),
                                    (root / "candidate" / "summary.json", candidate)]), \
                 patch("bda_benchmark.experiments.damageformer_srac_q_factorial_v1.paired_prediction.verified_images",
                       side_effect=[(base, base, baseline_images),
                                    (candidate, candidate, candidate_images)]):
                result = paired_report(root / "baseline", root / "candidate")
            self.assertTrue(result["complete"])
            self.assertEqual(result["image_count"], 450)
            self.assertEqual(result["arms"]["q_grid_probq"]["classes"]["Damaged"]["correct_delta"], -450)
            self.assertEqual(result["per_event"]["synthetic"]["q_region"]["classes"]["Damaged"]["harm"], 900)

            altered = [dict(row) for row, path in candidate_images]
            altered[0]["confusion"] = {arm: np.zeros((4, 4), dtype=np.int64).tolist() for arm in ARMS}
            corrupt_images = list(zip(altered, [path for _, path in candidate_images]))
            with patch("bda_benchmark.experiments.damageformer_srac_q_factorial_v1.paired_prediction.checked_eval",
                       side_effect=[(root / "baseline" / "summary.json", base),
                                    (root / "candidate" / "summary.json", candidate)]), \
                 patch("bda_benchmark.experiments.damageformer_srac_q_factorial_v1.paired_prediction.verified_images",
                       side_effect=[(base, base, baseline_images),
                                    (candidate, candidate, corrupt_images)]):
                with self.assertRaisesRegex(ValueError, "confusion disagrees"):
                    paired_report(root / "baseline", root / "candidate")

            corrupt_formal = copy.deepcopy(candidate)
            corrupt_formal["metrics"]["q_region"]["confusion"][2][2] += 1
            with patch("bda_benchmark.experiments.damageformer_srac_q_factorial_v1.paired_prediction.checked_eval",
                       side_effect=[(root / "baseline" / "summary.json", base),
                                    (root / "candidate" / "summary.json", corrupt_formal)]), \
                 patch("bda_benchmark.experiments.damageformer_srac_q_factorial_v1.paired_prediction.verified_images",
                       side_effect=[(base, base, baseline_images),
                                    (corrupt_formal, corrupt_formal, candidate_images)]):
                with self.assertRaisesRegex(ValueError, "confusion disagrees"):
                    paired_report(root / "baseline", root / "candidate")

            changed_gt = gt.copy()
            changed_gt[0, 0] = 1
            wrong_target = root / "changed_target.npz"
            np.savez_compressed(wrong_target, target=changed_gt[None],
                                prediction_q_grid_probq=new[None], prediction_q_region=new[None])
            wrong_images = [(row, wrong_target) for row, _ in candidate_images]
            with patch("bda_benchmark.experiments.damageformer_srac_q_factorial_v1.paired_prediction.checked_eval",
                       side_effect=[(root / "baseline" / "summary.json", base),
                                    (root / "candidate" / "summary.json", candidate)]), \
                 patch("bda_benchmark.experiments.damageformer_srac_q_factorial_v1.paired_prediction.verified_images",
                       side_effect=[(base, base, baseline_images),
                                    (candidate, candidate, wrong_images)]):
                with self.assertRaisesRegex(ValueError, "target differs"):
                    paired_report(root / "baseline", root / "candidate")


if __name__ == "__main__":
    unittest.main()
