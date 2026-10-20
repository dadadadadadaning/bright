"""CPU contracts for replay sample attribution and numerical drift reporting."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import trajectory
from trajectory import METRICS, TrajectoryComparison, reconstruct_order


ARMS = ("e", "f", "b", "d", "r", "l")
SEED = 3407


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "train.jsonl"
        self.names = ["synthetic_%06d" % i for i in range(2918)]
        self.manifest = {"seed": SEED, "train_names": self.names}
        self.plan = {"arms": list(ARMS)}
        self.order = list(torch.utils.data.DataLoader(
            self.names, batch_size=1, shuffle=True, num_workers=0,
            drop_last=False, generator=torch.Generator().manual_seed(SEED),
        ))
        self.order = [str(batch[0]) for batch in self.order]
        self.rows = [
            dict(step=i, image=self.order[i - 1], epoch=0, epoch_seed=SEED,
                 arms={arm: {"optimizer_step": i} for arm in ARMS})
            for i in range(1, 1397)
        ]
        self.write()

    def write(self, suffix=""):
        self.path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.rows) + suffix,
            encoding="utf-8",
        )

    def reconstruct(self):
        return reconstruct_order(torch, self.manifest, self.plan, self.path)

    def test_reconstructs_complete_prefix_and_candidate_without_global_rng_change(self):
        before_rng = torch.get_rng_state().clone()
        before_file = self.path.read_bytes()
        candidate, order, rows = self.reconstruct()
        self.assertEqual(order, self.order)
        self.assertEqual(candidate["candidate_step"], 1397)
        self.assertEqual(candidate["image"], self.order[1396])
        self.assertEqual(candidate["verified_prefix_count"], 1396)
        self.assertEqual(rows, {row["step"]: row for row in self.rows})
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        self.assertEqual(self.path.read_bytes(), before_file)
        # JSON serialization sorts arm names; only their identities matter.
        self.assertNotEqual(tuple(rows[1]["arms"]), ARMS)

    def test_bare_randperm_order_is_rejected(self):
        indices = torch.randperm(
            len(self.names), generator=torch.Generator().manual_seed(SEED),
        ).tolist()
        self.assertNotEqual(self.names[indices[0]], self.order[0])
        self.rows[0]["image"] = self.names[indices[0]]
        self.write()
        with self.assertRaises(ValueError):
            self.reconstruct()

    def test_earlier_identity_or_partial_update_error_is_rejected(self):
        for mutate in (
            lambda row: row.update(image="different_image"),
            lambda row: row.update(epoch=1),
            lambda row: row.update(epoch_seed=SEED + 1),
            lambda row: row["arms"]["e"].update(optimizer_step=3),
            lambda row: row["arms"].pop("l"),
        ):
            with self.subTest(mutate=mutate):
                original = copy.deepcopy(self.rows[0])
                mutate(self.rows[0])
                self.write()
                with self.assertRaises(ValueError):
                    self.reconstruct()
                self.rows[0] = original

    def test_truncated_last_record_is_rejected(self):
        self.write('{"step":1397')
        with self.assertRaises(ValueError):
            self.reconstruct()

    def test_mutation_after_candidate_verification_is_rejected(self):
        real = trajectory.candidate_from_journal

        def alter_after_verification(*args):
            result = real(*args)
            self.rows[0]["image"] = "changed_during_read"
            self.write()
            return result

        with patch.object(trajectory, "candidate_from_journal", alter_after_verification):
            with self.assertRaisesRegex(ValueError, "changed"):
                self.reconstruct()


class ComparisonTests(unittest.TestCase):
    def rows(self, step=1001, image="train_image"):
        arms = {
            arm: dict(
                optimizer_step=step, lr=1e-4, original_loss=0.75,
                nll=0.25, lovasz=0.5, grad_norm_preclip=0.2,
                reader_grad_abs_sum=0.05, parameter_update_l2=None,
                parameter_l2=None,
            )
            for arm in ARMS
        }
        return copy.deepcopy(arms), dict(step=step, image=image, arms=arms)

    def test_exact_match_and_paired_nulls_are_counted(self):
        rows, original = self.rows()
        rows = dict(reversed(list(rows.items())))
        comparison = TrajectoryComparison(ARMS)
        comparison.observe(1001, "train_image", rows, original)
        summary = comparison.summary()
        self.assertEqual(summary["numeric_comparisons"], len(ARMS) * 5)
        self.assertEqual(summary["exact_matches"], len(ARMS) * 5)
        self.assertEqual(summary["within_tolerance"], len(ARMS) * 5)
        self.assertEqual(summary["paired_none"], len(ARMS) * 2)
        self.assertTrue(summary["all_exact"])
        self.assertTrue(summary["all_within_tolerance"])
        self.assertIsNone(summary["first_difference"])
        self.assertEqual(set(summary["per_field"]), set(METRICS))
        json.dumps(summary, allow_nan=False)

    def test_first_difference_and_later_drift_are_retained_without_aborting(self):
        comparison = TrajectoryComparison(ARMS)
        rows, original = self.rows()
        rows["e"]["nll"] += 1e-6
        rows["f"]["lovasz"] += 0.01
        comparison.observe(1001, "train_image", rows, original)
        rows, original = self.rows(1002, "next_image")
        rows["b"]["grad_norm_preclip"] += 0.2
        comparison.observe(1002, "next_image", rows, original)
        summary = comparison.summary()
        self.assertEqual(summary["steps_compared"], 2)
        self.assertEqual(summary["first_step"], 1001)
        self.assertEqual(summary["last_step"], 1002)
        self.assertEqual(summary["first_difference"]["arm"], "e")
        self.assertEqual(summary["first_difference"]["field"], "nll")
        self.assertEqual(summary["first_outside_tolerance"]["arm"], "f")
        self.assertEqual(summary["outside_tolerance"], 2)
        self.assertEqual(summary["exact_matches"], 57)
        self.assertEqual(summary["within_tolerance"], 58)
        self.assertEqual(summary["max_absolute_difference"]["step"], 1002)
        self.assertEqual(summary["max_absolute_difference"]["field"], "grad_norm_preclip")
        self.assertAlmostEqual(summary["max_absolute_difference"]["abs_diff"], 0.2)
        self.assertFalse(summary["all_exact"])
        self.assertFalse(summary["all_within_tolerance"])
        self.assertEqual(summary["per_field"]["lovasz"]["outside_tolerance"], 1)
        json.dumps(summary, allow_nan=False)

    def test_relative_tolerance_uses_original_value(self):
        comparison = TrajectoryComparison(ARMS, atol=0, rtol=0.1)
        rows, original = self.rows()
        original["arms"]["e"]["original_loss"] = 1.0
        rows["e"]["original_loss"] = 1.105
        comparison.observe(1001, "train_image", rows, original)
        self.assertEqual(comparison.summary()["outside_tolerance"], 1)

    def test_structural_counter_lr_and_nonfinite_errors_abort_atomically(self):
        cases = {
            "original step": lambda rows, orig: orig.update(step=1000),
            "original image": lambda rows, orig: orig.update(image="wrong"),
            "missing arm": lambda rows, orig: rows.pop("l"),
            "extra arm": lambda rows, orig: rows.update(x=copy.deepcopy(rows["e"])),
            "original arm": lambda rows, orig: orig["arms"].pop("l"),
            "counter": lambda rows, orig: rows["l"].update(optimizer_step=1000),
            "boolean counter": lambda rows, orig: rows["l"].update(optimizer_step=True),
            "original counter": lambda rows, orig: orig["arms"]["l"].update(optimizer_step=1000),
            "lr drift": lambda rows, orig: rows["l"].update(lr=1e-4 + 1e-15),
            "missing lr": lambda rows, orig: rows["l"].pop("lr"),
            "nan lr": lambda rows, orig: rows["l"].update(lr=float("nan")),
            "nan metric": lambda rows, orig: rows["l"].update(nll=float("nan")),
            "infinite reference": lambda rows, orig: orig["arms"]["l"].update(nll=float("inf")),
            "missing metric": lambda rows, orig: rows["l"].pop("nll"),
            "unpaired null": lambda rows, orig: rows["l"].update(parameter_l2=0.1),
            "boolean metric": lambda rows, orig: rows["l"].update(nll=True),
        }
        for label, mutate in cases.items():
            with self.subTest(case=label):
                rows, original = self.rows()
                mutate(rows, original)
                comparison = TrajectoryComparison(ARMS)
                before = comparison.summary()
                with self.assertRaises(ValueError):
                    comparison.observe(1001, "train_image", rows, original)
                self.assertEqual(comparison.summary(), before)

    def test_observations_must_be_consecutive(self):
        for next_step in (1001, 1003):
            with self.subTest(next_step=next_step):
                comparison = TrajectoryComparison(ARMS)
                rows, original = self.rows()
                comparison.observe(1001, "train_image", rows, original)
                before = comparison.summary()
                rows, original = self.rows(next_step)
                with self.assertRaises(ValueError):
                    comparison.observe(next_step, "train_image", rows, original)
                self.assertEqual(comparison.summary(), before)

    def test_summary_is_detached_and_inputs_are_not_modified(self):
        rows, original = self.rows()
        rows["e"]["nll"] += 0.5
        before = copy.deepcopy((rows, original))
        comparison = TrajectoryComparison(ARMS)
        comparison.observe(1001, "train_image", rows, original)
        report = comparison.summary()
        report["first_difference"]["actual"] = -10
        report["per_field"]["nll"]["numeric_comparisons"] = -1
        self.assertEqual((rows, original), before)
        self.assertGreater(comparison.summary()["first_difference"]["actual"], 0)
        self.assertEqual(comparison.summary()["per_field"]["nll"]["numeric_comparisons"], 6)

    def test_invalid_arms_or_tolerances_are_rejected(self):
        for arms, atol, rtol in (
            ((), 1e-5, 1e-4), (("e", "e"), 1e-5, 1e-4),
            (ARMS, -1, 1e-4), (ARMS, 1e-5, float("inf")),
        ):
            with self.subTest(arms=arms, atol=atol, rtol=rtol):
                with self.assertRaises(ValueError):
                    TrajectoryComparison(arms, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
