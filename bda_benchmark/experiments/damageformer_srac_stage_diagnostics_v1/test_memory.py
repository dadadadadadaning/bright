"""Synthetic contracts; never a substitute for real BRIGHT/CUDA/full-Val audit."""
import copy
import unittest

import torch

from .memory_core import (
    cohort_evidence, distribution_inventory, label_evidence, project_label_evidence,
    random_matched_ids, reproduction_report, shuffle_support_labels,
    transition_counts, transition_report, validate_selected,
)


class MemoryContracts(unittest.TestCase):
    def bank(self):
        return dict(tokens=torch.eye(4), distribution=torch.eye(4),
                    events=torch.tensor([0, 1, 2, 3]), images=torch.tensor([0, 1, 2, 3]))

    def test_transition_rescue_harm_identity_and_ignore(self):
        gt = torch.tensor([2, 2, 2, 2, 1, 255])
        old = torch.tensor([2, 1, 0, 3, 1, 0])
        new = torch.tensor([1, 2, 3, 2, 2, 2])
        counts = transition_counts(gt, old, new)
        row = transition_report(counts)
        self.assertEqual(row["classes"][2]["harm"], 1)
        self.assertEqual(row["classes"][2]["rescue"], 2)
        self.assertEqual(row["classes"][2]["net_correct"], 1)
        self.assertEqual(row["classes"][2]["false_positive_old"], 0)
        self.assertEqual(row["classes"][2]["false_positive_new"], 1)
        self.assertEqual(row["classes"][2]["delta_false_positive"], 1)
        self.assertEqual(int(counts.sum()), 5)

    def test_random_matches_validity_and_exclusion_without_replacement(self):
        bank = self.bank()
        ids = torch.tensor([[1, 2, -1], [3, 1, 2], [-1, -1, -1]])
        result = random_matched_ids(ids, bank, 0, 0, torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(result >= 0, ids >= 0))
        validate_selected(result, bank, 0, 0)
        with self.assertRaises(ValueError):
            validate_selected(torch.tensor([[0]]), bank, 0, 0)
        with self.assertRaises(ValueError):
            validate_selected(torch.tensor([[1, 1]]), bank, 0, 0)

    def test_no_support_is_not_uniform_and_does_not_become_class_zero(self):
        bank = self.bank()
        ids = torch.tensor([[-1, -1], [2, -1]])
        evidence, valid, weight = label_evidence(torch.eye(4)[:2], ids, bank)
        self.assertTrue(torch.equal(evidence[0], torch.zeros(4)))
        self.assertFalse(bool(valid[0]))
        self.assertEqual(float(weight[0].sum()), 0)
        q = torch.ones(1, 1, 2)
        pixel_ids = torch.tensor([[0, 1]])
        probability, mass, pred = project_label_evidence(evidence, valid, q, pixel_ids, (1, 2), (1, 2))
        self.assertEqual(int(pred[0, 0, 0]), -1)
        self.assertEqual(int(pred[0, 0, 1]), 2)
        self.assertEqual(float(mass[0, 0, 0]), 0)

    def test_partial_support_retains_unsupported_mass(self):
        evidence = torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 0.]])
        q = torch.tensor([[[0.1], [0.9]]])
        ids = torch.tensor([[0], [1]])
        probability, mass, pred = project_label_evidence(evidence, torch.tensor([True, False]), q, ids, (1, 1), (1, 1))
        self.assertAlmostEqual(float(mass.item()), .1, places=6)
        self.assertEqual(int(pred.item()), 2)
        row = cohort_evidence(torch.tensor([[[2]]]), torch.tensor([[[1]]]), torch.tensor([[[2]]]),
                              pred, mass, probability)["damaged_j1_wrong"]
        self.assertAlmostEqual(row["unsupported_mass_sum"], .9, places=6)
        self.assertEqual(row["evidence_damaged_and_final_damaged"], 1)

    def test_shuffle_changes_only_valid_label_payload(self):
        d = torch.eye(4).reshape(2, 2, 4)
        before = d.clone()
        valid = torch.tensor([[True, False], [True, True]])
        output = shuffle_support_labels(d, valid, torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(d, before))
        self.assertTrue(torch.equal(output[~valid], d[~valid]))
        self.assertTrue(torch.equal(output[valid].sum(0), d[valid].sum(0)))

    def test_inventory_does_not_hide_minority_damaged(self):
        d = torch.tensor([[0., .8, .2, 0.], [0., .1, .9, 0.]])
        result = distribution_inventory(d)
        self.assertEqual(result["dominant_counts"][2], 1)
        self.assertEqual(result["damaged_positive_regions"], 2)
        self.assertAlmostEqual(result["damaged_soft_sum_non_damaged_dominant"], .2, places=6)

    def test_reproduction_fails_without_relaxing_and_smoke_not_full(self):
        metric = dict(valid_miou=.6, iou=[.9, .6, .4, .8], precision=[.9] * 4,
                      recall=[.8] * 4, f1=[.85] * 4)
        actual = {n: copy.deepcopy(metric) for n in ("a1_pixel", "direct_region", "j1", "j2", "j3")}
        expected = copy.deepcopy(actual)
        self.assertTrue(reproduction_report(actual, expected, image_count=450)["passed"])
        actual["j3"]["recall"][2] += 2e-5
        self.assertFalse(reproduction_report(actual, expected, image_count=450)["passed"])
        self.assertIsNone(reproduction_report(actual, expected, smoke=True, image_count=2)["passed"])

    def test_completed_head_gate_rejects_best_and_mismatched_source(self):
        from .memory_audit import validate_head, validate_joint_header
        manifest = dict(format="SRAC_JOINT123_V1", smoke=False, probe=False, test_accessed=False,
                        arguments=dict(steps=5000), **{key: "abc" for key in
                        ("source_hash", "front_sha256", "a1_sha256", "cache_manifest_sha256", "bank_sha256", "metric_sha256")})
        summary = dict(complete=True, run_complete=True, frozen_fingerprints_unchanged=True,
                       smoke=False, probe=False, test_accessed=False, train_image_counts=dict(j1=5000, j2=5000, j3=5000))
        self.assertEqual(validate_joint_header(manifest, summary), 5000)
        payload = dict(**manifest, arm="j1", step=5000, completed_steps=5000, run_complete=True)
        validate_head(payload, manifest, "j1", 5000)
        payload["step"] = 2000
        with self.assertRaises(ValueError):
            validate_head(payload, manifest, "j1", 5000)
        summary["complete"] = False
        with self.assertRaises(ValueError):
            validate_joint_header(manifest, summary)

    def test_quality_inventory_mass_is_feature_grid_not_object_area(self):
        from .memory_audit import quality_inventory
        d = torch.tensor([[0., .6, .4, 0.], [0., .1, .9, 0.], [0., 0., 1., 0.]])
        report = quality_inventory(d, torch.tensor([8., 32., 96.]))
        self.assertEqual(report["purity"]["lt_080"]["regions"], 1)
        self.assertEqual(report["purity"]["080_095"]["regions"], 1)
        self.assertEqual(report["purity"]["ge_095"]["regions"], 1)
        self.assertEqual(report["q_mass"]["lt_16_cells"]["regions"], 1)
        self.assertEqual(report["q_mass"]["16_64_cells"]["regions"], 1)
        self.assertEqual(report["q_mass"]["ge_64_cells"]["regions"], 1)

    def test_hash_ledger_detects_changed_file(self):
        from unittest.mock import patch
        from .memory_audit import input_fingerprints
        with patch("bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit.digest",
                   side_effect=["before", "after"]):
            first = input_fingerprints({"front": "immutable_front.pth"})
            second = input_fingerprints({"front": "immutable_front.pth"})
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
