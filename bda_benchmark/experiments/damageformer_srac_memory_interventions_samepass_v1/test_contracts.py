"""CPU-only contracts; no images, checkpoints or Test data are opened."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from .evaluate import (bank_consistent_shuffled_labels, check_seeds,
                       intervention_supports, uniform_matched_ids,
                       validate_reference_e1)
from .gates import formal_gate
from .launch import arguments_from_reference
from .report import rows_from_audit
from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import gather_support


class GateTests(unittest.TestCase):
    def test_report_only_accepts_formal_and_summarizes_transitions(self):
        names = ("files", "inherited_code", "new_code", "frozen_models", "runtime", "protocol")
        summary = dict(complete=True, full_validation_evidence=True, image_count=450)
        metric = dict(valid_miou=.6, iou=[.9, .7, .4, .7])
        transition = dict(classes=[{}, {}, dict(rescue=17, harm=2,
                                                 delta_false_positive=3), {}],
                          gt_old_new=[[[0] * 4 for _ in range(4)] for _ in range(4)])
        transition["gt_old_new"][1][1][2] = 5
        audit = dict(format="SRAC_MEMORY_INTERVENTIONS_SAMEPASS_V1", complete=True,
                     full_validation_evidence=True, image_count=450,
                     reproduction={"passed": True}, immutable={name: True for name in names},
                     random_seeds=[7], metrics={name: metric for name in (
                         "a1_pixel", "j1", "j2", "j2_random_7", "j2_shuffle_labels",
                         "j2_off", "j3", "j3_random_7", "j3_shuffle_labels", "j3_off")},
                     transitions={"j1_to_j2": transition,
                                  "j1_to_j2_random_7": transition,
                                  "j1_to_j2_shuffle_labels": transition,
                                  "j1_to_j2_off": transition,
                                  "j1_to_j3": transition,
                                  "j1_to_j3_random_7": transition,
                                  "j1_to_j3_shuffle_labels": transition,
                                  "j1_to_j3_off": transition})
        rows = rows_from_audit(summary, audit)
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[2]["damaged_rescue_vs_j1"], 17)
        self.assertEqual(rows[2]["intact_correct_to_damaged_vs_j1"], 5)
        summary["full_validation_evidence"] = False
        with self.assertRaises(ValueError):
            rows_from_audit(summary, audit)

    def test_only_full_reproduced_immutable_val450_is_formal(self):
        checks = {"files": True, "runtime": True}
        self.assertTrue(formal_gate(450, 450, {"passed": True}, checks, True, True, True))
        self.assertFalse(formal_gate(449, 450, {"passed": True}, checks, True, True, True))
        self.assertFalse(formal_gate(450, 450, {"passed": False}, checks, True, True, True))
        self.assertFalse(formal_gate(450, 450, {"passed": True}, {"runtime": False}, True, True, True))
        self.assertFalse(formal_gate(450, 450, {"passed": True}, checks, False, True, True))
        self.assertFalse(formal_gate(450, 450, {"passed": True}, checks, True, False, True))
        self.assertFalse(formal_gate(450, 450, {"passed": True}, checks, True, True, False))

    def test_launcher_reuses_exact_e0_source_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "protocol").mkdir()
            original = dict(joint_run="/fixed/joint", a1_checkpoint="/fixed/a1",
                            dataset="/fixed/dataset", train_list="/fixed/train.txt",
                            val_list="/fixed/val.txt", device="cuda:0", workers=4,
                            seed=3407, profile="legacy", pool_mode="native", smoke=False)
            (root / "protocol" / "protocol.json").write_text(json.dumps(dict(
                format="SRAC_HARDQUERY_PROTOCOL_V1", complete=True,
                full_validation_evidence=True, scope="all", image_count=450,
                reproduction={"passed": True}, arguments=original)), encoding="utf-8")
            args = arguments_from_reference(root, root / "fresh", (7, 8), 9)
            self.assertEqual(args.dataset, original["dataset"])
            self.assertEqual(args.val_list, original["val_list"])
            self.assertEqual(args.random_seeds, (7, 8))
            self.assertEqual(args.shuffle_seed, 9)

    def test_seeds_must_be_distinct_and_safe(self):
        self.assertEqual(check_seeds((3, 4), 8), (3, 4))
        for seeds in ((), (3, 3), (-1,), (2**63 - 1,)):
            with self.assertRaises(ValueError):
                check_seeds(seeds, 8)

    def test_reference_e1_rejects_failed_or_partial(self):
        # check_artifact is expected to fail before these partial records can pass.
        with self.assertRaises((ValueError, KeyError)):
            validate_reference_e1({"complete": False}, {})


class InterventionTests(unittest.TestCase):
    def test_label_shuffle_is_bank_consistent_for_repeated_row_ids(self):
        labels = torch.eye(4)[torch.tensor([0, 1, 2, 3])]
        bank = {"distribution": labels}
        selected = torch.tensor([[1, 2, -1], [1, 3, -1], [2, 1, -1]])
        valid = selected >= 0
        distribution = labels[selected.clamp_min(0)].clone()
        shuffled = bank_consistent_shuffled_labels(
            selected, distribution, valid, bank,
            torch.Generator().manual_seed(8))
        self.assertTrue(torch.equal(shuffled[0, 0], shuffled[1, 0]))
        self.assertTrue(torch.equal(shuffled[0, 0], shuffled[2, 1]))
        self.assertTrue(torch.equal(shuffled[0, 1], shuffled[2, 0]))
        self.assertTrue(torch.equal(shuffled[~valid], distribution[~valid]))
        # A permutation over unique retrieved rows preserves their label set.
        original_unique = labels[torch.tensor([1, 2, 3])]
        shuffled_unique = torch.stack((shuffled[0, 0], shuffled[0, 1], shuffled[1, 1]))
        self.assertTrue(torch.equal(original_unique.sort(dim=0).values,
                                    shuffled_unique.sort(dim=0).values))

    def test_uniform_sampler_sparse_slots_small_pool_and_reproducibility(self):
        bank = dict(tokens=torch.zeros(4, 8), events=torch.tensor([0, 1, 2, 3]),
                    images=torch.tensor([0, 1, 2, 3]))
        selected = torch.tensor([[1, -1, 2], [3, -1, -1], [-1, -1, -1]])
        first = uniform_matched_ids(selected, bank, 0, -1,
                                    torch.Generator().manual_seed(100))
        second = uniform_matched_ids(selected, bank, 0, -1,
                                     torch.Generator().manual_seed(100))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first >= 0, selected >= 0))
        self.assertEqual(first[0, 1].item(), -1)
        self.assertNotEqual(first[0, 0].item(), first[0, 2].item())
        self.assertTrue(((first[first >= 0] >= 1) & (first[first >= 0] <= 3)).all())
        # The pool has exactly k rows, so a full query must get each once.
        full = torch.tensor([[1, 2, 3]])
        drawn = uniform_matched_ids(full, bank, 0, -1,
                                    torch.Generator().manual_seed(2))
        self.assertEqual(set(drawn[0].tolist()), {1, 2, 3})

    def test_uniform_sampler_rejects_forbidden_source_support(self):
        bank = dict(tokens=torch.zeros(3, 8), events=torch.tensor([0, 0, 1]),
                    images=torch.arange(3))
        with self.assertRaises(ValueError):
            uniform_matched_ids(torch.tensor([[0, 1]]), bank, 0, -1,
                                torch.Generator().manual_seed(1))

    def test_interventions_keep_bank_and_validity_and_event_exclusion(self):
        bank = dict(tokens=torch.arange(40, dtype=torch.float32).reshape(5, 8),
                    distribution=torch.eye(4)[torch.tensor([0, 1, 2, 3, 1])],
                    geometry=torch.zeros(5, 5), events=torch.arange(5),
                    images=torch.arange(5), disasters=torch.zeros(5, dtype=torch.long))
        selected = torch.tensor([[1, 2], [1, -1]], dtype=torch.long)
        original = gather_support(selected, bank)
        snapshot = {key: value.clone() for key, value in bank.items()}
        random_generators = {7: torch.Generator().manual_seed(7),
                             8: torch.Generator().manual_seed(8)}
        shuffled_generator = torch.Generator().manual_seed(9)
        variants, counts = intervention_supports(
            selected, original, bank, 0, -1, random_generators, shuffled_generator)
        self.assertEqual(set(variants), {"random_7", "random_8", "shuffle_labels", "off"})
        self.assertTrue(all(torch.equal(bank[k], v) for k, v in snapshot.items()))
        for mode in ("random_7", "random_8"):
            self.assertTrue(torch.equal(variants[mode][2], original[2]))
            self.assertTrue((variants[mode][3]["events"][original[2]] != 0).all())
            self.assertEqual(counts[mode]["valid_slots"], int(original[2].sum()))
        self.assertTrue(torch.equal(variants["shuffle_labels"][0], original[0]))
        self.assertTrue(torch.equal(variants["shuffle_labels"][3]["events"],
                                    original[3]["events"]))
        self.assertFalse(variants["off"][2].any())


if __name__ == "__main__":
    unittest.main()
