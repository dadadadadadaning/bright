"""Small exact CPU contracts for offline pool evidence and its coordinate map."""
import copy
import unittest

import torch

from .core import bin_exposure, label_fractions, region_evidence


def route(q, ids, k, shape):
    return {"q": torch.tensor(q, dtype=torch.float64),
            "ids": torch.tensor(ids, dtype=torch.long), "k": k, "shape": shape}


class PoolEvidenceContracts(unittest.TestCase):
    def test_attention_and_stats_measure_different_actual_content(self):
        r = route([[[1, 1]]], [[[0, 0]]], 1, (1, 2))
        result = region_evidence(r, torch.tensor([[[[1., .25]]]]), torch.tensor([[[0, 2]]]))
        stats, attention = result["stats"], result["attention"]
        torch.testing.assert_close(stats["class_mass"], torch.tensor([[[1., 0., 1., 0.]]], dtype=torch.float64))
        self.assertAlmostEqual(stats["foreground_fraction"].item(), .5)
        self.assertAlmostEqual(attention["foreground_fraction"].item(), .2)
        self.assertAlmostEqual(stats["ess"].item(), 2.)
        self.assertAlmostEqual(attention["ess"].item(), 1.25 ** 2 / 1.0625)
        self.assertAlmostEqual(attention["mass"].item(), 1.25)

    def test_ignored_gt_has_independent_denominator(self):
        r = route([[[1, 1]]], [[[0, 0]]], 1, (1, 2))
        result = region_evidence(r, torch.ones(1, 1, 1, 2), torch.tensor([[[255, 2]]]))["stats"]
        self.assertEqual(result["mass"].item(), 2.)
        self.assertEqual(result["valid_mass"].item(), 1.)
        self.assertEqual(result["valid_fraction"].item(), .5)
        self.assertEqual(result["purity"].item(), 1.)
        self.assertEqual(result["foreground_fraction"].item(), 1.)

    def test_full_resolution_small_gt_survives_area_mapping(self):
        gt = torch.tensor([[[0, 0], [0, 2]]])
        fractions = label_fractions(gt, (1, 1))
        torch.testing.assert_close(fractions, torch.tensor([[[.75, 0., .25, 0.]]], dtype=torch.float64))
        r = route([[[1]]], [[[0]]], 1, (1, 1))
        result = region_evidence(r, torch.ones(1, 1, 1, 1), gt)["stats"]
        self.assertEqual(result["foreground_fraction"].item(), .25)
        self.assertEqual(result["foreground_purity"].item(), 1.)

    def test_noninteger_area_mapping_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "integer area factors"):
            label_fractions(torch.zeros(1, 3, 4, dtype=torch.long), (2, 2))

    def test_empty_and_tiny_slots_are_distinct_and_finite(self):
        r = route([[[1e-12], [1 - 1e-12]]], [[[0], [1]]], 3, (1, 1))
        result = region_evidence(r, torch.ones(1, 1, 1, 1), torch.tensor([[[2]]]))["stats"]
        self.assertEqual(result["occupied"].tolist(), [[True, True, False]])
        self.assertEqual(result["purity"].tolist(), [[1., 1., 0.]])
        self.assertEqual(result["foreground_purity"].tolist(), [[1., 1., 0.]])
        self.assertAlmostEqual(result["ess"][0, 0].item(), 1.)
        self.assertTrue(result["slot_masks"]["mass__tiny"][0, 0])
        for value in result.values():
            if torch.is_tensor(value):
                self.assertTrue(torch.isfinite(value).all())

    def test_duplicate_candidates_are_coalesced_before_ess(self):
        r = route([[[.4, .2], [.6, .8]]], [[[0, 0], [0, 0]]], 1, (1, 2))
        result = region_evidence(r, torch.ones(1, 1, 1, 2), torch.tensor([[[1, 2]]]))["stats"]
        self.assertEqual(result["mass"].item(), 2.)
        self.assertEqual(result["weight_square_sum"].item(), 2.)
        self.assertEqual(result["ess"].item(), 2.)

    def test_full_bins_cover_read_exposure_and_bilinear_mapping(self):
        r = route([[[.75, .25], [.25, .75]]], [[[0, 0], [1, 1]]], 2, (1, 2))
        bins = {"a": torch.tensor([[True, False]]), "b": torch.tensor([[False, True]])}
        out = bin_exposure(r, bins, (1, 4))
        torch.testing.assert_close(out.sum(1), torch.ones(1, 1, 4))
        torch.testing.assert_close(out[0, 0], torch.tensor([[.75, .625, .375, .25]]))

    def test_foreground_subset_exposure_is_not_renormalized(self):
        r = route([[[.75, .25], [.25, .75]]], [[[0, 0], [1, 1]]], 2, (1, 2))
        bins = {"foreground_only": torch.tensor([[False, True]])}
        out = bin_exposure(r, bins, (1, 2))
        torch.testing.assert_close(out, torch.tensor([[[[.25, .75]]]]))
        self.assertTrue((out.sum(1) <= 1).all())

    def test_pure_background_not_counted_as_pure_foreground(self):
        r = route([[[1, 1]]], [[[0, 1]]], 2, (1, 2))
        result = region_evidence(r, torch.ones(1, 1, 1, 2), torch.tensor([[[0, 2]]]))["stats"]
        self.assertEqual(result["foreground_valid"].tolist(), [[False, True]])
        self.assertEqual(result["slot_masks"]["foreground_purity__high"].tolist(), [[False, True]])

    def test_bin_families_are_mutually_exclusive_and_correctly_scoped(self):
        r = route([[[1, 1, 1, 1]]], [[[0, 1, 2, 3]]], 5, (1, 4))
        result = region_evidence(r, torch.ones(1, 1, 1, 4), torch.tensor([[[0, 1, 2, 255]]]))["stats"]
        for family in ("mass", "ess", "semantic_purity", "binary_purity", "foreground_purity"):
            masks = [value for name, value in result["slot_masks"].items() if name.startswith(family + "__")]
            count = torch.stack(masks).long().sum(0)
            self.assertTrue((count <= 1).all())
            expected = result["foreground_valid"] if family == "foreground_purity" else (
                result["gt_valid"] if "purity" in family else result["occupied"])
            self.assertTrue(torch.equal(count.bool(), expected))

    def test_gt_is_diagnostic_only_and_no_inputs_mutate(self):
        r = route([[[.5, .2], [.5, .8]]], [[[0, 0], [1, 1]]], 2, (1, 2))
        r["q"].requires_grad_(True)
        before = copy.deepcopy(r)
        attention = torch.full((1, 1, 1, 2), .8, requires_grad=True)
        first = region_evidence(r, attention, torch.tensor([[[0, 1]]]))
        second = region_evidence(r, attention, torch.tensor([[[2, 3]]]))
        torch.testing.assert_close(r["q"], before["q"], rtol=0, atol=0)
        self.assertTrue(torch.equal(r["ids"], before["ids"]))
        torch.testing.assert_close(first["stats"]["mass"], second["stats"]["mass"], rtol=0, atol=0)
        self.assertFalse(first["attention"]["class_mass"].requires_grad)
        self.assertIsNone(r["q"].grad)
        self.assertIsNone(attention.grad)

    def test_invalid_route_or_attention_is_rejected(self):
        r = route([[[.9]]], [[[0]]], 1, (1, 1))
        with self.assertRaisesRegex(ValueError, "sum to one"):
            region_evidence(r, torch.ones(1, 1, 1, 1), torch.zeros(1, 1, 1, dtype=torch.long))
        r["q"].fill_(1)
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            region_evidence(r, torch.full((1, 1, 1, 1), 1.1), torch.zeros(1, 1, 1, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
