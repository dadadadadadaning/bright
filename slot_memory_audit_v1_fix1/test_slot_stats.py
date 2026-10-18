"""Small analytical checks for mass, ignore handling and post-hoc aggregation."""
import unittest

import torch
from torch.nn import functional as F

from slot_stats import RegionAccumulator, slot_diagnostics


def fixture(feature, building, target, q=None, ids=None, count=1):
    """Independent dense reference for tiny test images only."""
    feature = torch.as_tensor(feature, dtype=torch.float32)
    building = torch.as_tensor(building, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.long)
    h, w = feature.shape[-2:]
    n = h * w
    if q is None:
        q = torch.zeros(1, 9, n)
        q[:, 0] = 1
    if ids is None:
        ids = torch.zeros(9, n, dtype=torch.long)
    if ids.ndim == 3:
        flat_ids = ids[0]
    else:
        flat_ids = ids
    b = F.interpolate(building, size=(h, w), mode="bilinear", align_corners=False).flatten(2)
    routes = torch.cat((q * (1 - b), q * b), dim=1)
    route_ids = torch.cat((2 * flat_ids, 2 * flat_ids + 1), dim=0)
    dense = torch.zeros(count * 2, n)
    for s in range(18):
        for pixel in range(n):
            dense[route_ids[s, pixel], pixel] += routes[0, s, pixel]
    mass = dense.sum(-1)
    means = dense @ feature[0].flatten(1).T
    means /= torch.where(mass > 0, mass, torch.ones_like(mass))[:, None]
    mass = mass.reshape(1, count, 2)
    parent_mass = mass.sum(-1, keepdim=True)
    slots = {"slot_f": means.reshape(1, count, 2, feature.shape[1]),
             "slot_mass": mass,
             "slot_fraction": mass / torch.where(parent_mass > 0, parent_mass, torch.ones_like(parent_mass)),
             "slot_valid": mass > 0, "route_q": routes, "route_ids": route_ids,
             "slot_count": 2 * count}
    return feature, q, ids, building, target, slots


class SlotDiagnosticsTests(unittest.TestCase):
    def test_constant_building_makes_identical_summaries(self):
        inputs = fixture([[[[1, 2], [3, 4]]]], [[[[0.75, 0.75], [0.75, 0.75]]]], [[[1, 2], [0, 3]]])
        d = slot_diagnostics(*inputs)
        r = d["region_rows"]
        self.assertAlmostEqual(r["feature_cosine"].item(), 1.0)
        self.assertAlmostEqual(r["feature_relative_l2"].item(), 0.0)
        self.assertAlmostEqual(r["label_total_variation"].item(), 0.0)
        self.assertAlmostEqual(r["label_js_bits"].item(), 0.0)
        self.assertAlmostEqual(r["building_probability_variance"].item(), 0.0)
        self.assertAlmostEqual(r["slot0_mass"].item(), 1.0)
        self.assertAlmostEqual(r["slot1_mass"].item(), 3.0)
        self.assertAlmostEqual(r["slot0_ess"].item(), 4.0)
        self.assertAlmostEqual(r["slot1_ess"].item(), 4.0)
        self.assertAlmostEqual(d["weights"]["all_labeled"].sum().item(), 4.0)

    def test_separated_features_and_labels(self):
        inputs = fixture([[[[1, 0]], [[0, 1]]]], [[[[0, 1]]]], [[[1, 2]]])
        d = slot_diagnostics(*inputs)
        r = d["region_rows"]
        self.assertAlmostEqual(r["feature_cosine"].item(), 0.0)
        self.assertAlmostEqual(r["label_total_variation"].item(), 1.0)
        self.assertAlmostEqual(r["label_js_bits"].item(), 1.0)
        self.assertAlmostEqual(r["slot0_ess"].item(), 1.0)
        self.assertAlmostEqual(r["slot1_ess"].item(), 1.0)
        self.assertEqual(d["query_labels"].tolist(), [[[0, 1, 0, 0], [0, 0, 1, 0]]])

    def test_ignore_mass_not_renormalized_before_pool(self):
        inputs = fixture([[[[1, 2]]]], [[[[0.5, 0.5]]]], [[[2, 255, 1, 1], [2, 255, 1, 1]]])
        d = slot_diagnostics(*inputs)
        expected = torch.tensor([[[0, 2 / 3, 1 / 3, 0], [0, 2 / 3, 1 / 3, 0]]])
        torch.testing.assert_close(d["query_labels"], expected)
        torch.testing.assert_close(d["query_label_mass"], torch.tensor([[0.75, 0.75]]))
        self.assertAlmostEqual(d["weights"]["all_labeled"].item(), 1.5)
        self.assertEqual(d["pixel_masks"]["all_labeled"].sum().item(), 6)

    def test_repeated_candidate_ids_do_not_inflate_ess(self):
        q = torch.zeros(1, 9, 4)
        q[:, 0] = 0.4
        q[:, 1] = 0.6
        inputs = fixture([[[[1, 2, 3, 4]]]], [[[[0.5, 0.5, 0.5, 0.5]]]], [[[1, 1, 1, 1]]], q=q)
        d = slot_diagnostics(*inputs)
        for field in ("parent_ess", "slot0_ess", "slot1_ess"):
            self.assertAlmostEqual(d["region_rows"][field].item(), 4.0, places=5)

    def test_effective_sample_size_for_concentrated_slot(self):
        inputs = fixture([[[[1, 2, 3, 4]]]], [[[[1, 0, 0, 0]]]], [[[1, 1, 1, 1]]])
        r = slot_diagnostics(*inputs)["region_rows"]
        self.assertAlmostEqual(r["slot0_ess"].item(), 3.0)
        self.assertAlmostEqual(r["slot1_ess"].item(), 1.0)
        self.assertAlmostEqual(r["parent_ess"].item(), 4.0)

    def test_empty_slot_and_zero_norm_are_missing_not_identical(self):
        inputs = fixture([[[[0, 0]]]], [[[[0, 0]]]], [[[1, 2]]])
        d = slot_diagnostics(*inputs)
        for field in ("feature_cosine", "feature_relative_l2", "label_total_variation", "label_js_bits", "slot1_ess"):
            self.assertTrue(torch.isnan(d["region_rows"][field]).all())
        self.assertEqual(d["query_valid"].tolist(), [[True, False]])
        acc = RegionAccumulator()
        acc.add(d["region_rows"], d["weights"])
        field = acc.summary()["strata"]["all_labeled"]["feature_cosine"]
        self.assertEqual(field["n"], 0)
        self.assertEqual(field["weight_sum"], 0)
        self.assertIsNone(field["mean"])

    def test_all_ignore_has_no_label_distance_or_label_weight(self):
        inputs = fixture([[[[1, 2]]]], [[[[0.5, 0.5]]]], [[[255, 255]]])
        d = slot_diagnostics(*inputs)
        self.assertTrue(torch.isnan(d["region_rows"]["label_total_variation"]).all())
        self.assertEqual(d["query_valid"].sum().item(), 0)
        self.assertEqual(sum(v.sum().item() for v in d["weights"].values()), 0)

    def test_boundary_padding_and_ignore_do_not_create_background(self):
        inputs = fixture(torch.ones(1, 1, 3, 3), torch.full((1, 1, 3, 3), 0.5), [[[255, 1, 1], [1, 1, 1], [1, 1, 1]]])
        d = slot_diagnostics(*inputs)
        self.assertEqual(d["pixel_masks"]["gt_building_boundary"].sum().item(), 0)
        self.assertEqual(d["pixel_masks"]["gt_building_interior"].sum().item(), 8)

    def test_binary_boundary_and_low_B_are_independent_overlapping_strata(self):
        target = torch.ones(1, 5, 5, dtype=torch.long)
        target[:, :, 0] = 0
        building = torch.full((1, 1, 5, 5), 0.8)
        building[:, :, :, 1] = 0.2  # boundary buildings
        building[:, :, 2, 3] = 0.2  # one interior building
        d = slot_diagnostics(*fixture(torch.ones(1, 1, 5, 5), building, target))
        masks = d["pixel_masks"]
        self.assertEqual(masks["gt_building_boundary"].sum().item(), 5)
        self.assertEqual(masks["gt_building_interior"].sum().item(), 15)
        self.assertEqual(masks["low_B_true_building"].sum().item(), 6)
        self.assertEqual((masks["gt_building_boundary"] & masks["low_B_true_building"]).sum().item(), 5)
        self.assertEqual((masks["gt_building_interior"] & masks["low_B_true_building"]).sum().item(), 1)
        self.assertTrue(torch.equal(masks["gt_building_boundary"] | masks["gt_building_interior"], masks["gt_building"]))

    def test_parent_and_slot_mass_conservation_with_multiple_parents(self):
        q = torch.zeros(1, 9, 4)
        q[:, 0] = 0.75
        q[:, 1] = 0.25
        ids = torch.zeros(1, 9, 4, dtype=torch.long)
        ids[:, 1] = 1
        d = slot_diagnostics(*fixture([[[[1, 2, 3, 4]]]], [[[[0.1, 0.2, 0.7, 0.9]]]], [[[0, 1, 2, 3]]], q, ids, 2))
        r = d["region_rows"]
        torch.testing.assert_close(r["parent_mass"], torch.tensor([3., 1.]))
        torch.testing.assert_close(r["slot0_mass"] + r["slot1_mass"], r["parent_mass"])
        torch.testing.assert_close(r["slot0_fraction"] + r["slot1_fraction"], torch.ones(2))
        self.assertAlmostEqual(d["query_label_mass"].sum().item(), 4.0)
        self.assertAlmostEqual(sum(d["weights"][name].sum().item() for name in ("gt_intact", "gt_damaged", "gt_destroyed")), 3.0)

    def test_aggregation_is_mass_weighted_with_per_metric_denominators(self):
        acc = RegionAccumulator()
        acc.add({"a": torch.tensor([1.]), "b": torch.tensor([float("nan")])}, {"all": torch.tensor([1.])})
        acc.add({"a": torch.tensor([0.]), "b": torch.tensor([0.5])}, {"all": torch.tensor([9.])})
        result = acc.report()["strata"]["all"]
        self.assertAlmostEqual(result["a"]["mean"], 0.1)
        self.assertEqual(result["a"]["n"], 2)
        self.assertEqual(result["a"]["weight_sum"], 10)
        self.assertAlmostEqual(result["b"]["mean"], 0.5)
        self.assertEqual(result["b"]["n"], 1)
        self.assertEqual(result["b"]["weight_sum"], 9)

    def test_invalid_target_and_nonconserved_routes_rejected(self):
        inputs = list(fixture([[[[1, 2]]]], [[[[0.5, 0.5]]]], [[[1, 4]]]))
        with self.assertRaisesRegex(ValueError, "only 0..3"):
            slot_diagnostics(*inputs)
        inputs[4] = torch.tensor([[[1, 2]]])
        inputs[-1]["route_q"] *= 0.5
        with self.assertRaisesRegex(ValueError, "conserve"):
            slot_diagnostics(*inputs)


if __name__ == "__main__":
    unittest.main()
