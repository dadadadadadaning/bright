import unittest

import numpy as np
import torch

from bda_benchmark.experiments.spi_selective_context_v1.core import (
    SelectiveContextConfig,
    foreground_region_distribution,
)
from bda_benchmark.script.standard_ML.spi_region_feedback.core import (
    gate_region,
    region_evidence,
)
from bda_benchmark.experiments.spi_s2_utility_gate_v1.core import (
    GATE_INPUT_CHANNELS,
    RegionLabelMap,
    UtilityGate,
    compute_transition_audit,
    derive_direction_policy,
    direction_policy_mask,
    empty_transition_audit,
    finalize_transition_audit,
    fixed_s2_candidate,
    fixed_s2_candidate_from_labels,
    merge_transition_audit,
    predicted_building_label_map,
    select_prediction,
    selector_targets,
    update_transition_audit,
    utility_features,
)


def legacy_candidate(probabilities, masks, config=None):
    """Small reference implementation of the original per-mask S2 loop."""

    config = config or SelectiveContextConfig()
    prediction = probabilities.argmax(0)
    output = probabilities.clone()
    support = torch.zeros_like(prediction, dtype=torch.bool)
    features = torch.zeros((6, *prediction.shape), dtype=torch.float32)
    records = []
    image_area = float(prediction.numel())
    for index, mask in enumerate(masks):
        evidence = region_evidence(prediction, probabilities, mask)
        accepted = gate_region(
            evidence,
            minimum_purity=config.minimum_purity,
            minimum_margin=config.minimum_margin,
        )
        distribution = foreground_region_distribution(probabilities, mask)
        changed = 0
        if accepted:
            mixed = ((1.0 - config.blend_strength) * output[:, mask]
                     + config.blend_strength * distribution[:, None])
            mixed /= mixed.sum(0, keepdim=True).clamp_min(1e-8)
            output[:, mask] = mixed
            support[mask] = True
            features[:, mask] = torch.tensor(
                [
                    evidence.purity,
                    evidence.margin,
                    np.log1p(evidence.area) / np.log1p(image_area),
                    float(distribution[1]),
                    float(distribution[2]),
                    float(distribution[3]),
                ],
                dtype=torch.float32,
                device=features.device,
            )[:, None]
            changed = int((output[:, mask].argmax(0) != prediction[mask]).sum())
        records.append({
            "region_index": index,
            "accepted": accepted,
            "area": evidence.area,
            "purity": evidence.purity,
            "margin": evidence.margin,
            "changed_pixels": changed,
        })
    return output, support, features, records


class UtilityGateCoreTests(unittest.TestCase):
    def test_label_map_keeps_four_connected_components_after_area_filter(self):
        localization = torch.tensor([
            [1, 1, 0, 0, 0],
            [1, 0, 0, 1, 0],
            [0, 0, 0, 1, 1],
        ])
        result = predicted_building_label_map(localization, minimum_area=2)
        self.assertEqual(result.count, 2)
        self.assertEqual(int((result.labels == 1).sum()), 3)
        self.assertEqual(int((result.labels == 2).sum()), 3)

    def test_vectorized_candidate_matches_original_per_mask_definition(self):
        generator = torch.Generator().manual_seed(7)
        logits = torch.randn((4, 7, 8), generator=generator)
        probabilities = logits.softmax(0)
        first = torch.zeros((7, 8), dtype=torch.bool)
        second = torch.zeros_like(first)
        first[1:4, 1:4] = True
        second[4:7, 5:8] = True
        labels = torch.zeros((7, 8), dtype=torch.long)
        labels[first] = 1
        labels[second] = 2
        config = SelectiveContextConfig(minimum_purity=0.0, minimum_margin=0.0)
        expected = legacy_candidate(probabilities, [first, second], config)
        actual = fixed_s2_candidate_from_labels(
            probabilities, RegionLabelMap(labels, 2), config
        )
        self.assertTrue(torch.allclose(actual.probabilities, expected[0], atol=1e-6))
        self.assertTrue(torch.equal(actual.support, expected[1]))
        self.assertTrue(torch.allclose(actual.region_features, expected[2], atol=1e-6))
        for observed, reference in zip(actual.records, expected[3]):
            self.assertEqual(observed["region_index"], reference["region_index"])
            self.assertEqual(observed["accepted"], reference["accepted"])
            self.assertEqual(observed["area"], reference["area"])
            self.assertEqual(observed["changed_pixels"], reference["changed_pixels"])
            self.assertAlmostEqual(observed["purity"], reference["purity"], places=6)
            self.assertAlmostEqual(observed["margin"], reference["margin"], places=6)

    def test_candidate_is_label_free_normalized_and_supported(self):
        probabilities = torch.full((4, 4, 4), 0.01)
        probabilities[1] = 0.60
        probabilities[2] = 0.38
        probabilities /= probabilities.sum(0, keepdim=True)
        mask = torch.ones((4, 4), dtype=torch.bool)
        result = fixed_s2_candidate(probabilities, [mask])
        self.assertTrue(result.support.all())
        self.assertTrue(torch.allclose(result.probabilities.sum(0), torch.ones(4, 4)))
        self.assertEqual(result.region_features.shape, (6, 4, 4))

    def test_features_and_gate_only_cover_changed_pixels(self):
        base = torch.zeros(1, 4, 2, 2)
        candidate = torch.zeros_like(base)
        base[:, 1] = 0.8
        base[:, 2] = 0.2
        candidate.copy_(base)
        candidate[:, 1, 0, 0] = 0.2
        candidate[:, 2, 0, 0] = 0.8
        support = torch.ones(1, 1, 2, 2)
        region = torch.zeros(1, 6, 2, 2)
        features, changed = utility_features(base, candidate, support, region)
        self.assertEqual(features.shape[1], GATE_INPUT_CHANNELS)
        self.assertEqual(int(changed.sum()), 1)
        gate = UtilityGate()(features, changed)
        self.assertEqual(int((gate > 0).sum()), 1)

    def test_selector_ignores_wrong_to_wrong(self):
        base = torch.tensor([[[1, 1, 1]]])
        candidate = torch.tensor([[[2, 2, 2]]])
        target = torch.tensor([[[2, 1, 3]]])
        selector, decisive, positive, negative = selector_targets(base, candidate, target)
        self.assertEqual(int(positive.sum()), 1)
        self.assertEqual(int(negative.sum()), 1)
        self.assertEqual(int(decisive.sum()), 2)
        self.assertEqual(float(selector[0, 0, 0, 0]), 1.0)

    def test_transition_audit_and_policy(self):
        base_prob = torch.nn.functional.one_hot(torch.tensor([[1, 1, 1]]), 4).permute(2, 0, 1).float()
        candidate_prob = torch.nn.functional.one_hot(torch.tensor([[2, 2, 2]]), 4).permute(2, 0, 1).float()
        target = torch.tensor([[2, 1, 3]])
        audit = empty_transition_audit()
        update_transition_audit(audit, base_prob, candidate_prob, target, torch.ones_like(target, dtype=torch.bool))
        final = finalize_transition_audit(audit)
        self.assertEqual((final["corrected"], final["harmed"], final["neutral"]), (1, 1, 1))
        policy = derive_direction_policy(audit, {"event": audit}, minimum_pixels=1,
                                         minimum_decisive_precision=0.5,
                                         minimum_event_nonnegative_fraction=1.0)
        self.assertFalse(policy["policy"]["1->2"])

    def test_one_image_audit_can_be_reused_for_global_and_event_totals(self):
        base = torch.nn.functional.one_hot(torch.tensor([[1, 1, 2]]), 4).permute(2, 0, 1).float()
        candidate = torch.nn.functional.one_hot(torch.tensor([[2, 1, 3]]), 4).permute(2, 0, 1).float()
        target = torch.tensor([[2, 1, 2]])
        image = compute_transition_audit(base, candidate, target)
        global_total = empty_transition_audit()
        event_total = empty_transition_audit()
        merge_transition_audit(global_total, image)
        merge_transition_audit(event_total, image)
        self.assertEqual(global_total, event_total)
        self.assertEqual((global_total["corrected"], global_total["harmed"]), (1, 1))

    def test_direction_selection(self):
        base = torch.tensor([[[1, 1, 2]]])
        candidate = torch.tensor([[[2, 3, 1]]])
        mask = direction_policy_mask(base, candidate, {"1->2": True})
        output = select_prediction(base, candidate, mask)
        self.assertTrue(torch.equal(output, torch.tensor([[[2, 1, 2]]])))


if __name__ == "__main__":
    unittest.main()
