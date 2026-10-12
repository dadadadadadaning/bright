import unittest

import torch

from bda_benchmark.model.DamageFormer_RGBSAR_H7_CER_V2 import (
    KEEP_P3,
    REJECT_BOTH,
    USE_A2,
    PairConditionalRejectableSelector,
)
from bda_benchmark.script.standard_ML.train_DamageFormer_RGBSAR_H7_CER_V2 import (
    event_balanced_sampler,
    event_name,
    event_pair_balanced_selector_loss,
    selector_targets,
)
from bda_benchmark.script.standard_ML.analyze_H7_P3_A2_oracle import (
    fast_histogram,
)


def logits_from_prediction(prediction: torch.Tensor) -> torch.Tensor:
    logits = torch.full(
        (prediction.shape[0], 4, *prediction.shape[-2:]), -6.0
    )
    return logits.scatter_(1, prediction.long(), 6.0)


class H7CERV2Tests(unittest.TestCase):
    def test_exact_p3_and_zero_route_at_initialization(self):
        selector = PairConditionalRejectableSelector()
        p3 = torch.randn(2, 4, 8, 8)
        a2 = torch.randn_like(p3)
        selector.set_context(torch.randn(2, 1, 8, 8), torch.randn(2, 2, 8, 8))
        output, diagnostics = selector(p3, a2)
        self.assertTrue(torch.equal(output, p3))
        self.assertEqual(int(diagnostics["cer_v2_route_open_mask"].sum()), 0)

    def test_only_two_background_correction_pairs_are_candidates(self):
        selector = PairConditionalRejectableSelector(
            building_threshold=1.0, boundary_threshold=0.0
        )
        p3_pred = torch.tensor([[[[2, 3, 1, 0]]]])
        a2_pred = torch.tensor([[[[0, 0, 2, 3]]]])
        p3 = logits_from_prediction(p3_pred)
        a2 = logits_from_prediction(a2_pred)
        selector.set_context(torch.zeros(1, 1, 1, 4), torch.zeros(1, 2, 1, 4))
        _, diagnostics = selector(p3, a2)
        candidate = diagnostics["cer_v2_candidate_mask"].bool().flatten()
        self.assertEqual(candidate.tolist(), [True, True, False, False])

    def test_keep_use_reject_targets(self):
        p3 = torch.tensor([[[1, 2, 3]]])
        a2 = torch.tensor([[[0, 0, 0]]])
        truth = torch.tensor([[[1, 0, 2]]])
        target, supervised = selector_targets(
            p3, a2, truth, torch.ones_like(truth, dtype=torch.bool), 255
        )
        self.assertTrue(supervised.all())
        self.assertEqual(target.flatten().tolist(), [KEEP_P3, USE_A2, REJECT_BOTH])

    def test_route_budget_caps_open_pixels(self):
        selector = PairConditionalRejectableSelector(
            route_budget=0.10,
            building_threshold=1.0,
            boundary_threshold=0.0,
        )
        with torch.no_grad():
            selector.selector[-1].bias.copy_(torch.tensor([-4.0, 4.0, -4.0]))
        p3_pred = torch.full((1, 1, 10, 10), 2)
        a2_pred = torch.zeros_like(p3_pred)
        p3 = logits_from_prediction(p3_pred)
        a2 = logits_from_prediction(a2_pred)
        localization = torch.zeros(1, 2, 10, 10)
        localization[:, 0] = 6.0
        selector.set_context(torch.zeros(1, 1, 10, 10), localization)
        _, diagnostics = selector(p3, a2)
        self.assertEqual(int(diagnostics["cer_v2_route_open_mask"].sum()), 10)

    def test_event_pair_balanced_ce_is_single_finite_objective(self):
        logits = torch.zeros(2, 3, 1, 3, requires_grad=True)
        outputs = {
            "cer_v2_selector_logits": logits,
            "cer_v2_pair_id": torch.tensor([[[[8, 8, 8]]], [[[12, 12, 12]]]]),
            "cer_v2_candidate_mask": torch.ones(2, 1, 1, 3),
            "cer_v2_p3_prediction": torch.tensor([[[[2, 2, 2]]], [[[3, 3, 3]]]]),
            "cer_v2_a2_prediction": torch.zeros(2, 1, 1, 3, dtype=torch.long),
        }
        damage = torch.tensor([[[2, 0, 1]], [[3, 0, 2]]])
        values = event_pair_balanced_selector_loss(
            outputs, damage, 255, ["event_a_1", "event_b_1"]
        )
        loss = values["extra_supervised_loss"]
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(values["cer_v2_event_count"]), 2.0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_event_name_is_not_a_model_feature(self):
        self.assertEqual(event_name("la_palma-volcano_0042"), "la_palma-volcano")
        selector = PairConditionalRejectableSelector()
        parameter_names = [name for name, _ in selector.named_parameters()]
        self.assertFalse(any("event" in name or "disaster" in name for name in parameter_names))

    def test_event_balanced_sampler_uses_inverse_event_frequency(self):
        class Toy:
            data_list = ["large-event_1", "large-event_2", "small-event_1"]

        sampler = event_balanced_sampler(Toy(), torch.Generator().manual_seed(1))
        weights = sampler.weights.tolist()
        self.assertEqual(weights, [0.5, 0.5, 1.0])

    def test_evaluator_histogram_contract_is_four_classes_with_ignore(self):
        prediction = torch.tensor([[0, 1, 2, 3, 0]])
        target = torch.tensor([[0, 1, 2, 3, 255]])
        histogram = fast_histogram(
            prediction, target, classes=4, ignore_index=255
        )
        self.assertEqual(tuple(histogram.shape), (4, 4))
        self.assertEqual(float(histogram.diag().sum()), 4.0)


if __name__ == "__main__":
    unittest.main()
