# -*- coding: utf-8 -*-
import unittest

import torch

from bda_benchmark.experiments.salt_cde_validation_v1.audit_model import shift_zero
from bda_benchmark.model.DamageFormer_SALT_CDE import (
    counterfactual_transport_loss,
    make_counterfactual_sar,
    outer_ring,
    propagate_anchor_offsets,
)


class SALTCoreTests(unittest.TestCase):
    def test_outer_ring_excludes_high_confidence_core(self):
        probability = torch.zeros(1, 1, 9, 9)
        probability[:, :, 3:6, 3:6] = 1.0
        ring = outer_ring(probability, 3)
        self.assertEqual(float(ring[0, 0, 4, 4]), 0.0)
        self.assertGreater(float(ring[0, 0, 2, 4]), 0.0)

    def test_anchor_offset_is_propagated_into_building(self):
        offsets = torch.zeros(1, 2, 7, 7)
        anchor = torch.zeros(1, 1, 7, 7)
        anchor[:, :, 1, 1:6] = 1.0
        offsets[:, 0] = 2.0 * anchor[:, 0]
        propagated, support = propagate_anchor_offsets(offsets, anchor, 5)
        self.assertGreater(float(support[0, 0, 2, 3]), 0.0)
        self.assertAlmostEqual(float(propagated[0, 0, 2, 3]), 2.0, places=4)

    def test_zero_shift_has_no_wraparound(self):
        value = torch.arange(20).reshape(1, 1, 4, 5).float()
        shifted = shift_zero(value, dx=2)
        self.assertTrue(torch.equal(shifted[..., :2], torch.zeros_like(shifted[..., :2])))
        self.assertTrue(torch.equal(shifted[..., 2:], value[..., :-2]))

    def test_batch_counterfactual_changes_pairing(self):
        sar = torch.arange(3).reshape(3, 1, 1, 1).float()
        wrong = make_counterfactual_sar(sar, "batch_shuffle")
        self.assertTrue(torch.equal(wrong[:, 0, 0, 0], torch.tensor([2.0, 0.0, 1.0])))

    def test_counterfactual_loss_penalises_wrong_high_gate(self):
        target = torch.ones(1, 3, 3, dtype=torch.long)
        logits = torch.zeros(1, 4, 3, 3)
        correct = {"output_dam": logits, "base_damage_logits": logits.clone()}
        wrong = {"output_dam": logits.clone(), "base_damage_logits": logits.clone()}
        for stage in range(1, 5):
            correct[f"stage{stage}_sar_gate"] = torch.full((1, 1, 3, 3), 0.8)
            wrong[f"stage{stage}_sar_gate"] = torch.full((1, 1, 3, 3), 0.9)
        losses = counterfactual_transport_loss(correct, wrong, target, gate_margin=0.1)
        self.assertGreater(float(losses["counterfactual_gate_rank"]), 0.19)
        self.assertTrue(torch.isfinite(losses["counterfactual_loss"]))

    def test_explicit_gate_supervision_rewards_pair_separation(self):
        target = torch.ones(1, 3, 3, dtype=torch.long)
        logits = torch.zeros(1, 4, 3, 3)

        def payload(gate):
            result = {"output_dam": logits, "base_damage_logits": logits.clone()}
            for stage in range(1, 5):
                result[f"stage{stage}_sar_gate"] = torch.full((1, 1, 3, 3), gate)
            return result

        separated = counterfactual_transport_loss(
            payload(0.8), payload(0.2), target, explicit_gate_bce=True
        )
        confused = counterfactual_transport_loss(
            payload(0.8), payload(0.8), target, explicit_gate_bce=True
        )
        self.assertLess(
            float(separated["counterfactual_gate_bce"]),
            float(confused["counterfactual_gate_bce"]),
        )


if __name__ == "__main__":
    unittest.main()
