import unittest
import numpy as np
import torch

from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import soft_pool_unpool
from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import SRACTransformerFront, project_region_probabilities
from .core import MODES, capture, gt_mixing_diagnostics, interventions, probability_read, sharpen, split_ids
from .diagnostics import DiagnosticAccumulator, paired_summary


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)

    def test_sharpen_normalizes_and_preserves_zero_supports(self):
        q = torch.tensor([[[.8, .0], [.2, 1.0]]])
        saved = q.clone()
        sharp = sharpen(q)
        self.assertGreater(float(sharp[0, 0, 0]), .8)
        torch.testing.assert_close(sharp.sum(1), torch.ones(1, 2))
        self.assertEqual(float(sharp[0, 0, 1]), 0)
        torch.testing.assert_close(q, saved, rtol=0, atol=0)

    def test_probability_read_is_not_double_softmax_and_empty_slots_safe(self):
        q = torch.zeros(1, 9, 4)
        q[:, 4] = 1
        ids = torch.zeros(9, 4, dtype=torch.long)
        token = torch.tensor([[[.9, .08, .01, .01], [0, 0, 0, 0]]])
        actual = probability_read(token, q, ids, (2, 2), (2, 2))
        torch.testing.assert_close(actual[0, :, 0, 0], token[0, 0])
        expected = project_region_probabilities(token.clamp_min(1e-30).log(), q, ids, (2, 2), (2, 2))
        torch.testing.assert_close(actual, expected)

    def test_predicted_split_removes_predicted_mixing_not_GT_mixing(self):
        q = torch.zeros(1, 9, 4); q[:, 4] = 1
        ids = torch.zeros(9, 4, dtype=torch.long)
        b = torch.tensor([[[[.1, .1], [.9, .9]]]])
        new_ids, count = split_ids(ids, b, 1)
        self.assertEqual(count, 2)
        binary = (b >= .5).float()
        pools = soft_pool_unpool(binary, q, new_ids, count)
        torch.testing.assert_close(pools["tokens"][0, :, 0], torch.tensor([0., 1.]))
        # Incorrect B: every predicted group still mixes GT building/background.
        truth = torch.tensor([[[0, 2], [0, 2]]])
        mixing, _ = gt_mixing_diagnostics(truth, q, new_ids, count, (2, 2))
        self.assertGreater(mixing["per_class"]["Damaged"]["binary_impurity_weighted_sum"], 0)

    def test_full_real_model_capture_interventions_keep_A1_Q_buffers(self):
        model = SRACTransformerFront(dim=16, decoder_width=16, stride=4, iterations=1).eval()
        model.requires_grad_(False)
        buffers = {k: v.clone() for k, v in model.named_buffers()}
        pre, post = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            native, aux = capture(model, pre, post)
            q = native["q"].clone()
            outputs, b, routes = interventions(model, native, aux, (64, 64))
            token, _, _, _ = model.embedding(native["adapted_content"], aux["fused_roi"], q,
                                             native["ids"], native["shape"], native["count"])
            torch.testing.assert_close(model.region_classifier(token), native["region_logits"])
        self.assertEqual(set(outputs), set(MODES))
        torch.testing.assert_close(outputs["region"], native["prob"], rtol=0, atol=0)
        torch.testing.assert_close(native["q"], q, rtol=0, atol=0)
        self.assertEqual(routes["predicted_split"]["allocated"], 2*routes["native"]["allocated"])
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)

    def test_paired_transition_orientation_and_ignore_exclusion(self):
        joint = np.zeros((4, 4, 4), dtype=np.int64)
        joint[2, 2, 1] = 7; joint[2, 1, 2] = 3
        row = paired_summary(joint)
        self.assertEqual(row["damaged_to_intact_harm"], 7)
        self.assertEqual(row["damaged_from_intact_rescue"], 3)
        self.assertEqual(row["net"], -4)
        y = torch.full((1, 8, 8), 2, dtype=torch.long); y[:, 0] = 255
        prediction = {key: torch.full_like(y, 2) for key in MODES}
        prediction["region"][:] = 1
        accumulator = DiagnosticAccumulator(MODES)
        accumulator.update(y, prediction, torch.full((1, 1, 2, 2), .1),
                           torch.full((1, 1, 2, 2), .9), torch.zeros(1, 1, 2, 2),
                           "haiti-earthquake_0001", {}, {})
        result = accumulator.finalize()
        self.assertEqual(result["metrics"]["a1"]["count"], 56)
        self.assertEqual(result["groups"]["roi_false_negative"]["a1__region"]["damaged_to_intact_harm"], 56)


if __name__ == "__main__":
    unittest.main()
