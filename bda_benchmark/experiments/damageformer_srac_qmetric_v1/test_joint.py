import copy
import inspect
import unittest

import torch
from torch import nn

from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import candidate_map

from bda_benchmark.experiments.damageformer_srac_joint_context_v1.model import (
    JointContextDecoder,
)

from .joint_runtime import (
    ARMS,
    _new_retrieval_diagnostic,
    _finalize_retrieval_diagnostic,
    _update_retrieval_diagnostic,
    bank_fingerprint,
    decoder_forward,
    evaluate_joint,
    module_fingerprint,
    retrieve_pair,
)
from .metric import RetrievalProjector, select_ids


class JointContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(109)
        torch.set_num_threads(2)

    @staticmethod
    def association():
        ids = torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]])
        q = torch.tensor([[[0.75, 0.50, 0.25, 0.00], [0.25, 0.50, 0.75, 1.00]]])
        return q, ids

    @staticmethod
    def bank(dim=16):
        tokens = torch.randn(8, dim)
        return {
            "tokens": tokens,
            "distribution": torch.eye(4).repeat(2, 1),
            "events": torch.arange(8),
            "images": torch.arange(8) + 20,
            "regions": torch.arange(8),
            "geometry": torch.rand(8, 5),
            "disasters": torch.arange(8) % 3,
            "event_names": [str(index) for index in range(8)],
        }

    def test_all_arms_start_at_exact_a1_identity(self):
        a1 = nn.Conv2d(256, 4, 1)
        template = JointContextDecoder(a1, dim=16, num_events=8, slots=4).eval()
        decoders = {arm: copy.deepcopy(template) for arm in ARMS}
        pixel = torch.randn(1, 256, 2, 2)
        tokens = torch.randn(1, 3, 16)
        q, ids = self.association()
        result = {"tokens": tokens, "q": q, "ids": ids, "shape": (2, 2)}
        bank = self.bank()
        projector = RetrievalProjector(16).eval()
        _, cosine_support, _, metric_support = retrieve_pair(
            tokens[0], bank, event_id=-1, image_id=-1, topk=4, metric_projector=projector
        )
        expected = a1(pixel)
        for arm, decoder in decoders.items():
            output = decoder_forward(
                decoder, arm, pixel, result, cosine_support, metric_support
            )["logits"]
            torch.testing.assert_close(output, expected, atol=0, rtol=0)

    def test_j2_j3_share_decoder_path_and_only_ids_change(self):
        a1 = nn.Conv2d(256, 4, 1)
        decoder = JointContextDecoder(a1, dim=16, num_events=8, slots=4).eval()
        with torch.no_grad():
            decoder.classifier.context.weight.normal_(std=0.1)
        pixel = torch.randn(1, 256, 2, 2)
        tokens = torch.randn(1, 3, 16)
        q, ids = self.association()
        result = {"tokens": tokens, "q": q, "ids": ids, "shape": (2, 2)}
        bank = self.bank()
        projector = RetrievalProjector(16).eval()
        _, support, _, _ = retrieve_pair(
            tokens[0], bank, event_id=-1, image_id=-1, topk=4, metric_projector=projector
        )
        j2 = decoder_forward(decoder, "j2", pixel, result, support, support)["logits"]
        j3 = decoder_forward(decoder, "j3", pixel, result, support, support)["logits"]
        torch.testing.assert_close(j3, j2, atol=0, rtol=0)

    def test_metric_changes_only_selected_rows_not_source_bank(self):
        bank = {
            "tokens": torch.tensor([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]]),
            "distribution": torch.eye(4)[:3],
            "events": torch.tensor([1, 2, 3]),
            "images": torch.tensor([11, 12, 13]),
            "regions": torch.arange(3),
            "geometry": torch.zeros(3, 5),
            "disasters": torch.zeros(3, dtype=torch.long),
            "event_names": ["a", "b", "c"],
        }
        before = bank_fingerprint(bank)
        projector = RetrievalProjector(2).eval()
        with torch.no_grad():
            projector.linear.weight.copy_(torch.diag(torch.tensor([0.1, 10.0])))
        query = torch.tensor([[1.0, 0.2]])
        cosine_ids, cosine_support, metric_ids, metric_support = retrieve_pair(
            query, bank, event_id=-1, image_id=-1, topk=1, metric_projector=projector
        )
        self.assertNotEqual(int(cosine_ids[0, 0]), int(metric_ids[0, 0]))
        torch.testing.assert_close(cosine_support[0][0, 0], bank["tokens"][cosine_ids[0, 0]])
        torch.testing.assert_close(metric_support[0][0, 0], bank["tokens"][metric_ids[0, 0]])
        self.assertEqual(bank_fingerprint(bank), before)

    def test_retrieval_api_cannot_receive_query_gt_or_class(self):
        parameters = inspect.signature(select_ids).parameters
        self.assertNotIn("target", parameters)
        self.assertNotIn("gt", parameters)
        self.assertNotIn("query_class", parameters)
        self.assertNotIn("label", parameters)

    def test_zero_weight_query_is_excluded_only_from_diagnostics(self):
        bank = self.bank(dim=4)
        selected = torch.tensor([[0, 1], [2, 3], [4, 5]])
        distribution = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        weight = torch.tensor([0.0, 2.0, 3.0])
        valid = torch.ones(3, dtype=torch.bool)
        diagnostic = _new_retrieval_diagnostic()
        bank_class = bank["distribution"].argmax(-1)
        query_class = distribution.argmax(-1)
        usable = (weight > 0) & valid
        _update_retrieval_diagnostic(
            diagnostic, selected, bank_class, query_class, usable
        )
        self.assertEqual(int(diagnostic["query_count"].sum()), 2)
        self.assertEqual(int(diagnostic["query_count"][1]), 0)
        final = _finalize_retrieval_diagnostic(diagnostic)
        self.assertIsNone(final["recall_at_k"][1])
        self.assertIsNone(final["supported_query_coverage"][1])

    def test_deepcopied_arms_are_equal_and_independent(self):
        template = JointContextDecoder(nn.Conv2d(256, 4, 1), dim=8, num_events=2, slots=4)
        models = {arm: copy.deepcopy(template) for arm in ARMS}
        reference = dict(models["j0"].named_parameters())
        for arm in ARMS[1:]:
            candidate = dict(models[arm].named_parameters())
            self.assertEqual(reference.keys(), candidate.keys())
            for name in reference:
                torch.testing.assert_close(reference[name], candidate[name])
                self.assertNotEqual(reference[name].data_ptr(), candidate[name].data_ptr())

    def test_fingerprints_detect_no_read_only_mutation(self):
        projector = RetrievalProjector(8).eval().requires_grad_(False)
        bank = self.bank(dim=8)
        module_before = module_fingerprint(projector)
        bank_before = bank_fingerprint(bank)
        query = torch.randn(3, 8)
        retrieve_pair(query, bank, -1, -1, 4, projector)
        self.assertEqual(module_fingerprint(projector), module_before)
        self.assertEqual(bank_fingerprint(bank), bank_before)

    def test_full_joint_evaluation_callback_runs_on_cpu(self):
        a1 = nn.Conv2d(256, 4, 1)
        template = JointContextDecoder(a1, dim=8, num_events=2, slots=2).eval()
        decoders = {arm: copy.deepcopy(template) for arm in ARMS}
        pixel = torch.randn(1, 256, 2, 2)
        ids, _valid, grid_h, grid_w = candidate_map(2, 2, 1, torch.device("cpu"))
        q = torch.zeros(1, 9, 4)
        q[:, 4] = 1.0
        tokens = torch.randn(1, grid_h * grid_w, 8)
        old_pixel = torch.nn.functional.interpolate(
            a1(pixel), size=(4, 4), mode="bilinear", align_corners=False
        )
        result = {
            "tokens": tokens,
            "q": q,
            "ids": ids,
            "shape": (2, 2),
            "valid_regions": torch.ones(1, grid_h * grid_w, dtype=torch.bool),
            "old_pixel": old_pixel,
            "prob": old_pixel.softmax(1),
        }

        class Capture:
            def __call__(self, _pre, _post):
                return result, pixel

        bank = self.bank(dim=8)
        bank["events"].fill_(1)
        bank["event_names"] = ["foo", "other"]
        batch = (
            torch.randn(1, 3, 4, 4),
            torch.randn(1, 3, 4, 4),
            torch.zeros(1, 4, 4, dtype=torch.long),
            torch.tensor([[[0, 0, 1, 1], [0, 2, 2, 1], [3, 3, 2, 1], [3, 0, 1, 2]]]),
            ["foo.png"],
        )
        output = evaluate_joint(
            Capture(),
            decoders,
            [batch],
            torch.device("cpu"),
            bank,
            RetrievalProjector(8).eval(),
            topk=2,
        )
        self.assertEqual(output["image_count"], 1)
        self.assertEqual(set(output["metrics"]), {"a1_pixel", "direct_region", *ARMS})
        self.assertIn("damaged_j3_vs_j2", output)
        self.assertIn("supported_query_coverage", output["retrieval"]["metric"])


if __name__ == "__main__":
    unittest.main()
