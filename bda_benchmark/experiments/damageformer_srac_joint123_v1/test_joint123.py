import copy
import inspect
import unittest

import torch
from torch import nn

from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import (
    candidate_map,
)
from bda_benchmark.experiments.damageformer_srac_joint_context_v1.model import (
    JointContextDecoder,
)
from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import (
    RetrievalProjector,
    select_ids,
)

from .joint_runtime import (
    ARMS,
    JOINT_FORMAT,
    bank_fingerprint,
    decoder_forward,
    evaluate_joint,
    module_fingerprint,
    retrieve_pair,
)
from .train import probe_train_names, validate_formal_front_payload


class Joint123Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(211)
        torch.set_num_threads(2)

    @staticmethod
    def association():
        ids = torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]])
        q = torch.tensor([[[0.75, 0.50, 0.25, 0.00], [0.25, 0.50, 0.75, 1.00]]])
        return q, ids

    @staticmethod
    def bank(dim=16):
        return {
            "tokens": torch.randn(8, dim),
            "distribution": torch.eye(4).repeat(2, 1),
            "events": torch.arange(8),
            "images": torch.arange(8) + 20,
            "regions": torch.arange(8),
            "geometry": torch.rand(8, 5),
            "disasters": torch.arange(8) % 3,
            "event_names": [str(index) for index in range(8)],
        }

    def test_exactly_j1_j2_j3_and_no_j0(self):
        self.assertEqual(ARMS, ("j1", "j2", "j3"))
        self.assertNotIn("j0", ARMS)
        self.assertEqual(JOINT_FORMAT, "SRAC_JOINT123_V1")

    def test_all_three_heads_start_at_exact_a1_identity(self):
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

    def test_three_heads_are_equal_but_parameter_independent(self):
        template = JointContextDecoder(nn.Conv2d(256, 4, 1), dim=8, num_events=2, slots=4)
        models = {arm: copy.deepcopy(template) for arm in ARMS}
        reference = dict(models["j1"].named_parameters())
        for arm in ("j2", "j3"):
            candidate = dict(models[arm].named_parameters())
            self.assertEqual(reference.keys(), candidate.keys())
            for name in reference:
                torch.testing.assert_close(reference[name], candidate[name], atol=0, rtol=0)
                self.assertNotEqual(reference[name].data_ptr(), candidate[name].data_ptr())

    def test_j2_j3_use_same_decoder_branch_and_only_support_rows_can_differ(self):
        decoder = JointContextDecoder(
            nn.Conv2d(256, 4, 1), dim=16, num_events=8, slots=4
        ).eval()
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

    def test_retrieval_interface_cannot_receive_query_gt(self):
        parameters = inspect.signature(select_ids).parameters
        for forbidden in ("target", "gt", "query_class", "label"):
            self.assertNotIn(forbidden, parameters)

    def test_probe_selection_uses_cache_event_metadata(self):
        records = [
            {"name": "a0", "event": 3},
            {"name": "a1", "event": 3},
            {"name": "b0", "event": 7},
            {"name": "c0", "event": 8},
            {"name": "d0", "event": 9},
            {"name": "e0", "event": 10},
        ]
        self.assertEqual(probe_train_names(records), ["a0", "b0", "c0", "d0"])

    def test_q_annealed_or_incomplete_front_is_rejected(self):
        valid = {
            "format": "SRAC_TRANSFORMER_FRONT_V1",
            "run_complete": True,
            "completed_steps": 5000,
            "arguments": {"steps": 5000},
            "smoke": False,
            "test_accessed": False,
            "train_sha256": "train",
            "val_sha256": "val",
        }
        validate_formal_front_payload(valid, "train", "val")
        q_front = dict(valid, format="SRAC_QMETRIC_FRONT_V1")
        with self.assertRaisesRegex(ValueError, "rejects Q-annealed"):
            validate_formal_front_payload(q_front, "train", "val")
        incomplete = dict(valid, run_complete=False)
        with self.assertRaisesRegex(ValueError, "completed formal"):
            validate_formal_front_payload(incomplete, "train", "val")

    def test_full_evaluation_cpu_and_frozen_inputs_unchanged(self):
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
            def __init__(self):
                self.front = nn.Linear(2, 2).eval().requires_grad_(False)

            def __call__(self, _pre, _post):
                return result, pixel

        capture = Capture()
        projector = RetrievalProjector(8).eval().requires_grad_(False)
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
        frozen_before = (
            module_fingerprint(capture.front),
            module_fingerprint(projector),
            bank_fingerprint(bank),
        )
        output = evaluate_joint(
            capture,
            decoders,
            [batch],
            torch.device("cpu"),
            bank,
            projector,
            topk=2,
        )
        frozen_after = (
            module_fingerprint(capture.front),
            module_fingerprint(projector),
            bank_fingerprint(bank),
        )
        self.assertEqual(frozen_after, frozen_before)
        self.assertEqual(output["image_count"], 1)
        self.assertEqual(
            set(output["metrics"]), {"a1_pixel", "direct_region", "j1", "j2", "j3"}
        )
        self.assertNotIn("j0", output["metrics"])
        self.assertEqual(set(output["max_abs_probability_to_a1"]), set(ARMS))
        self.assertIn("damaged_j3_vs_j2", output)
        self.assertIn("supported_query_coverage", output["retrieval"]["metric"])


if __name__ == "__main__":
    unittest.main()
