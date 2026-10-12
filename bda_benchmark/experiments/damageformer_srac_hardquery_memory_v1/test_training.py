"""Synthetic CPU integration checks, never a claim of BRIGHT reproduction."""
from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from . import runtime
from .prepare_queries import pool_base_probability
from .reader import GatedMemoryCorrection, drop_support, retrieve_evidence
from .train_reader import assert_initial_identity
from bda_benchmark.experiments.damageformer_srac_fourstage_v2.losses import region_nll
from bda_benchmark.experiments.damageformer_srac_qmetric_v1.metric import RetrievalProjector
from bda_benchmark.experiments.damageformer_srac_joint123_v1.joint_runtime import (
    module_fingerprint, upsample_probability,
)
from bda_benchmark.script.standard_ML.train_S5_S8_GPU import lovasz_softmax


class TinyFrozenCapture(nn.Module):
    """Small fixed front with nine normalized sparse routing slots."""

    def __init__(self):
        super().__init__()
        self.feature = nn.Conv2d(3, 256, 1)
        self.embedding = nn.Linear(256, 128)
        self.a1 = nn.Conv2d(256, 4, 1)
        self.register_buffer("q", torch.randn(1, 9, 16).softmax(1))
        self.register_buffer("ids", torch.arange(9 * 16).reshape(9, 16).remainder(6))
        self.requires_grad_(False).eval()

    def forward(self, pre, post):
        feature = self.feature((pre + post) / 2)
        tokens = self.embedding(feature.flatten(2).transpose(1, 2)[:, :6])
        old = F.interpolate(self.a1(feature), size=(8, 8), mode="bilinear", align_corners=False)
        return dict(tokens=tokens, q=self.q, ids=self.ids, shape=(4, 4),
                    old_pixel=old, prob=old.softmax(1), valid_regions=torch.ones(1, 6, dtype=torch.bool)), feature


class RecordingGuard:
    def __init__(self):
        self.names = []

    def set_image(self, name):
        self.names.append(name)


class TrainingIntegration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        torch.set_num_threads(2)
        self.capture = TinyFrozenCapture()
        self.decoders = {key: nn.Conv2d(256, 4, 1).eval().requires_grad_(False)
                         for key in ("j1", "j2", "j3")}
        self.projectors = {key: RetrievalProjector(128).eval().requires_grad_(False)
                           for key in ("bank", "hard")}
        with torch.no_grad():
            # A distinct metric so the 2x2 fixture does not silently exercise
            # only one set of retrieval scores.
            self.projectors["hard"].linear.weight[:64] *= 1.4
        count = 24
        self.bank = dict(tokens=torch.randn(count, 128), distribution=torch.eye(4)[torch.arange(count) % 4],
                         events=torch.arange(count) % 4, images=torch.arange(count),
                         geometry=torch.zeros(count, 5), disasters=torch.arange(count) % 8,
                         event_names=["train_a", "train_b", "train_c", "train_d"])
        self.template = GatedMemoryCorrection(dim=128, hidden=16)
        self.heads = {arm: copy.deepcopy(self.template) for arm in runtime.TRAIN_ARMS}
        self.pre, self.post = torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)
        self.target = torch.arange(64).reshape(1, 8, 8).remainder(4)
        self.target[:, 0, 0] = 255

    @staticmethod
    def decode(decoder, arm, feature, result, cosine_support, metric_support):
        # The source heavyweight backbone/old Reader is outside this new
        # package's unit scope. Its replacement is a frozen, deterministic
        # linear head with the same output contract; all new paths are real.
        if arm == "j2":
            assert cosine_support is not None
        if arm == "j3":
            assert metric_support is not None
        return {"logits": decoder(feature)}

    def test_six_arms_twenty_steps_real_sameq_losses_frozen_identity_and_reload(self):
        self.assertEqual(len(runtime.TRAIN_ARMS), 6)
        fixed = {"front": self.capture, **self.decoders, **self.projectors}
        fingerprints = {key: module_fingerprint(module) for key, module in fixed.items()}
        bank_before = {key: value.clone() for key, value in self.bank.items() if torch.is_tensor(value)}
        with torch.no_grad():
            result, feature = self.capture(self.pre, self.post)
            base_logits = self.decoders["j1"](feature)
            base_probability = upsample_probability(base_logits, self.target.shape[-2:])
            base_region, mass = pool_base_probability(base_probability, result)
        torch.testing.assert_close(base_region.sum(-1), torch.ones(6), atol=1e-6, rtol=0)
        self.assertTrue((mass > 0).all())
        evidence = {key: retrieve_evidence(result["tokens"][0], self.bank, 0, 0, 8, projector)
                    for key, projector in self.projectors.items()}
        optimizers = {arm: torch.optim.AdamW(head.parameters(), lr=0.005) for arm, head in self.heads.items()}
        seen_delta_gradient = {arm: False for arm in self.heads}
        seen_gate_gradient = {arm: False for arm in self.heads if arm.endswith("_gated")}
        for arm, head in self.heads.items():
            key, mode = arm.rsplit("_", 1)
            item = runtime.query_only_evidence(evidence["bank"]) if key == "query" else evidence[key]
            delta, _, _ = head(result["tokens"][0], base_region, item, mode)
            self.assertTrue(torch.equal(runtime.corrected_logits(base_logits, delta, result), base_logits))
        for step in range(20):
            dropout_patterns = {}
            for arm, head in self.heads.items():
                key, mode = arm.rsplit("_", 1)
                item = drop_support(evidence["bank" if key == "query" else key], .1,
                                    torch.Generator().manual_seed(700 + step), row_probability=.1)
                if key == "query":
                    item = runtime.query_only_evidence(item)
                    self.assertEqual(torch.count_nonzero(item["posterior"]).item(), 0)
                    self.assertEqual(torch.count_nonzero(item["quality"]).item(), 0)
                dropout_patterns[arm] = item["valid"]
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                delta, _, _ = head(result["tokens"][0], base_region, item, mode)
                probability = upsample_probability(runtime.corrected_logits(base_logits, delta, result), (8, 8))
                loss = region_nll(probability, self.target) + lovasz_softmax(probability, self.target, classes="present", ignore=255)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                seen_delta_gradient[arm] |= bool(head.delta_head.weight.grad.abs().sum() > 0)
                if arm in seen_gate_gradient:
                    seen_gate_gradient[arm] |= bool(head.gate_head.weight.grad.abs().sum() > 0)
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1, error_if_nonfinite=True)
                self.assertTrue(torch.isfinite(norm))
                optimizer.step()
            for pattern in dropout_patterns.values():
                self.assertTrue(torch.equal(pattern, dropout_patterns["bank_ungated"]))
        self.assertTrue(all(seen_delta_gradient.values()))
        self.assertTrue(all(seen_gate_gradient.values()))
        for key, module in fixed.items():
            self.assertEqual(module_fingerprint(module), fingerprints[key])
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))
        for key, tensor in bank_before.items():
            self.assertTrue(torch.equal(tensor, self.bank[key]))
        for arm, head in self.heads.items():
            key, mode = arm.rsplit("_", 1)
            item = runtime.query_only_evidence(evidence["bank"]) if key == "query" else evidence[key]
            delta, _, _ = head(result["tokens"][0], base_region, item, mode)
            self.assertGreater(delta.abs().sum().item(), 0)
            delta_off, gate_off, _ = head(result["tokens"][0], base_region, item, mode, force_off=True)
            logits_off = runtime.corrected_logits(base_logits, delta_off, result)
            self.assertTrue(torch.equal(logits_off, base_logits))
            self.assertTrue(torch.equal(upsample_probability(logits_off, (8, 8)), base_probability))
            self.assertEqual(torch.count_nonzero(gate_off).item(), 0)
            stream = io.BytesIO()
            torch.save(head.state_dict(), stream)
            stream.seek(0)
            restored = GatedMemoryCorrection(dim=128, hidden=16)
            restored.load_state_dict(torch.load(stream, weights_only=True))
            reloaded, _, _ = restored(result["tokens"][0], base_region, item, mode)
            torch.testing.assert_close(reloaded, delta, rtol=0, atol=0)

    def test_two_image_evaluate_all_paths_and_fixed_cohorts(self):
        bundle = dict(bank=self.bank, device=torch.device("cpu"), metric=self.projectors["bank"],
                      decoders=self.decoders, topk=8, reference={})
        names = ["validation_event_001", "validation_event_002"]
        data = [(self.pre, self.post, (self.target > 0).long(), self.target, [name]) for name in names]
        guard = RecordingGuard()
        with tempfile.TemporaryDirectory() as directory, patch.object(runtime, "decoder_forward", side_effect=self.decode):
            log = Path(directory) / "images.jsonl"
            report = runtime.evaluate(bundle, self.capture, data, guard, projectors=self.projectors,
                                      heads=self.heads, image_log=log, smoke=True)
            rows = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(guard.names, names)
        self.assertEqual(report["image_count"], 2)
        self.assertEqual(len(rows), 2)
        self.assertIsNone(report["reproduction"]["passed"])
        self.assertEqual(report["reproduction"]["tolerance"], 1e-5)
        self.assertFalse(report["test_accessed"])
        assert_initial_identity(report)
        expected = {"a1_pixel", "direct_region", "j1", "j2", "j3", *runtime.TRAIN_ARMS,
                    "label_cosine", "label_old_metric", "label_bank", "label_hard"}
        self.assertEqual(len(expected), 15)
        self.assertEqual(set(report["metrics"]), expected)
        for metric in report["metrics"].values():
            self.assertEqual(metric["count"], 126)
        for arm in runtime.TRAIN_ARMS:
            self.assertEqual(report["max_abs_probability_to_fixed_j1"][arm], 0)
            self.assertEqual(report["gate_totals"][arm]["count"], 12)
            self.assertIn("damaged_j1_wrong", report["difficult_cohorts"][arm])
            self.assertIn("validation_event", report["per_event"][arm])
        for metric in ("cosine", "old_metric", "bank", "hard"):
            self.assertIn(metric, report["retrieval"])
            self.assertIn("damaged_j1_wrong", report["difficult_retrieval"][metric])

    def test_query_only_control_removes_label_score_quality_but_matches_availability(self):
        result, feature = self.capture(self.pre, self.post)
        base_probability = upsample_probability(self.decoders["j1"](feature), (8, 8))
        base_region, _ = pool_base_probability(base_probability, result)
        source = retrieve_evidence(result["tokens"][0], self.bank, 0, 0)
        dropped = drop_support(source, .4, torch.Generator().manual_seed(311), row_probability=.2)
        control = runtime.query_only_evidence(dropped)
        self.assertTrue(torch.equal(control["valid"], dropped["valid"]))
        self.assertTrue(torch.equal(control["supported"], dropped["supported"]))
        self.assertEqual(torch.count_nonzero(control["posterior"]).item(), 0)
        self.assertEqual(torch.count_nonzero(control["quality"]).item(), 0)
        self.assertGreater(dropped["posterior"].abs().sum().item(), 0)
        changed = dict(dropped, posterior=torch.rand_like(dropped["posterior"]),
                       quality=torch.rand_like(dropped["quality"]), scores=torch.randn_like(dropped["scores"]))
        stripped_changed = runtime.query_only_evidence(changed)
        # Nonzero arbitrary weights exercise the actual input dependency;
        # zero-initialized corrections alone could hide a leaked feature.
        head = GatedMemoryCorrection(dim=128, hidden=16)
        with torch.no_grad():
            head.delta_head.weight.normal_(std=.1)
            head.gate_head.weight.normal_(std=.1)
        delta, gate, _ = head(result["tokens"][0], base_region, control)
        altered_delta, altered_gate, _ = head(result["tokens"][0], base_region, stripped_changed)
        torch.testing.assert_close(delta, altered_delta, rtol=0, atol=0)
        torch.testing.assert_close(gate, altered_gate, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
