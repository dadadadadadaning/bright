import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import (
    SwinLocalSuperpixelAttention,
)
from bda_benchmark.experiments.damageformer_srac_transformer_v1 import common as parent_common

from .model import (
    AnnealedFront,
    _upgrade_partition_without_initialization,
    q_beta_schedule,
)
from .train_front import audit_parent_reference
from .common import digest


class PartitionContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)
        torch.set_num_threads(2)

    def test_beta_one_is_exact_parent_including_gradients_and_ids(self):
        parent = SwinLocalSuperpixelAttention(16, 16, stride=4, iterations=3)
        annealed = copy.deepcopy(parent)
        _upgrade_partition_without_initialization(annealed, 1.0)
        feature_parent = torch.randn(2, 16, 8, 8, requires_grad=True)
        feature_annealed = feature_parent.detach().clone().requires_grad_(True)
        building = torch.rand(2, 1, 8, 8)

        expected = parent(feature_parent, building)
        actual = annealed(feature_annealed, building)
        self.assertTrue(torch.equal(actual["ids"], expected["ids"]))
        torch.testing.assert_close(actual["q"], expected["q"], atol=0, rtol=0)
        torch.testing.assert_close(actual["entropy_map"], expected["entropy_map"], atol=0, rtol=0)

        expected_loss = expected["q"].square().mean() + expected["entropy_map"].mean()
        actual_loss = actual["q"].square().mean() + actual["entropy_map"].mean()
        expected_loss.backward()
        actual_loss.backward()
        torch.testing.assert_close(feature_annealed.grad, feature_parent.grad, atol=0, rtol=0)
        expected_parameters = dict(parent.named_parameters())
        actual_parameters = dict(annealed.named_parameters())
        self.assertEqual(expected_parameters.keys(), actual_parameters.keys())
        for name in expected_parameters:
            torch.testing.assert_close(
                actual_parameters[name].grad,
                expected_parameters[name].grad,
                atol=0,
                rtol=0,
            )

    def test_beta_sharpens_every_candidate_distribution(self):
        module = SwinLocalSuperpixelAttention(12, 12, stride=4, iterations=1)
        _upgrade_partition_without_initialization(module, 1.0)
        feature = torch.randn(2, 12, 8, 8)
        building = torch.rand(2, 1, 8, 8)
        at_one = module(feature, building)
        module.set_beta(2.0)
        at_two = module(feature, building)

        torch.testing.assert_close(at_two["q"].sum(1), torch.ones(2, 64), atol=1e-6, rtol=1e-6)
        self.assertTrue((at_two["q"] >= 0).all())
        self.assertGreater(float(at_two["q_top1_mean"]), float(at_one["q_top1_mean"]))
        self.assertLess(float(at_two["q_entropy_valid_mean"]), float(at_one["q_entropy_valid_mean"]))
        self.assertTrue(torch.equal(at_two["ids"], at_one["ids"]))

    def test_schedule_boundaries(self):
        self.assertEqual(q_beta_schedule(0), 1.0)
        self.assertEqual(q_beta_schedule(1000), 1.0)
        self.assertAlmostEqual(q_beta_schedule(5500), 1.5)
        self.assertEqual(q_beta_schedule(10000), 2.0)
        self.assertEqual(q_beta_schedule(50000), 2.0)
        with self.assertRaises(ValueError):
            q_beta_schedule(-1)

    def test_upgrade_has_no_parameters_and_consumes_no_rng(self):
        module = SwinLocalSuperpixelAttention(8, 8, stride=4, iterations=1)
        parameter_count = sum(parameter.numel() for parameter in module.parameters())
        before = torch.random.get_rng_state().clone()
        _upgrade_partition_without_initialization(module, 1.0)
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(sum(parameter.numel() for parameter in module.parameters()), parameter_count)
        self.assertIn("q_beta", dict(module.named_buffers()))

    def test_adapter_roundtrip_preserves_beta_and_output_exactly(self):
        torch.manual_seed(101)
        source = AnnealedFront(
            dim=16,
            decoder_width=16,
            stride=4,
            iterations=1,
            variant="q_anneal",
            q_beta_start=1.0,
            q_beta_end=2.0,
            q_beta_hold=2,
            q_beta_ramp_end=6,
        ).eval()
        source.set_step(4)
        pre = torch.randn(1, 3, 64, 64)
        post = torch.randn_like(pre)
        with torch.no_grad():
            expected = source(pre, post)
        state = {name: value.detach().clone() for name, value in source.adapter_state_dict().items()}

        torch.manual_seed(101)
        target = AnnealedFront(
            dim=16,
            decoder_width=16,
            stride=4,
            iterations=1,
            variant="q_anneal",
            q_beta_start=1.0,
            q_beta_end=2.0,
            q_beta_hold=2,
            q_beta_ramp_end=6,
        ).eval()
        target.load_adapter_state_dict(state)
        self.assertEqual(target.current_beta, 1.5)
        with torch.no_grad():
            actual = target(pre, post)
        torch.testing.assert_close(actual["q"], expected["q"], atol=0, rtol=0)
        torch.testing.assert_close(actual["prob"], expected["prob"], atol=0, rtol=0)
        torch.testing.assert_close(actual["adapted_content"], expected["adapted_content"], atol=0, rtol=0)

    def test_front_config_serializes_schedule_and_uses_same_final_q(self):
        model = AnnealedFront(
            dim=16,
            decoder_width=16,
            stride=4,
            iterations=1,
            variant="control",
            q_beta_start=1.0,
            q_beta_end=1.0,
        ).eval()
        self.assertEqual(model.config["variant"], "control")
        self.assertEqual(model.config["q_beta_end"], 1.0)
        with torch.no_grad():
            result = model(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
        torch.testing.assert_close(result["q"].sum(1), torch.ones_like(result["q"][:, 0]), atol=1e-6, rtol=1e-6)
        self.assertEqual(float(result["current_beta"]), 1.0)
        self.assertGreaterEqual(float(result["q_soft_occupied_mean"]), float(result["q_hard_occupied_mean"]))
        self.assertTrue(torch.isfinite(result["adapter_delta_norm"]))
        self.assertTrue(torch.isfinite(result["adapter_base_norm"]))

    def test_live_parent_manifest_is_audited_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a1, train_list, val_list = root / "a1.pt", root / "train_set.txt", root / "val_set.txt"
            a1.write_bytes(b"a1")
            train_list.write_text("train")
            val_list.write_text("val")
            base_config = {
                "dim": 16,
                "decoder_width": 16,
                "stride": 4,
                "iterations": 1,
                "temperature": 0.2,
                "spatial_weight": 0.1,
                "building_weight": 0.5,
                "region_context": False,
            }
            arguments = {
                "steps": 50000,
                "eval_every": 1000,
                "batch": 2,
                "accum": 8,
                "seed": 3407,
                "lr": 1e-4,
                "amp": "bf16",
            }
            manifest = {
                "format": parent_common.FRONT_FORMAT,
                "source_hash": parent_common.source_hash(),
                "a1_sha256": digest(a1),
                "train_sha256": digest(train_list),
                "val_sha256": digest(val_list),
                "config": base_config,
                "arguments": arguments,
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            args = SimpleNamespace(
                a1_checkpoint=str(a1),
                train_list=str(train_list),
                val_list=str(val_list),
                **arguments,
            )
            model = SimpleNamespace(
                config={
                    **base_config,
                    "variant": "q_anneal",
                    "q_beta_start": 1.0,
                    "q_beta_end": 2.0,
                    "q_beta_hold": 1000,
                    "q_beta_ramp_end": 10000,
                }
            )
            report = audit_parent_reference(root, args, model)
            self.assertTrue(report["passed"])
            self.assertFalse(report["weights_reused"])
            manifest["arguments"]["seed"] = 1
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                audit_parent_reference(root, args, model)


if __name__ == "__main__":
    unittest.main()
