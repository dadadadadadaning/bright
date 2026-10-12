# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest
from pathlib import Path

import torch

from model.DamageFormer_S8_MICB_DSMIL import MICBDSMILHead
from model.DamageFormer_S8_P6_SoftCritical import SoftCriticalMICBDSMILHead
from script.standard_ML.train_S8_P6_softcritical import configure_p6_trainable
from script.standard_ML.train_S8_P6C1_occurrence import (
    configure_p6c1_trainable,
)
from script.standard_ML.train_S8_P4_native_sar_micb import (
    mil_supervision_scale,
)


class _TinyP6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        holder = torch.nn.Module()
        holder.features = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Conv2d(1, 4, 1))]
        )
        encoder = torch.nn.Module()
        encoder.swin_transformer = holder
        self.encoder_2 = encoder
        self.fusion_layer_3 = torch.nn.Conv2d(8, 4, 1)
        self.clf_2 = torch.nn.Conv2d(4, 4, 1)
        self.other = torch.nn.Conv2d(4, 4, 1)


class S8P6SoftCriticalTests(unittest.TestCase):
    def test_soft_critical_is_convex_and_differentiable(self):
        head = SoftCriticalMICBDSMILHead(
            in_channels=16,
            token_dim=16,
            topk=2,
            max_tokens=8,
            occurrence_temperature=0.5,
        )
        tokens = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            requires_grad=True,
        )
        logits = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
        critical, evidence = head._aggregate_occurrence_critical(tokens, logits)
        self.assertEqual(tuple(critical.shape), (2,))
        self.assertEqual(tuple(evidence.shape), (1,))
        self.assertTrue(torch.isfinite(critical).all())
        self.assertTrue(torch.isfinite(evidence).all())
        (critical.sum() + evidence.sum()).backward()
        self.assertTrue(bool((logits.grad.abs() > 0).sum() > 1))
        self.assertTrue(bool((tokens.grad.abs().sum(dim=1) > 0).all()))

    def test_identical_logits_produce_mean_and_size_normalised_evidence(self):
        head = SoftCriticalMICBDSMILHead(
            in_channels=16,
            token_dim=16,
            topk=2,
            max_tokens=8,
            occurrence_temperature=0.5,
        )
        tokens = torch.randn(5, 16)
        logits = torch.full((5,), 1.25)
        critical, evidence = head._aggregate_occurrence_critical(tokens, logits)
        torch.testing.assert_close(critical, tokens.mean(dim=0))
        torch.testing.assert_close(evidence, torch.tensor([1.25]))

    def test_soft_head_keeps_p3_parameter_namespace(self):
        hard = MICBDSMILHead(in_channels=16, token_dim=16, topk=2, max_tokens=8)
        soft = SoftCriticalMICBDSMILHead(
            in_channels=16,
            token_dim=16,
            topk=2,
            max_tokens=8,
            occurrence_temperature=0.5,
        )
        self.assertEqual(tuple(hard.state_dict()), tuple(soft.state_dict()))
        soft.load_state_dict(hard.state_dict(), strict=True)

    def test_runner_is_four_thousand_step_validation_only_contract(self):
        runner = Path(__file__).parents[1] / "run" / "run_S8_P6_SoftCritical_5090.sh"
        text = runner.read_text(encoding="utf-8")
        block = text.split("  pilot4000)", 1)[1].split("    ;;", 1)[0]
        self.assertIn("train_run 4000 500 1000", block)
        self.assertNotIn("test_set.txt", text)
        self.assertIn("VARIANT=a4", text)

    def test_default_p3_critical_remains_hard_argmax(self):
        head = MICBDSMILHead(in_channels=16, token_dim=16, topk=2, max_tokens=8)
        tokens = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        logits = torch.tensor([-1.0, 2.0, 0.5])
        critical, evidence = head._aggregate_occurrence_critical(tokens, logits)
        torch.testing.assert_close(critical, tokens[1])
        torch.testing.assert_close(evidence, torch.tensor([2.0]))

    def test_phase_two_locks_a4_input_and_opens_only_late_core(self):
        model = _TinyP6()
        configure_p6_trainable(model, input_only=False)
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(any(name.startswith("fusion_layer_3.") for name in trainable))
        self.assertTrue(any(name.startswith("clf_2.") for name in trainable))
        self.assertFalse(any(name.startswith("encoder_2.") for name in trainable))
        self.assertFalse(any(name.startswith("other.") for name in trainable))

    def test_p6c1_opens_fusion_only_and_freezes_pixel_classifier(self):
        model = _TinyP6()
        configure_p6c1_trainable(model, input_only=False)
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(any(name.startswith("fusion_layer_3.") for name in trainable))
        self.assertFalse(any(name.startswith("clf_2.") for name in trainable))
        self.assertFalse(any(name.startswith("encoder_2.") for name in trainable))

    def test_p6c1_runner_is_resume_only_occurrence_contract(self):
        runner = (
            Path(__file__).parents[1]
            / "run"
            / "run_S8_P6C1_Occurrence_5090.sh"
        )
        text = runner.read_text(encoding="utf-8")
        self.assertIn("--expected_resume_step 1000", text)
        self.assertIn("--lambda_mil 0.0", text)
        self.assertIn("--lambda_severity_aux 0.0", text)
        self.assertIn("--lambda_exist_aux 0.05", text)
        self.assertIn("--mil_ramp_steps 500", text)
        self.assertNotIn("test_set.txt", text)

    def test_occurrence_supervision_ramps_without_a_step_1000_jump(self):
        self.assertEqual(mil_supervision_scale(999, 1000, 500), 0.0)
        self.assertAlmostEqual(mil_supervision_scale(1000, 1000, 500), 0.002)
        self.assertAlmostEqual(mil_supervision_scale(1249, 1000, 500), 0.5)
        self.assertEqual(mil_supervision_scale(1499, 1000, 500), 1.0)
        self.assertEqual(mil_supervision_scale(1000, 1000, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
