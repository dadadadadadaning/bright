import unittest

import torch

from bda_benchmark.model.DamageFormer_RGBSAR_A2_DCSM import (
    A2_DCSM_VARIANTS,
    DCSMFeaturePredictor,
    DamageFormerRGBSARA2DCSM,
    FixedScharrMagnitude,
    StructuralSARAdapter,
    class_balanced_region_weights,
    damage_conditioned_target,
    masked_cosine_prediction_loss,
    unchanged_mask_from_damage,
)
from bda_benchmark.model.DamageFormer_RGBSAR_H6_SDCR import (
    DamageFormerRGBSARH6SDCR,
)
from bda_benchmark.script.standard_ML.train_A2_DCSM import (
    configure_freeze_policy,
    event_balanced_sampler,
    validate_dataset_interface,
)


class A2DCSMTests(unittest.TestCase):
    def test_variants_are_controlled(self):
        self.assertEqual(tuple(A2_DCSM_VARIANTS), ("m0", "m1", "m2", "m3"))
        self.assertFalse(A2_DCSM_VARIANTS["m0"].feature_pretraining)
        self.assertTrue(A2_DCSM_VARIANTS["m1"].feature_pretraining)
        self.assertFalse(A2_DCSM_VARIANTS["m1"].structural_adapter)
        self.assertTrue(A2_DCSM_VARIANTS["m2"].structural_adapter)
        self.assertFalse(A2_DCSM_VARIANTS["m2"].migrate_deep_neck)
        self.assertTrue(A2_DCSM_VARIANTS["m3"].migrate_deep_neck)

    def test_scharr_is_one_channel_bounded_and_finite(self):
        module = FixedScharrMagnitude()
        value = module(torch.randn(2, 1, 16, 16))
        self.assertEqual(tuple(value.shape), (2, 1, 16, 16))
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue((value >= 0).all())

    def test_structural_adapter_is_safe_but_receives_gradient(self):
        module = StructuralSARAdapter(hidden_channels=4)
        value = torch.randn(2, 1, 16, 16)
        output = module(value)
        self.assertTrue(torch.equal(output, value))
        output.square().mean().backward()
        self.assertIsNotNone(module.transition.grad)
        self.assertGreater(abs(float(module.transition.grad)), 0.0)

    def test_rgb_anchor_is_used_only_on_unchanged_labels(self):
        damage = torch.tensor([[[0, 1], [2, 3]]])
        unchanged = unchanged_mask_from_damage(damage)
        sar = torch.zeros(1, 2, 2, 2)
        rgb = torch.ones_like(sar)
        target = damage_conditioned_target(sar, rgb, unchanged, 0.25)
        self.assertTrue(torch.equal(target[0, :, 0, 0], torch.full((2,), 0.25)))
        self.assertTrue(torch.equal(target[0, :, 0, 1], torch.full((2,), 0.25)))
        self.assertTrue(torch.equal(target[0, :, 1, 0], torch.zeros(2)))
        self.assertTrue(torch.equal(target[0, :, 1, 1], torch.zeros(2)))

    def test_region_weights_prioritize_damage_without_a_new_loss(self):
        damage = torch.tensor([[[0, 1], [2, 3]]])
        weights = class_balanced_region_weights(damage, (2, 2))
        expected = torch.tensor([[[0.25, 1.0], [2.0, 2.0]]])
        self.assertTrue(torch.equal(weights, expected))

    def test_single_prediction_loss_is_finite(self):
        predictor = DCSMFeaturePredictor(4)
        value = torch.randn(2, 8, 8, 8)
        prediction = predictor(value)
        target = torch.randn_like(prediction)
        mask = torch.ones(2, 1, 32, 32)
        weights = torch.ones(2, 8, 8)
        loss = masked_cosine_prediction_loss(prediction, target, mask, weights)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_m0_has_exact_h6_a2_parameter_contract(self):
        a2 = DamageFormerRGBSARH6SDCR(variant="a2", pretrained_rgb=False)
        m0 = DamageFormerRGBSARA2DCSM(variant="m0", pretrained_rgb=False)
        self.assertEqual(set(a2.state_dict()), set(m0.state_dict()))
        for key, value in a2.state_dict().items():
            self.assertEqual(tuple(value.shape), tuple(m0.state_dict()[key].shape))

    def test_event_sampler_equalizes_total_event_mass(self):
        class Dataset:
            data_list = ["large-event_0", "large-event_1", "small-event_0"]

        sampler = event_balanced_sampler(Dataset(), torch.Generator().manual_seed(1))
        weights = sampler.weights.tolist()
        self.assertAlmostEqual(weights[0] + weights[1], weights[2], places=7)

    def test_dataset_declares_damage_focused_crop_interface(self):
        self.assertIsNone(validate_dataset_interface())

    def test_freeze_policy_protects_migrated_shallow_path(self):
        model = DamageFormerRGBSARA2DCSM(variant="m2", pretrained_rgb=False)
        configure_freeze_policy(model)
        parameters = dict(model.named_parameters())
        self.assertFalse(parameters["structural_adapter.transition"].requires_grad)
        shallow = [
            parameter.requires_grad
            for name, parameter in parameters.items()
            if name.startswith("sar_encoder.stem")
        ]
        self.assertTrue(shallow and not any(shallow))
        deep = [
            parameter.requires_grad
            for name, parameter in parameters.items()
            if name.startswith("sar_encoder.stages.3")
        ]
        self.assertTrue(deep and all(deep))


if __name__ == "__main__":
    unittest.main()
