"""CPU-only tests: no checkpoint, real dataset, CUDA or training job required."""
import copy
import unittest

import torch
from torch import nn

from bda_benchmark.experiments.damageformer_srac_transformer_v1.model import (
    StructureTokenAdapter,
    SuperpixelEmbedding,
    SwinLocalSuperpixelAttention,
    project_region_probabilities,
    soft_assignment_boundary,
)

from .freeze_probe import (
    FrozenSource,
    FusionReadoutArm,
    arm_checkpoint,
    bn_buffers,
    buffers_unchanged,
    clip_gradients,
    independent_parameter_contract,
    objective,
    reproduction_gate,
)


class SyntheticBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_rgb = nn.Conv2d(3, 4, 1)
        self.encoder_sar = nn.Conv2d(3, 4, 1)
        self.building_classifier = nn.Conv2d(4, 2, 1)
        self.fusion_layer_3 = nn.Sequential(
            nn.Conv2d(8, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4), nn.GELU()
        )
        self.pixel_classifier = nn.Conv2d(4, 4, 1)
        self.forward_calls = 0

    def forward(self, pre, post, return_aux=False):
        self.forward_calls += 1
        rgb, sar = self.encoder_rgb(pre), self.encoder_sar(post)
        loc = self.building_classifier(rgb)
        building = loc.softmax(1)[:, 1:2]
        roi = 0.1 + 0.9 * building
        fused = self.fusion_layer_3(torch.cat((rgb, sar), 1)) * roi
        pixel = self.pixel_classifier(fused)
        if not return_aux:
            return loc, pixel
        return loc, pixel, dict(rgb_feature=rgb, sar_feature=sar, p_building=building,
                                roi_weight=roi, fused_roi=fused)


class SyntheticFront(nn.Module):
    """Real parent partition/readout primitives, inexpensive substitute A1."""

    def __init__(self):
        super().__init__()
        self.base = SyntheticBase()
        self.decoder = nn.Conv2d(4, 8, 1)
        self.partition = SwinLocalSuperpixelAttention(8, 8, stride=2, iterations=1)
        self.adapter = StructureTokenAdapter(4, 8)
        self.embedding = SuperpixelEmbedding(4, 8, 8)
        self.region_classifier = nn.Linear(8, 4)

    def forward(self, pre, post):
        loc, old_pixel, aux = self.base(pre, post, return_aux=True)
        q = self.partition(self.decoder(aux["rgb_feature"]), aux["p_building"])
        boundary = soft_assignment_boundary(q["q"], q["ids"], q["shape"])
        adapted, _ = self.adapter(aux["fused_roi"], aux["p_building"], boundary, q["entropy_map"])
        tokens, _, _, _ = self.embedding(adapted, aux["fused_roi"], q["q"], q["ids"], q["shape"], q["count"])
        logits = self.region_classifier(tokens)
        probability = project_region_probabilities(logits, q["q"], q["ids"], q["shape"], pre.shape[-2:])
        return dict(**q, prob=probability, old_pixel=old_pixel, loc=loc, soft_boundary=boundary)


def identical_state(module, before):
    return all(torch.equal(value, before[name]) for name, value in module.state_dict().items())


class FreezeProbeContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(41)
        self.model = SyntheticFront().eval()
        self.source = FrozenSource(self.model)
        self.arms = {"F": FusionReadoutArm(self.model, False), "U": FusionReadoutArm(self.model, True)}
        self.pre, self.post = torch.randn(2, 3, 6, 6), torch.randn(2, 3, 6, 6)
        self.target = torch.arange(72).reshape(2, 6, 6) % 4

    def test_identical_initial_copies_and_one_frozen_forward(self):
        independent_parameter_contract(self.model, self.arms)
        before = self.model.base.forward_calls
        fixed = self.source(self.pre, self.post)
        self.assertEqual(self.model.base.forward_calls - before, 1)
        f, u = (self.arms[name](fixed) for name in ("F", "U"))
        self.assertIs(f["q"], fixed.q)
        self.assertIs(u["q"], fixed.q)
        torch.testing.assert_close(f["prob"], u["prob"], rtol=0, atol=0)
        torch.testing.assert_close(f["prob"], fixed.normal_probability, rtol=0, atol=0)
        self.assertFalse(fixed.q.requires_grad)
        self.assertFalse(fixed.building.requires_grad)
        self.assertFalse(fixed.roi_weight.requires_grad)
        self.assertFalse(any(parameter.requires_grad for parameter in self.model.parameters()))

    def test_only_u_fusion_changes_both_readouts_learn_q_and_bn_stay_fixed(self):
        fixed = self.source(self.pre, self.post)
        source_before = copy.deepcopy(self.model.state_dict())
        q_before = fixed.q.clone()
        before = {name: copy.deepcopy(arm.state_dict()) for name, arm in self.arms.items()}
        buffers = {name: bn_buffers(arm.fusion_layer_3) for name, arm in self.arms.items()}
        optimizers = {name: torch.optim.AdamW([p for p in arm.parameters() if p.requires_grad],
                                             lr=1e-3, weight_decay=0)
                      for name, arm in self.arms.items()}
        for name, arm in self.arms.items():
            arm.train()
            self.assertFalse(arm.fusion_layer_3.training)
            self.assertFalse(arm.fusion_layer_3[1].training)
            optimizers[name].zero_grad(set_to_none=True)

        objective(self.arms["F"](fixed)["prob"], self.target).backward()
        self.assertTrue(all(p.grad is None for p in self.arms["U"].parameters()))
        f_grad = {name: p.grad.clone() for name, p in self.arms["F"].named_parameters() if p.grad is not None}
        self.assertTrue(all(p.grad is None for p in self.arms["F"].fusion_layer_3.parameters()))
        objective(self.arms["U"](fixed)["prob"], self.target).backward()
        self.assertTrue(all(torch.equal(dict(self.arms["F"].named_parameters())[name].grad, value)
                            for name, value in f_grad.items()))
        self.assertGreater(float(self.arms["U"].fusion_layer_3[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.arms["U"].fusion_layer_3[1].weight.grad.abs().sum()), 0)
        for name, arm in self.arms.items():
            for part in ("adapter", "embedding", "region_classifier"):
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in getattr(arm, part).parameters()
                                       if p.grad is not None), 0)
            optimizers[name].step()
            self.assertTrue(buffers_unchanged(arm.fusion_layer_3, buffers[name]))
            for part in ("adapter", "embedding", "region_classifier"):
                self.assertTrue(any(not torch.equal(value, before[name][part + "." + key])
                                    for key, value in getattr(arm, part).state_dict().items()))
        self.assertTrue(identical_state(self.model, source_before))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertTrue(torch.equal(q_before, fixed.q))
        self.assertTrue(torch.equal(q_before, self.source(self.pre, self.post).q))
        self.assertTrue(torch.equal(self.arms["F"].fusion_layer_3[0].weight,
                                    before["F"]["fusion_layer_3.0.weight"]))
        self.assertFalse(torch.equal(self.arms["U"].fusion_layer_3[0].weight,
                                     before["U"]["fusion_layer_3.0.weight"]))
        self.assertFalse(torch.equal(self.arms["U"].fusion_layer_3[1].weight,
                                     before["U"]["fusion_layer_3.1.weight"]))

    def test_inference_context_produces_trainable_consumer_inputs(self):
        with torch.inference_mode():
            fixed = self.source(self.pre, self.post)
        self.assertFalse(torch.is_inference(fixed.rgb_feature))
        self.assertFalse(torch.is_inference(fixed.q))
        objective(self.arms["U"](fixed)["prob"], self.target).backward()
        self.assertIsNotNone(self.arms["U"].fusion_layer_3[0].weight.grad)

    def test_checkpoint_contains_only_copied_modules(self):
        manifest = dict(source_checkpoint_sha256="front-hash", a1_sha256="a1-hash",
                        source_checkpoint="front.pth", a1_checkpoint="a1.pth", smoke=True)
        for name, arm in self.arms.items():
            state = arm_checkpoint(arm, name, 1, 0.5, manifest)
            expected = {"adapter", "embedding", "region_classifier"}
            if name == "U":
                expected.add("fusion_layer_3")
            self.assertEqual(set(state["modules"]), expected)
            self.assertEqual(state["a1_sha256"], "a1-hash")
            self.assertNotIn("base", state)

    def test_extra_fusion_norm_cannot_shrink_u_readout_gradients(self):
        for arm in self.arms.values():
            for parameter in arm.readout_parameters():
                parameter.grad = torch.full_like(parameter, 2.0)
        for parameter in self.arms["U"].fusion_layer_3.parameters():
            parameter.grad = torch.full_like(parameter, 1e6)
        f_norm = clip_gradients(self.arms["F"])
        u_norm = clip_gradients(self.arms["U"])
        self.assertEqual(f_norm["readout"], u_norm["readout"])
        self.assertIsNone(f_norm["fusion"])
        self.assertGreater(u_norm["fusion"], u_norm["readout"])
        for f_parameter, u_parameter in zip(self.arms["F"].readout_parameters(),
                                            self.arms["U"].readout_parameters()):
            torch.testing.assert_close(f_parameter.grad, u_parameter.grad, atol=0, rtol=0)

    def test_reproduction_gate_never_substitutes_partial_val_or_looser_tolerance(self):
        initial = dict(image_count=450, metrics={"normal": {"valid_miou": 0.6},
                                                 "a1_pixel": {"valid_miou": 0.61}})
        payload = dict(valid_miou=0.6, load_report={"expected_valid_miou": 0.61})
        self.assertTrue(reproduction_gate(initial, payload, False)["reproduction_pass"])
        payload["valid_miou"] = 0.60002
        self.assertFalse(reproduction_gate(initial, payload, False)["reproduction_pass"])
        payload["valid_miou"] = 0.6
        payload["load_report"]["expected_valid_miou"] = 0.61002
        a1_fail = reproduction_gate(initial, payload, False)
        self.assertTrue(a1_fail["normal_reproduction_pass"])
        self.assertFalse(a1_fail["a1_reproduction_pass"])
        self.assertFalse(a1_fail["reproduction_pass"])
        payload["load_report"]["expected_valid_miou"] = 0.61
        initial["image_count"] = 2
        with self.assertRaisesRegex(RuntimeError, "450"):
            reproduction_gate(initial, payload, False)
        smoke = reproduction_gate(initial, payload, True)
        self.assertIsNone(smoke["reproduction_pass"])
        self.assertIsNone(smoke["normal_reproduction_pass"])
        self.assertIsNone(smoke["a1_reproduction_pass"])
        self.assertTrue(smoke["smoke_only_unverified"])


if __name__ == "__main__":
    unittest.main()
