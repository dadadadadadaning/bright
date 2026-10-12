"""Exercise the real inherited A1 forward with small offline feature fixtures.

Only the expensive feature extractor is replaced. The S8 forward that applies
the building prior, calls clf_2, and interpolates the logits is production code.
"""
from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from bda_benchmark.model.DamageFormer_S8_MICB_DSMIL import DamageFormer_S8_MICB_DSMIL
from bda_benchmark.experiments.damageformer_srac_joint_context_v1.common import FrozenFrontCapture
from .native import NativeA1LogitsCapture, native_replay


class SmallFeatureA1(DamageFormer_S8_MICB_DSMIL):
    """Keep DamageFormer_S8_MICB_DSMIL.forward without constructing a backbone."""

    def __init__(self):
        nn.Module.__init__(self)
        self.clf_1 = nn.Conv2d(256, 2, 1)
        self.fusion_layer_3 = nn.Conv2d(512, 256, 1)
        self.clf_2 = nn.Conv2d(256, 4, 1)
        self.detach_building_prior = True
        self.roi_alpha = 0.1
        self.mil_residual_scale = 0.0

    def _encode_and_fuse(self, pre_data, post_data):
        rgb = F.adaptive_avg_pool2d(pre_data.mean(1, keepdim=True), (2, 3))
        sar = F.adaptive_avg_pool2d(post_data.mean(1, keepdim=True), (2, 3))
        return rgb.repeat(1, 256, 1, 1), sar.repeat(1, 256, 1, 1)


class NativeForwardContracts(unittest.TestCase):
    def test_actual_s8_forward_calls_damage_classifier_once_with_or_without_aux(self):
        self.assertIs(SmallFeatureA1.forward, DamageFormer_S8_MICB_DSMIL.forward)
        base = SmallFeatureA1().eval()
        pre = torch.linspace(-1, 1, 3 * 5 * 7).reshape(1, 3, 5, 7)
        post = pre[:, :1].flip(-1)
        outputs = []
        handle = base.clf_2.register_forward_hook(lambda _m, _i, value: outputs.append(value.detach()))
        try:
            for return_aux in (False, True):
                with self.subTest(return_aux=return_aux), torch.no_grad():
                    outputs.clear()
                    result = base(pre, post, return_aux=return_aux)
                    self.assertEqual(len(outputs), 1)
                    expected = F.interpolate(outputs[0], post.shape[-2:],
                                             mode="bilinear", align_corners=False)
                    self.assertTrue(torch.equal(expected, result[1]))
                    if return_aux:
                        self.assertEqual(result[2]["bag_count"], 0)
                        self.assertEqual(result[2]["mil_residual_scale"], 0.0)
        finally:
            handle.remove()

    def test_real_a1_forward_and_frozen_capture_preserve_native_replay_across_images(self):
        class Front(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = SmallFeatureA1()

            def forward(self, pre, post):
                _loc, old_pixel, auxiliary = self.base(pre, post, return_aux=True)
                return {"old_pixel": old_pixel, "fused_roi": auxiliary["fused_roi"]}

        front = Front()
        feature_capture = FrozenFrontCapture(front)
        native_capture = NativeA1LogitsCapture(front)
        pre = torch.linspace(-1, 1, 3 * 5 * 7).reshape(1, 3, 5, 7)
        post = pre[:, :1].flip(-1)
        try:
            for offset in (0.0, 0.1):
                native_capture.clear()
                result, feature = feature_capture(pre + offset, post)
                low_logits = native_capture.take()
                self.assertTrue(torch.equal(feature, result["fused_roi"]))
                self.assertFalse(feature.requires_grad)
                replay = native_replay(low_logits, result["old_pixel"], (5, 7))
                self.assertTrue(replay["passed"])
                self.assertEqual(replay["max_abs"], 0.0)
                # Later diagnostic calls are outside the native forward window.
                front.base.clf_2(feature)
                front.base.clf_2(feature + 1)
        finally:
            feature_capture.close()
            native_capture.close()
        self.assertEqual(len(front.base.clf_2._forward_hooks), 0)
        self.assertEqual(len(front.base.clf_2._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
