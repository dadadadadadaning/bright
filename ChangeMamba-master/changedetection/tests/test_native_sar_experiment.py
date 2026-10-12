import unittest

import torch
import torch.nn as nn

from changedetection.experiments.native_sar.model import (
    freeze_except_native_adapter,
    native_adapter_parameters,
)


class NativeSARExperimentTests(unittest.TestCase):
    def test_adapter_exactly_reconstructs_old_pseudo_rgb_input(self):
        mean, std = 41.5, 12.25
        weight, bias = native_adapter_parameters(mean, std)
        adapter = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        with torch.no_grad():
            adapter.weight.copy_(weight)
            adapter.bias.copy_(bias)
        raw = torch.rand(2, 1, 13, 17) * 100.0
        native = (raw - mean) / std
        observed = adapter(native)
        rgb_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        rgb_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
        expected = (raw.repeat(1, 3, 1, 1) - rgb_mean) / rgb_std
        self.assertTrue(torch.allclose(observed, expected, atol=2e-6, rtol=2e-6))

    def test_only_native_adapter_is_trainable(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Conv2d(3, 4, 1)
                self.sar_adapter = nn.Conv2d(1, 3, 3, padding=1)

        model = Tiny()
        report = freeze_except_native_adapter(model)
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertEqual(trainable, ["sar_adapter.weight", "sar_adapter.bias"])
        self.assertEqual(report["trainable"], 30)

    def test_adapter_rejects_nonpositive_sar_std(self):
        with self.assertRaises(ValueError):
            native_adapter_parameters(0.0, 0.0)


if __name__ == "__main__":
    unittest.main()

