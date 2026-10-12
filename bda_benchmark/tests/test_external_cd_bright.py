import json
import tempfile
import unittest
from pathlib import Path

import torch

from bda_benchmark.model.DamageFormer_RGBSAR_ExternalCD import (
    PlainDualBRIGHT,
    damage_to_localization_logits,
)
from bda_benchmark.dataset.make_data_loader_heterogeneous_rgbsar import (
    HeterogeneousRGBSARDataset,
)


class ExternalCDBRIGHTTests(unittest.TestCase):
    def test_official_adapter_replaces_binary_task_head(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "model"
            / "DamageFormer_RGBSAR_ExternalCD.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "self.fusion_decoder.make_pred_bcd = official.make_prediction",
            source,
        )
        self.assertIn("out_channels=4", source)

    def test_external_variants_use_native_sar_dataset_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            stats = Path(directory) / "stats.json"
            stats.write_text(
                json.dumps(
                    {
                        "normalization": {
                            "p01": 0.0,
                            "p99": 255.0,
                            "mean": 64.0,
                            "std": 32.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            for variant in ("x1", "x2"):
                dataset = HeterogeneousRGBSARDataset(
                    directory,
                    ["contract_only"],
                    crop_size=32,
                    split="val",
                    variant=variant,
                    sar_stats_path=str(stats),
                )
                self.assertEqual(dataset.variant, variant)
                self.assertIsNotNone(dataset.sar_stats)

    def test_localization_is_derived_from_four_class_logits(self):
        logits = torch.randn(2, 4, 5, 7)
        location = damage_to_localization_logits(logits)
        self.assertEqual(tuple(location.shape), (2, 2, 5, 7))
        damage_probability = logits.softmax(1)
        location_probability = location.softmax(1)
        self.assertTrue(
            torch.allclose(
                location_probability[:, 1],
                damage_probability[:, 1:].sum(1),
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_plain_dual_accepts_native_one_channel_sar(self):
        model = PlainDualBRIGHT(pretrained_rgb=False).eval()
        with torch.no_grad():
            output = model(
                torch.randn(1, 3, 128, 128),
                torch.randn(1, 1, 128, 128),
                return_dict=True,
            )
        self.assertEqual(tuple(output["output_dam"].shape), (1, 4, 128, 128))
        self.assertEqual(tuple(output["output_loc"].shape), (1, 2, 128, 128))


if __name__ == "__main__":
    unittest.main()
