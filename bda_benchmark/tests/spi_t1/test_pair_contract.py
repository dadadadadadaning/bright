import unittest

import torch

from bda_benchmark.script.standard_ML.spi_t1.audit import (
    PAIR_FIELDS,
    finalize_dual_ground_truth_report,
    validate_pair_metadata,
)
from bda_benchmark.script.standard_ML.spi_stage0.core import new_variant_accumulator
from bda_benchmark.script.standard_ML.spi_t1.core import VARIANTS


def checkpoint(variant, *, seed=3407, steps=3, lambda_cb=0.0):
    args = {field: f"same-{field}" for field in PAIR_FIELDS}
    args["seed"] = seed
    args["optimizer_steps"] = steps
    return {
        "protocol": "SPI_T0_T1_FIXED_STEP_PAIRED_v1",
        "variant": variant,
        "test_accessed": False,
        "base_checkpoint_sha256": "abc",
        "step": steps,
        "seed": seed,
        "effective_lambda_cb": lambda_cb,
        "args": args,
    }


class PairContractTests(unittest.TestCase):
    def test_valid_pair_passes(self):
        manifest = validate_pair_metadata(
            checkpoint(VARIANTS[0], lambda_cb=0.0),
            checkpoint(VARIANTS[1], lambda_cb=0.1),
        )
        self.assertEqual(manifest["seed"], 3407)
        self.assertEqual(manifest["optimizer_steps"], 3)

    def test_different_seed_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "paired-control mismatch"):
            validate_pair_metadata(
                checkpoint(VARIANTS[0], seed=3407, lambda_cb=0.0),
                checkpoint(VARIANTS[1], seed=4407, lambda_cb=0.1),
            )

    def test_test_access_is_rejected(self):
        t1 = checkpoint(VARIANTS[1], lambda_cb=0.1)
        t1["test_accessed"] = True
        with self.assertRaisesRegex(RuntimeError, "test isolation"):
            validate_pair_metadata(checkpoint(VARIANTS[0]), t1)

    def test_dual_ground_truth_keeps_both_semantic_protocols(self):
        accumulator = new_variant_accumulator()
        accumulator["semantic_histogram"] = torch.eye(4, dtype=torch.float64)
        semantic_histogram = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        report = finalize_dual_ground_truth_report(
            accumulator, semantic_histogram
        )
        self.assertEqual(
            report["instance_raster_semantic"]["valid_miou"], 1.0
        )
        self.assertNotEqual(
            report["semantic"]["valid_miou"],
            report["instance_raster_semantic"]["valid_miou"],
        )


if __name__ == "__main__":
    unittest.main()
