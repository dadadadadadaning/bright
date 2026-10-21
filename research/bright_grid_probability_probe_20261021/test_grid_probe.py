import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import grid_probe as probe


def reference_validator(p, target, path):
    if (not isinstance(p, torch.Tensor) or not p.is_floating_point()
            or p.shape != (1, 4, *target.shape[1:])
            or p.device != target.device or not bool(torch.isfinite(p).all())
            or bool(((p < 0) | (p > 1)).any())):
        raise ValueError("Invalid full-image probabilities: " + path)
    tolerance = max(1e-5, 4 * torch.finfo(p.dtype).eps)
    total = p.to(torch.float64).sum(1)
    if not torch.allclose(total, torch.ones_like(total), atol=tolerance, rtol=0):
        raise ValueError("Unnormalized full-image probabilities: " + path)


class ProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.target = torch.zeros(1, 2, 2, dtype=torch.long)
        self.p = torch.full((1, 4, 2, 2), .25)

    def test_valid_distribution_unchanged(self):
        before = self.p.clone()
        state = torch.get_rng_state().clone()
        result = probe.probability_details(torch, self.p, self.target)
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["sum_max_abs_error"], 0.)
        self.assertTrue(torch.equal(before, self.p))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_one_ulp_overshoot_is_reported_not_fixed(self):
        self.p.zero_()
        self.p[:, 0] = 1.
        self.p[0, 0, 0, 0] = 1. + torch.finfo(torch.float32).eps
        result = probe.probability_details(torch, self.p, self.target)
        self.assertEqual(result["reasons"], ["outside_0_1"])
        self.assertEqual(result["above_one_count"], 1)
        self.assertEqual(result["above_one_excess"], torch.finfo(torch.float32).eps)
        with self.assertRaisesRegex(ValueError, "Invalid"):
            reference_validator(self.p, self.target, "grid_probq")

    def test_nonfinite_values_are_separated_and_json_safe(self):
        self.p[0, 0, 0, 0] = float("nan")
        self.p[0, 1, 0, 0] = float("inf")
        self.p[0, 2, 0, 0] = float("-inf")
        result = probe.probability_details(torch, self.p, self.target)
        self.assertEqual(result["nonfinite_count"], 3)
        self.assertEqual(result["nan_count"], 1)
        self.assertEqual(result["positive_inf_count"], 1)
        self.assertEqual(result["negative_inf_count"], 1)
        json.dumps(result, allow_nan=False)

    def test_type_shape_and_normalization_are_distinct(self):
        self.assertEqual(probe.probability_details(torch, None, self.target)["reasons"], ["not_tensor"])
        self.assertEqual(probe.probability_details(torch, self.p.long(), self.target)["reasons"], ["not_floating"])
        self.assertIn("shape_mismatch", probe.probability_details(torch, self.p[0], self.target)["reasons"])
        result = probe.probability_details(torch, self.p * .5, self.target)
        self.assertEqual(result["reasons"], ["unnormalized"])

    def test_device_mismatch_detected_without_target_copy(self):
        target = torch.empty(1, 2, 2, device="meta", dtype=torch.long)
        result = probe.probability_details(torch, self.p, target)
        self.assertEqual(result["reasons"], ["device_mismatch"])


class FakeRuntime:
    def __init__(self):
        self.args = SimpleNamespace(seed=3407)
        self.model = torch.nn.Linear(1, 1)
        self.seen = []

    def evaluation_batches(self, names, seed):
        return names

    def prepare(self, name):
        self.seen.append(name)
        return SimpleNamespace(image=name, target=torch.zeros(1, 1, 1, dtype=torch.long),
                               common={"base_probability_full": torch.full((1, 4, 1, 1), .25)})


class LoopTests(unittest.TestCase):
    def test_prefix_stop_identifies_actual_image_and_preserves_mode(self):
        runtime = FakeRuntime()
        before = {k: v.clone() for k, v in runtime.model.state_dict().items()}
        def prepare(example, arms):
            if example.image == "image98":
                example.common["base_probability_full"][0, 0, 0, 0] = -0.1
            return {a: {} for a in arms}
        evaluator = SimpleNamespace(_validate_probability=reference_validator,
                                    _capture_modes=lambda r, h: [(r.model, r.model.training)])
        rows = []
        result = probe.inspect_images(torch, runtime, SimpleNamespace(prepare=prepare), evaluator,
                                      SimpleNamespace(_assert_example=lambda x: None),
                                      ["image%d" % i for i in range(1, 100)], ("e", "d"), rows.append)
        self.assertEqual(result["images_checked"], 98)
        self.assertEqual(result["first_invalid"]["image"], "image98")
        self.assertEqual(result["first_invalid"]["before_support_probability"]["reasons"], [])
        self.assertIn("outside_0_1", result["first_invalid"]["probability"]["reasons"])
        self.assertNotIn("image99", runtime.seen)
        self.assertTrue(runtime.model.training)
        for k, v in runtime.model.state_dict().items():
            self.assertTrue(torch.equal(before[k], v))

    def test_order_mismatch_fails_and_restores_mode(self):
        runtime = FakeRuntime()
        runtime.evaluation_batches = lambda names, seed: ["wrong"]
        evaluator = SimpleNamespace(_capture_modes=lambda r, h: [(r.model, r.model.training)])
        with self.assertRaisesRegex(ValueError, "order"):
            probe.inspect_images(torch, runtime, None, evaluator, None, ["expected"], ("d",), lambda x: None)
        self.assertTrue(runtime.model.training)

    def test_passing_frontend_not_reported_as_formal_evaluation(self):
        runtime = FakeRuntime()
        evaluator = SimpleNamespace(_validate_probability=reference_validator,
                                    _capture_modes=lambda r, h: [(r.model, r.model.training)])
        result = probe.inspect_images(torch, runtime, SimpleNamespace(prepare=lambda e, a: {x: {} for x in a}),
                                      evaluator, SimpleNamespace(_assert_example=lambda x: None),
                                      ["candidate"], ("d",), lambda x: None)
        self.assertEqual(result["phase"], "frozen_forward_passed")
        self.assertNotIn("complete", result)


class EvidenceTests(unittest.TestCase):
    def test_candidate_prefix_and_checkpoint_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoints").mkdir()
            (root / "val_step_001500").mkdir()
            cp = root / "checkpoints/checkpoint_001500.pt"
            cp.write_bytes(b"fixture, not a real checkpoint")
            names = ["image%d" % i for i in range(450)]
            probe.write_json(root / "manifest.json", dict(format="BRIGHT_SLOT_MEMORY_RESUME_V1", val_names=names))
            probe.write_json(root / "summary.json", dict(phase="failed", failure_phase="evaluate_val_1500",
                             last_checkpoint=dict(step=1500, roundtrip_verified=True, sha256=probe.sha256(cp))))
            probe.write_json(cp.with_suffix(".pt.json"), dict(sha256=probe.sha256(cp)))
            journal = root / "val_step_001500/images.jsonl"
            journal.write_text("".join(json.dumps(dict(image=n)) + "\n" for n in names[:97]))
            _, evidence, prefix = probe.locate(root, names[97])
            self.assertEqual(evidence["completed_old_images"], 97)
            self.assertEqual(len(prefix), 98)
            with self.assertRaisesRegex(ValueError, "next image"):
                probe.locate(root, names[98])
            cp.write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                probe.locate(root, names[97])


if __name__ == "__main__":
    unittest.main()
