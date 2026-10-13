"""CPU-only runner contracts; these do not use BRIGHT images or CUDA."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from common import ARMS, Example, digest
import train


class FakeRuntime:
    instances = []
    fail_final_verify = False
    fail_restore = False
    protocol_tag = "synthetic-bank"

    def __init__(self, args, output):
        self.args = args
        self.output = output
        self.train_names = ["train_a", "train_b", "train_c", "train_d"]
        self.val_names = ["val_a", "val_b"]
        self.device = torch.device("cpu")
        self.protocol_info = {"bank_sha256": self.protocol_tag, "synthetic": True}
        self.guard_report = None
        self.prepare_count = 0
        self.verify_count = 0
        self.__class__.instances.append(self)

    def __enter__(self):
        # Mirror historical bundle loading resetting the E0 seed.
        torch.manual_seed(3407)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.guard_report = {"restored": not self.fail_restore, "restore_conflicts": []}

    def smoke_names(self):
        return self.train_names, self.val_names

    def batches(self, names, train, seed):
        # Deterministic small generator; real Runtime tests cover permutation.
        yield from names

    def prepare(self, image):
        self.prepare_count += 1
        g = torch.Generator().manual_seed(19 + len(image))
        base = torch.softmax(torch.randn(1, 4, 2, 2, generator=g), dim=1)
        common = {
            "pixel_features": torch.randn(1, 256, 2, 2, generator=g),
            "region_f": torch.randn(1, 2, 256, generator=g),
            "region_t": torch.randn(1, 2, 128, generator=g),
            "q": torch.ones(1, 1, 4),
            "ids": torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
            "base_probability_full": base,
        }
        supports = {}
        for index, arm in enumerate(ARMS):
            labels = torch.zeros(1, 2, 2, 4)
            labels[..., (index + 1) % 4] = 1
            supports[arm] = {
                "support_f": torch.randn(1, 2, 2, 256, generator=g),
                "support_t": torch.randn(1, 2, 2, 128, generator=g),
                "support_labels": labels,
                "support_valid": torch.ones(1, 2, 2, dtype=torch.bool),
            }
        return Example(image=image, event="synthetic", target=torch.tensor([[[0, 1], [2, 3]]]),
                       common=common, supports=supports, diagnostics={"synthetic": True})

    def loss(self, probability, target):
        return torch.nn.functional.nll_loss(probability.clamp_min(1e-8).log(), target)

    def verify(self):
        self.verify_count += 1
        if self.fail_final_verify and self.verify_count >= 2:
            raise RuntimeError("Synthetic final integrity failure")
        return {"passed": True, "synthetic": True}


class TrainerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        FakeRuntime.instances = []

    def tearDown(self):
        self.temp.cleanup()

    def args(self, name="train", **extra):
        fields = dict(repo="synthetic", v3_root="synthetic", protocol=None,
                      output=str(self.base / name), seed=3407, workers=0,
                      device="cpu", topk=64, chunk_size=2, mode="smoke", steps=2,
                      eval_every=20, lr=1e-4, weight_decay=5e-3)
        fields.update(extra)
        return SimpleNamespace(**fields)

    def run_small(self, args=None, runtime=FakeRuntime):
        args = args or self.args()
        with redirect_stdout(io.StringIO()):
            result = train.run_training(args, runtime_factory=runtime)
        return args, result

    def test_cpu_smoke_shared_preparation_and_matched_updates(self):
        args, result = self.run_small()
        self.assertTrue(result["execution_complete"])
        self.assertTrue(result["complete"])
        self.assertFalse(result["full_validation_evidence"])
        self.assertTrue(result["smoke"])
        self.assertFalse(result["test_accessed"])
        self.assertTrue(result["training_complete"])
        self.assertEqual(result["updates_per_arm"], {a: 2 for a in ARMS})
        # prepare once per image, not once per arm: initial 2 + train 2 + final 2.
        self.assertEqual(FakeRuntime.instances[-1].prepare_count, 6)
        manifest = json.loads((Path(args.output) / "manifest.json").read_text())
        self.assertEqual(len(set(manifest["initial_head_hashes"].values())), 1)
        self.assertEqual(len(set(manifest["parameters_per_arm"].values())), 1)
        for arm in ARMS:
            self.assertTrue(result["initial_probability_checks"][arm]["passed"])
            self.assertTrue(result["initial_probability_checks"][arm]["delta_grid_exact_zero"])
            self.assertGreater(result["gradient_evidence"][arm]["reader_positive_gradient_updates"], 0)
        rows = [json.loads(x) for x in (Path(args.output)/"train.jsonl").read_text().splitlines()]
        for arm in ARMS:
            self.assertEqual(rows[0]["heads"][arm]["reader_grad_abs_sum"], 0)
            self.assertGreater(rows[1]["heads"][arm]["reader_grad_abs_sum"], 0)
        self.assertEqual(rows[-1]["lr"], 0.0)
        self.assertTrue(result["guard_report"]["restored"])

    def test_saved_predictions_reconstruct_confusion_and_ignore_label(self):
        args, result = self.run_small()
        directory = Path(args.output) / "final_val"
        report = json.loads((directory / "result.json").read_text())
        self.assertTrue(report["saved_predictions_reopened"])
        aggregate = {a: np.zeros((4, 4), dtype=np.int64) for a in train.PATHS}
        for path in sorted((directory / "predictions").glob("*.npz")):
            with np.load(path, allow_pickle=False) as d:
                for arm in train.PATHS:
                    aggregate[arm] += train._confusion(d["target"], d["prediction_"+arm])
        for arm in train.PATHS:
            self.assertEqual(aggregate[arm].tolist(), result["metrics"][arm]["confusion"])
            self.assertEqual(result["metrics"][arm]["valid_miou"], result["metrics"][arm]["foreground_miou"])
        cm = train._confusion(np.array([0, 1, 255]), np.array([0, 2, 3]))
        self.assertEqual(int(cm.sum()), 2)

    def test_checkpoint_manifest_and_re_evaluation_match(self):
        args, result = self.run_small()
        checkpoint = Path(result["last_checkpoint"])
        manifest = json.loads(checkpoint.with_suffix(".pt.json").read_text())
        self.assertEqual(digest(checkpoint), manifest["sha256"])
        evaluation = self.args("eval", checkpoint=str(checkpoint))
        with redirect_stdout(io.StringIO()):
            again = train.run_evaluation(evaluation, runtime_factory=FakeRuntime)
        self.assertEqual(again["metrics"], result["metrics"])
        self.assertFalse(again["full_validation_evidence"])
        self.assertFalse(again["primary_endpoint"])
        self.assertEqual(again["checkpoint_step"], 2)

    def test_tampered_checkpoint_stops_before_runtime(self):
        args, result = self.run_small()
        checkpoint = Path(result["last_checkpoint"])
        with checkpoint.open("ab") as f:
            f.write(b"changed")
        before = len(FakeRuntime.instances)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            train.run_evaluation(self.args("eval", checkpoint=str(checkpoint)), runtime_factory=FakeRuntime)
        self.assertEqual(len(FakeRuntime.instances), before)
        self.assertFalse(json.loads((self.base/"eval"/"summary.json").read_text())["complete"])

    def test_different_bank_binding_refused(self):
        _, result = self.run_small()
        class OtherBank(FakeRuntime):
            protocol_tag = "different-bank"
        with self.assertRaisesRegex(ValueError, "binding differs"):
            train.run_evaluation(self.args("eval", checkpoint=result["last_checkpoint"]),
                                 runtime_factory=OtherBank)

    def test_terminal_weights_survive_final_integrity_failure(self):
        class FailingFinal(FakeRuntime):
            fail_final_verify = True
        args = self.args()
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "final integrity"):
            train.run_training(args, runtime_factory=FailingFinal)
        result = json.loads((Path(args.output)/"summary.json").read_text())
        self.assertFalse(result["complete"])
        self.assertTrue(result["training_complete"])
        self.assertTrue(Path(result["last_checkpoint"]).is_file())
        self.assertEqual(result["updates_per_arm"], {a: 2 for a in ARMS})
        self.assertTrue((Path(args.output)/"traceback.txt").is_file())
        final = json.loads((Path(args.output)/"final_val"/"result.json").read_text())
        self.assertTrue(final["final_integrity_pending"])
        self.assertFalse(final["full_validation_evidence"])

    def test_guard_restore_failure_prevents_complete(self):
        class FailingRestore(FakeRuntime):
            fail_restore = True
        args = self.args()
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "not cleanly restored"):
            train.run_training(args, runtime_factory=FailingRestore)
        result = json.loads((Path(args.output)/"summary.json").read_text())
        self.assertFalse(result["complete"])
        self.assertFalse(result["full_validation_evidence"])

    def test_old_output_is_never_overwritten(self):
        args = self.args()
        Path(args.output).mkdir()
        marker = Path(args.output)/"original.txt"
        marker.write_text("preserve")
        with self.assertRaises(FileExistsError):
            train.run_training(args, runtime_factory=FakeRuntime)
        self.assertEqual(marker.read_text(), "preserve")

    def test_distinct_training_seed_after_historical_seed_reset(self):
        args1, _ = self.run_small(self.args("one", seed=3407))
        args2, _ = self.run_small(self.args("two", seed=3408))
        m1 = json.loads((Path(args1.output)/"manifest.json").read_text())
        m2 = json.loads((Path(args2.output)/"manifest.json").read_text())
        self.assertNotEqual(m1["initial_head_hashes"]["r0"], m2["initial_head_hashes"]["r0"])

    def test_transition_accounting(self):
        # GT D rescued from I; correct I wrongly changed to D; one both-wrong.
        a = train._transition(np.array([2, 1, 3]), np.array([1, 1, 0]), np.array([2, 2, 1]))
        report = train._transition_report(a)
        self.assertEqual(report["rescued_by_gt_class"], [0, 0, 1, 0])
        self.assertEqual(report["harmed_by_gt_class"], [0, 1, 0, 0])
        self.assertEqual(report["both_wrong_by_gt_class"], [0, 0, 0, 1])

    def test_formal_budget_cannot_be_silently_shortened(self):
        with self.assertRaisesRegex(ValueError, "5000"):
            train.run_training(self.args(mode="train", steps=2), runtime_factory=FakeRuntime)

    def test_probability_tolerance_not_argmax_tie_is_initial_gate(self):
        class TiedRuntime(FakeRuntime):
            def prepare(self, image):
                example = super().prepare(image)
                example.common["base_probability_full"].fill_(0.25)
                return example
        class CloseProbability(torch.nn.Module):
            def forward(self, **kwargs):
                probability = kwargs["base_probability_full"].clone()
                probability[:, 1] += 1e-6
                probability[:, 2] -= 1e-6
                return {"probabilities": probability, "delta_grid": probability.new_zeros(1,4,2,2)}
        runtime = TiedRuntime(self.args(), self.base)
        with redirect_stdout(io.StringIO()):
            result = train._evaluate(runtime, {a: CloseProbability() for a in ARMS}, runtime.val_names,
                                     self.base/"tied", seed=3407, smoke=True, initial=True)
        self.assertTrue(result["execution_complete"])
        for arm in ARMS:
            self.assertTrue(result["initial_probability_checks"][arm]["passed"])
            self.assertEqual(result["initial_probability_checks"][arm]["prediction_differences"], 8)

    def test_illegal_target_is_checked_before_uint8_wrap(self):
        runtime = FakeRuntime(self.args(), self.base)
        example = runtime.prepare("val_a")
        example.target[0,0,0] = 256
        with self.assertRaisesRegex(ValueError, "before any uint8"):
            train._assert_example(example)

    def test_twenty_updates_epoch_rollover_and_intermediate_checkpoints(self):
        args, result = self.run_small(self.args(steps=20, eval_every=5))
        self.assertEqual(result["updates_per_arm"], {a: 20 for a in ARMS})
        self.assertEqual(result["train_image_count"], 4)
        rows = [json.loads(x) for x in (Path(args.output)/"train.jsonl").read_text().splitlines()]
        self.assertEqual([r["epoch"] for r in rows], [i//4 for i in range(20)])
        # 20 training + 2 initial + 3*2 intermediate + 2 final, shared across arms.
        self.assertEqual(FakeRuntime.instances[-1].prepare_count, 30)
        checkpoints = Path(args.output)/"checkpoints"
        self.assertEqual(len(list(checkpoints.glob("*.pt"))), 4)
        for step in (5, 10, 15):
            metadata = json.loads((checkpoints/("checkpoint_%06d.pt.json" % step)).read_text())
            self.assertFalse(metadata["training_complete"])
            evidence = json.loads((Path(args.output)/("val_step_%06d" % step)/"result.json").read_text())
            self.assertFalse(evidence["full_validation_evidence"])
            self.assertFalse(evidence["saved_predictions_reopened"])
        self.assertTrue(json.loads((checkpoints/"checkpoint_last.pt.json").read_text())["training_complete"])


if __name__ == "__main__":
    unittest.main()
