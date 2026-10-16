"""CPU contracts for the checkpoint-500 LR fork; no BRIGHT/CUDA evidence."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import continuation as subject
import evaluation
import study


class TinyHead(torch.nn.Module):
    """Stochastic test head makes accidental unmatched RNG observable."""

    def __init__(self):
        super().__init__()
        self.query_projection = torch.nn.Conv2d(3, 4, 1)
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, *, features, offset, **unused):
        value = self.query_projection(self.dropout(features)) + offset
        return {"probabilities": value.softmax(dim=1)}


class TinyRuntime:
    @staticmethod
    def loss(probability, target):
        return torch.nn.functional.nll_loss(probability.clamp_min(1e-8).log(), target)


def example():
    features = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2) / 11
    return SimpleNamespace(
        common={"features": features},
        supports={"b": {"offset": 0.0}, "d": {"offset": 0.1}},
        target=torch.tensor([[[0, 1], [2, 3]]]),
    )


def trained_checkpoint():
    """Real AdamW states from 500 updates, not invented optimizer counters."""
    saved = {
        "step": 500, "mode": "train", "smoke": False,
        "training_complete": False, "heads": {}, "optimizers": {},
        "config": {"steps": 5000, "eval_every": 500, "lr": 1e-4,
                   "weight_decay": 0.005},
        "protocol_info": {"synthetic": True}, "source_hash": "synthetic",
        "input_file_sha256": {}, "updates_per_arm": {"b": 500, "d": 500},
    }
    fixture = example()
    for index, arm in enumerate(("b", "d")):
        torch.manual_seed(3407 + index)
        head = TinyHead()
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.005)
        for step in range(1, 501):
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = subject.lr_at_step(step)
            probability = head(**fixture.common, **fixture.supports[arm])["probabilities"]
            TinyRuntime.loss(probability, fixture.target).backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        saved["heads"][arm] = copy.deepcopy(head.state_dict())
        saved["optimizers"][arm] = copy.deepcopy(optimizer.state_dict())
    saved["rng"] = subject.capture_rng()
    return saved


class ScheduleContracts(unittest.TestCase):
    def test_preserves_original_warmup_and_horizon(self):
        for step in (1, 250, 499, 500):
            expected = 1e-4 * step / 500
            self.assertEqual(subject.lr_at_step(step), expected)
            self.assertEqual(subject.lr_at_step(step, multiplier=0.2), expected)
        for step in (501, 750, 1000, 1500, 4999, 5000):
            expected = 1e-4 * 0.5 * (1 + math.cos(math.pi * (step - 500) / 4500))
            self.assertEqual(subject.lr_at_step(step), expected)
            self.assertEqual(subject.lr_at_step(step, multiplier=0.2), 0.2 * expected)
        self.assertGreater(subject.lr_at_step(1500, multiplier=0.2), 0)
        self.assertEqual(subject.lr_at_step(5000), 0)

    def test_multiplier_starts_strictly_after_500(self):
        self.assertEqual(subject.lr_at_step(500, multiplier=0.2), 1e-4)
        self.assertAlmostEqual(subject.lr_at_step(501, multiplier=0.2), 2e-5, delta=1e-10)
        self.assertNotEqual(subject.lr_at_step(1500),
                            subject.lr_at_step(1500, horizon=1500))

    def test_invalid_schedule_fails(self):
        cases = ({"step": 0}, {"step": 5001}, {"step": True}, {"step": 1.5},
                 {"step": 501, "maximum": float("nan")},
                 {"step": 501, "maximum": -1},
                 {"step": 501, "multiplier": 0},
                 {"step": 501, "multiplier": float("inf")},
                 {"step": 501, "warmup": 5000})
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                subject.lr_at_step(**args)


class ContinuationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.saved = trained_checkpoint()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fork(self):
        return subject.fork_heads_and_optimizers(
            self.saved, SimpleNamespace(chunk_size=2), "cpu", model_factory=TinyHead)

    def test_checkpoint_requires_real_optimizer_rng_and_correct_origin(self):
        self.assertTrue(subject.validate_checkpoint_payload(self.saved))
        mutations = (
            lambda s: s.update(step=1500),
            lambda s: s.update(smoke=True),
            lambda s: s.update(training_complete=True),
            lambda s: s.pop("rng"),
            lambda s: s["config"].update(steps=1500),
            lambda s: s["config"].update(eval_every=100),
            lambda s: s["optimizers"].pop("b"),
            lambda s: s["optimizers"]["d"].update(state={}),
            lambda s: s["optimizers"]["b"]["param_groups"][0].update(lr=2e-5),
        )
        for mutate in mutations:
            bad = copy.deepcopy(self.saved)
            mutate(bad)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                subject.validate_checkpoint_payload(bad)

    def test_incomplete_or_nonfinite_adam_moments_rejected(self):
        for fault in ("step", "missing", "nonfinite"):
            bad = copy.deepcopy(self.saved)
            item = next(iter(bad["optimizers"]["b"]["state"].values()))
            if fault == "step":
                item["step"].fill_(499)
            elif fault == "missing":
                item.pop("exp_avg_sq")
            else:
                item["exp_avg"].flatten()[0] = float("nan")
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                subject.validate_checkpoint_payload(bad)

    def test_four_independent_branches_restore_model_and_adam_exactly(self):
        original_hash = subject.state_fingerprint(self.saved)
        heads, optimizers, bindings = self.fork()
        self.assertEqual(set(heads), set(subject.BINDINGS))
        all_parameter_addresses, all_moment_addresses = [], []
        for name, (base, multiplier) in subject.BINDINGS.items():
            self.assertEqual(subject.state_fingerprint(heads[name].state_dict()),
                             subject.state_fingerprint(self.saved["heads"][base]))
            self.assertEqual(subject.state_fingerprint(optimizers[name].state_dict()),
                             subject.state_fingerprint(self.saved["optimizers"][base]))
            self.assertEqual(bindings[name]["base_arm"], base)
            self.assertEqual(bindings[name]["post500_lr_multiplier"], multiplier)
            for parameter in heads[name].parameters():
                all_parameter_addresses.append(parameter.data_ptr())
                state = optimizers[name].state[parameter]
                self.assertEqual(float(state["step"]), 500)
                all_moment_addresses.extend(state[key].data_ptr() for key in ("exp_avg", "exp_avg_sq"))
        self.assertEqual(len(all_parameter_addresses), len(set(all_parameter_addresses)))
        self.assertEqual(len(all_moment_addresses), len(set(all_moment_addresses)))
        self.assertEqual(original_hash, subject.state_fingerprint(self.saved))

    def test_stochastic_matched_step_changes_only_learning_rate_within_pair(self):
        heads, optimizers, _ = self.fork()
        fixture = example()
        input_hash = subject.state_fingerprint((fixture.common, fixture.supports, fixture.target))
        saved_hash = subject.state_fingerprint(self.saved)
        # eval mode must not silently persist into a training update.
        for head in heads.values():
            head.eval()
        rows = subject.train_matched_step(heads, optimizers, fixture, TinyRuntime(), 501, 1e-4)
        for base in ("b", "d"):
            control, low = base + "_control", base + "_lr02"
            self.assertEqual(rows[control]["loss"], rows[low]["loss"])
            self.assertEqual(rows[control]["grad_norm"], rows[low]["grad_norm"])
            self.assertAlmostEqual(rows[low]["lr"] / rows[control]["lr"], 0.2)
            self.assertGreater(rows[control]["reader_grad_abs_sum"], 0)
            self.assertNotEqual(subject.state_fingerprint(heads[control].state_dict()),
                                subject.state_fingerprint(heads[low].state_dict()))
            for left, right in zip(heads[control].parameters(), heads[low].parameters()):
                self.assertEqual(float(optimizers[control].state[left]["step"]), 501)
                self.assertEqual(float(optimizers[low].state[right]["step"]), 501)
                # Same starting AdamW state + same gradients = same moments.
                for key in ("exp_avg", "exp_avg_sq"):
                    self.assertTrue(torch.equal(optimizers[control].state[left][key],
                                                optimizers[low].state[right][key]))
        self.assertTrue(all(head.training for head in heads.values()))
        self.assertEqual(input_hash, subject.state_fingerprint((fixture.common, fixture.supports, fixture.target)))
        self.assertEqual(saved_hash, subject.state_fingerprint(self.saved))

    def test_random_stream_advances_once_not_once_per_branch(self):
        heads, optimizers, _ = self.fork()
        fixture = example()
        before = subject.capture_rng()
        heads["b_control"].train()
        heads["b_control"](**fixture.common, **fixture.supports["b"])
        expected = subject.state_fingerprint(subject.capture_rng())
        subject.restore_rng(before)
        subject.train_matched_step(heads, optimizers, fixture, TinyRuntime(), 501, 1e-4)
        self.assertEqual(subject.state_fingerprint(subject.capture_rng()), expected)

    def test_no_weights_only_or_before500_update(self):
        heads, optimizers, _ = self.fork()
        for step in (500, 1501):
            with self.subTest(step=step), self.assertRaises(ValueError):
                subject.train_matched_step(heads, optimizers, example(), TinyRuntime(), step, 1e-4)
        del optimizers["b_lr02"]
        with self.assertRaises(ValueError):
            subject.train_matched_step(heads, optimizers, example(), TinyRuntime(), 501, 1e-4)

    def test_skipped_optimizer_step_is_not_reported_as_success(self):
        heads, optimizers, _ = self.fork()
        with self.assertRaisesRegex(ValueError, 'actual update count'):
            subject.train_matched_step(heads, optimizers, example(), TinyRuntime(), 503, 1e-4)

    def test_nonfinite_loss_stops_before_update(self):
        class BadRuntime:
            @staticmethod
            def loss(probability, target):
                return probability.sum() * float("nan")
        heads, optimizers, _ = self.fork()
        before = {name: subject.state_fingerprint(head.state_dict()) for name, head in heads.items()}
        with self.assertRaises(ValueError):
            subject.train_matched_step(heads, optimizers, example(), BadRuntime(), 501, 1e-4)
        self.assertEqual(before, {name: subject.state_fingerprint(head.state_dict()) for name, head in heads.items()})

    def test_missing_head_parameter_is_not_silently_ignored(self):
        bad = copy.deepcopy(self.saved)
        bad["heads"]["b"].pop(next(iter(bad["heads"]["b"])))
        with self.assertRaises(RuntimeError):
            subject.fork_heads_and_optimizers(bad, SimpleNamespace(chunk_size=2), "cpu", model_factory=TinyHead)


class UtilityContracts(unittest.TestCase):
    def test_rng_roundtrip_includes_python_numpy_torch(self):
        original = subject.capture_rng()
        first = (random.random(), np.random.uniform(size=3), torch.rand(3))
        subject.restore_rng(original)
        second = (random.random(), np.random.uniform(size=3), torch.rand(3))
        self.assertEqual(first[0], second[0])
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_invalid_rng_does_not_partially_change_global_state(self):
        original = subject.capture_rng()
        bad = copy.deepcopy(original)
        bad["torch_cpu"] = bad["torch_cpu"].to(torch.int64)
        with self.assertRaises(ValueError):
            subject.restore_rng(bad)
        self.assertEqual(subject.state_fingerprint(original), subject.state_fingerprint(subject.capture_rng()))

    def test_batch_identity_accepts_only_original_batch_one_shape(self):
        self.assertEqual(subject.raw_batch_identity((None, None, None, None, ["img"])), "img")
        for batch in ([], [None] * 4, [None] * 4 + ["img"],
                      [None] * 4 + [["a", "b"]], [None] * 4 + [[42]]):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                subject.raw_batch_identity(batch)

    def test_fingerprint_detects_dtype_shape_and_nested_state_changes(self):
        x = torch.tensor([1.0, 2.0])
        first = subject.state_fingerprint({"a": [x], "b": 500})
        self.assertEqual(first, subject.state_fingerprint({"b": 500, "a": [x.clone()]}))
        self.assertNotEqual(first, subject.state_fingerprint({"a": [x.double()], "b": 500}))
        self.assertNotEqual(first, subject.state_fingerprint({"a": [x.reshape(1, 2)], "b": 500}))
        self.assertNotEqual(first, subject.state_fingerprint({"a": [x], "b": 501}))


def simple_metrics(matrix):
    cm = np.asarray(matrix)
    tp, gt, pred = np.diag(cm), cm.sum(1), cm.sum(0)
    def ratio(numerator, denominator):
        return [float(n) / float(d) if d else None for n, d in zip(numerator, denominator)]
    iou = ratio(tp, gt + pred - tp)
    return {"confusion": cm.tolist(), "iou": iou,
            "precision": ratio(tp, pred), "recall": ratio(tp, gt),
            "foreground_miou": sum(x or 0 for x in iou[1:]) / 3}


class EvaluationRuntime(TinyRuntime):
    def __init__(self, unsafe=False):
        self.train_names = ["train0", "train1"]
        self.val_names = ["val0", "val1"]
        self.device = torch.device("cpu")
        self.protocol_info = {"synthetic": True}
        self.bank = {"events": torch.tensor([7]), "images": torch.tensor([7]),
                     "eligible": torch.tensor([True])}
        self.deps = SimpleNamespace(region_nll=TinyRuntime.loss)
        self.unsafe = unsafe
        self.prepared = []

    def batches(self, names, train, seed):
        # Consuming all RNG families makes the evaluator's isolation test real.
        random.random()
        np.random.uniform()
        torch.rand(1)
        yield from reversed(names) if train else names

    def prepare(self, name):
        self.prepared.append(name)
        sample = example()
        sample.image, sample.event = name, "query_event"
        sample.common["base_probability_full"] = torch.full((1, 4, 2, 2), 0.25)
        for arm in ("c", "d"):
            sample.supports.setdefault(arm, {})["support_ids"] = torch.tensor([[0]])
        sample.diagnostics = {"query_GT_used_for_supports": False,
                              "event_id": 7 if self.unsafe else 8, "image_id": 8}
        return sample


class EvaluationContracts(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name) / "result"
        self.heads = {name: TinyHead() for name in subject.BINDINGS}
        self.heads["b_lr02"].eval()
        self.original_train = SimpleNamespace(_assert_example=lambda x: None, _metrics=simple_metrics)

    def tearDown(self):
        self.temp.cleanup()
        torch.set_num_threads(self.threads)

    def evaluate(self, runtime, split="val", save=True):
        names = runtime.val_names if split == "val" else runtime.train_names
        with redirect_stdout(io.StringIO()):
            return evaluation.evaluate(runtime, self.heads, names, self.folder,
                step=1500, split=split, seed=3407, original_train=self.original_train,
                save_predictions=save)

    def test_cpu_small_val_is_not_formal_and_saved_predictions_are_reopened(self):
        runtime = EvaluationRuntime()
        modes = {name: head.training for name, head in self.heads.items()}
        initial = subject.capture_rng()
        result = self.evaluate(runtime)
        self.assertTrue(result["execution_complete"])
        self.assertFalse(result["full_validation_evidence"])
        self.assertFalse(result["whole_val450_processed"])
        self.assertFalse(result["complete"])
        self.assertTrue(result["smoke"])
        self.assertTrue(result["final_integrity_pending"])
        self.assertTrue(result["saved_predictions_reopened"])
        self.assertFalse(result["test_accessed"])
        self.assertEqual(result["image_count"], 2)
        self.assertEqual(len(list((self.folder / "predictions").glob("*.npz"))), 2)
        self.assertEqual(runtime.prepared, runtime.val_names)
        self.assertEqual(modes, {name: head.training for name, head in self.heads.items()})
        self.assertEqual(subject.state_fingerprint(initial), subject.state_fingerprint(subject.capture_rng()))
        for metrics in result["metrics"].values():
            self.assertEqual(np.asarray(metrics["confusion"]).sum(), 8)
            self.assertAlmostEqual(metrics["mean_image_loss"],
                                   metrics["mean_image_nll"] + metrics["mean_image_lovasz"])

    def test_fixed_train_preserves_identity_set_and_records_shuffled_order(self):
        runtime = EvaluationRuntime()
        result = self.evaluate(runtime, split="fixed_train", save=False)
        self.assertEqual(runtime.prepared, list(reversed(runtime.train_names)))
        self.assertEqual(result["names_sha256"], evaluation.names_hash(sorted(runtime.train_names)))
        self.assertNotEqual(result["order_sha256"], result["requested_order_sha256"])
        self.assertFalse(result["full_validation_evidence"])
        self.assertFalse(result["prediction_evidence_complete"])
        self.assertIn("not full Train2918", result["fixed_train_scope"])

    def test_same_event_support_fails_and_restores_rng_and_head_modes(self):
        initial = subject.capture_rng()
        modes = {name: head.training for name, head in self.heads.items()}
        with self.assertRaisesRegex(ValueError, "query events"):
            self.evaluate(EvaluationRuntime(unsafe=True))
        self.assertEqual(subject.state_fingerprint(initial), subject.state_fingerprint(subject.capture_rng()))
        self.assertEqual(modes, {name: head.training for name, head in self.heads.items()})
        self.assertFalse((self.folder / "result.json").exists())

    def test_test_and_cross_split_evaluation_refused(self):
        runtime = EvaluationRuntime()
        for split, names in (("test", runtime.val_names), ("val", runtime.train_names)):
            with self.subTest(split=split), self.assertRaises(ValueError):
                evaluation.evaluate(runtime, self.heads, names, self.folder,
                    step=1500, split=split, seed=3407, original_train=self.original_train)

    def test_val_identity_order_is_checked(self):
        class BadOrder(EvaluationRuntime):
            def batches(self, names, train, seed):
                yield from reversed(names)
        with self.assertRaisesRegex(ValueError, "Val loader"):
            self.evaluate(BadOrder())
        self.assertFalse((self.folder / "result.json").exists())

    def test_saved_prediction_corruption_is_rejected_before_evidence_is_returned(self):
        reopen = evaluation._reopen_predictions
        def corrupt(folder, journal, *args):
            with (Path(folder) / journal[0]["file"]).open("ab") as stream:
                stream.write(b"corrupt")
            return reopen(folder, journal, *args)
        with patch.object(evaluation, "_reopen_predictions", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "prediction hash mismatch"):
                self.evaluate(EvaluationRuntime())
        self.assertFalse((self.folder / "result.json").exists())

    def test_joint_transition_distinguishes_rescue_harm_from_net_difference(self):
        target = np.array([2, 2, 2, 1, 255], dtype=np.uint8)
        before = np.array([0, 2, 1, 1, 0], dtype=np.uint8)
        after = np.array([2, 0, 1, 2, 3], dtype=np.uint8)
        counts = evaluation.transition(target, before, after)
        report = evaluation.transition_report(counts, "old", "new")
        self.assertEqual(report["rescued_by_gt_class"][2], 1)
        self.assertEqual(report["harmed_by_gt_class"][2], 1)
        self.assertEqual(report["net_correct_change_by_gt_class"][2], 0)
        self.assertEqual(report["damaged_correct_to_background"], 1)
        self.assertEqual(report["damaged_background_to_correct"], 1)
        self.assertEqual(report["damaged_to_background_net_change"], 0)
        self.assertEqual(report["correct_intact_changed_to_damaged"], 1)
        self.assertEqual(int(counts.sum()), 4)

    def test_initial_replay_uses_percentage_points_and_requires_full_count(self):
        matrix = np.diag([10, 20, 30, 40])
        old = {"image_count": 450, "execution_complete": True,
               "metrics": {name: simple_metrics(matrix) for name in ("grid_probq", "b", "d")}}
        current = {"image_count": 450, "execution_complete": True,
                   "metrics": {name: simple_metrics(matrix) for name in evaluation.PATHS}}
        self.assertTrue(evaluation.compare_initial_to_old(current, old)["passed"])
        current["metrics"]["d_lr02"]["iou"][2] -= 0.0003
        self.assertFalse(evaluation.compare_initial_to_old(current, old)["passed"])
        current["metrics"]["d_lr02"]["iou"][2] += 0.0003
        current["image_count"] = 2
        self.assertFalse(evaluation.compare_initial_to_old(current, old)["passed"])


class StaticBindingContracts(unittest.TestCase):
    """Exercise provenance checks using synthetic files, never model loading."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root, self.tool = self.base / "old_run", self.base / "original_tool"
        self.train = self.root / "train"
        (self.train / "checkpoints").mkdir(parents=True)
        (self.train / "val_step_000500").mkdir()
        self.tool.mkdir()
        (self.tool / "run.py").write_text("# Synthetic source binding fixture\n")
        checksum = hashlib.sha256()
        checksum.update(b"run.py")
        checksum.update((self.tool / "run.py").read_bytes())
        self.source = checksum.hexdigest()
        self.dump(self.tool / "SHA256SUMS.json", {"run.py": study.digest(self.tool / "run.py")})
        self.dump(self.train / "summary.json", {"complete": True, "full_validation_evidence": True, "smoke": False})
        self.dump(self.root / "workflow_state.json", {"complete": True, "full_validation_evidence": True})
        self.names = ["train_%04d" % i for i in range(2918)]
        self.dump(self.train / "manifest.json", {"source_hash": self.source, "train_names": self.names,
                  "val_names": ["val_%04d" % i for i in range(450)]})
        checkpoint = self.train / "checkpoints/checkpoint_000500.pt"
        checkpoint.write_bytes(b"synthetic artifact; never deserialized")
        self.dump(checkpoint.with_suffix(".pt.json"), {"step": 500, "mode": "train", "smoke": False,
                  "training_complete": False, "sha256": study.digest(checkpoint), "source_hash": self.source})
        self.dump(self.train / "val_step_000500/result.json", {"image_count": 450, "execution_complete": True,
                  "metrics": {"grid_probq": {}, "b": {}, "d": {}}})
        self.rows = [{"step": i + 1, "image": self.names[i % 2918], "epoch": i // 2918}
                     for i in range(5000)]
        self.write_journal()
        self.args = SimpleNamespace(train_root=str(self.root), tool_dir=str(self.tool))

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def dump(path, data):
        path.write_text(json.dumps(data))

    def write_journal(self):
        (self.train / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def validate(self):
        with patch.object(study, "EXPECTED_SOURCE", self.source):
            return study.validate_static(self.args)

    def test_validates_binding_and_detects_original_artifact_changes(self):
        record = self.validate()
        self.assertEqual(len(record["rows"]), 5000)
        study.check_old_files(record)
        with (self.train / "summary.json").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            study.check_old_files(record)

    def test_incomplete_original_is_not_promoted_to_complete(self):
        self.dump(self.train / "summary.json", {"complete": False, "full_validation_evidence": True, "smoke": False})
        with self.assertRaisesRegex(ValueError, "must be complete"):
            self.validate()

    def test_checkpoint_file_must_match_original_manifest(self):
        (self.train / "checkpoints/checkpoint_000500.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "binding failed"):
            self.validate()

    def test_old_source_extra_python_is_not_silently_imported(self):
        (self.tool / "model.py").write_text("# unbound file\n")
        with self.assertRaisesRegex(ValueError, "Unexpected Python"):
            self.validate()

    def test_original_order_requires_contiguous_unique_first1500(self):
        self.rows[999]["image"] = self.rows[998]["image"]
        self.write_journal()
        with self.assertRaisesRegex(ValueError, "epoch-zero"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
