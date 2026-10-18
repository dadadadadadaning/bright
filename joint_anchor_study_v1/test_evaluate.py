"""CPU synthetic contracts using the pinned runtime's actual metric functions.

These tests are not evidence of BRIGHT data, CUDA execution or trained benefit.
The bootstrap supplies BRIGHT_ORIGINAL_TOOL and BRIGHT_LOSS_HELPERS.
"""
import ast
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import evaluate


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _functions(path, names, scope):
    """Load exact inherited helper definitions without importing its CLI."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name in names]
    if {n.name for n in nodes} != set(names):
        raise RuntimeError("Pinned dependency helper definitions missing")
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv2d(3, 4, 1)
        self.dropout = torch.nn.Dropout(.2)

    def forward(self, *, features, **unused):
        if self.training:
            raise AssertionError("Head must be evaluated in eval mode")
        logits = self.projection(self.dropout(features))
        return {"logits": logits, "probabilities": logits.softmax(1)}


class FakeRuntime:
    def __init__(self, objectives, unsafe=False, leak=False):
        self.train_names = ["train0", "train1"]
        self.val_names = ["val0", "val1"]
        self.device = torch.device("cpu")
        self.protocol_info = {"synthetic": True}
        self.bank = {"events": torch.tensor([7]), "images": torch.tensor([7])}
        self.objectives, self.unsafe, self.leak = objectives, unsafe, leak
        self.reference_calls, self.seen = 0, []

    def loss(self, probability, target):
        self.reference_calls += 1
        return self.objectives.objective_terms(probability, target)["loss"]

    def batches(self, names, train, seed):
        random.random()
        np.random.uniform()
        torch.rand(1)
        yield from reversed(names) if train else names

    def prepare(self, name):
        self.seen.append(name)
        target = torch.tensor([[[0, 1], [2, 255]]]) if name.endswith("0") else torch.tensor([[[0, 0], [3, 3]]])
        return SimpleNamespace(image=name, event="event" + name[-1], target=target,
            common={"features": torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2) / 11,
                    "base_probability_full": torch.tensor([.4, .3, .2, .1]).reshape(1, 4, 1, 1).expand(1, 4, 2, 2).clone()},
            supports={arm: {"support_ids": torch.tensor([[0]]),
                            "support_valid": torch.ones((1, 2), dtype=torch.bool),
                            "slot_valid": torch.ones((1, 1, 2), dtype=torch.bool)} for arm in "abcd"},
            diagnostics={"query_GT_used_for_supports": self.leak,
                         "event_id": 7 if self.unsafe else 8, "image_id": 8})


class EvaluationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original = Path(os.environ.get("BRIGHT_ORIGINAL_TOOL", "/workspace/bright-followup/lr_study_source/original_fix2"))
        helpers = Path(os.environ.get("BRIGHT_LOSS_HELPERS", "/workspace/bright-design-delivery/joint_loss_after500_v1"))
        if not (original / "train.py").is_file() or not (helpers / "objectives.py").is_file():
            raise unittest.SkipTest("Pinned original/legacy sources required; bootstrap validates these before CPU tests")
        cls.original = _functions(original / "train.py", ("_metrics", "_confusion", "_transition", "_transition_report", "_assert_example", "_coverage"),
                                  {"np": np, "torch": torch, "ARMS": tuple("abcd")})
        cls.legacy = SimpleNamespace(
            state=_load_module("anchor_eval_test_state", helpers / "continuation.py"),
            objectives=_load_module("anchor_eval_test_objectives", helpers / "objectives.py"),
            study=_functions(helpers / "study.py", ("validate_example",), {"torch": torch}))

    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name) / "val"
        head = TinyHead()
        self.heads = {p: copy.deepcopy(head) for p in ("control", "anchor")}
        self.heads["anchor"].eval()
        self.heads["control"].dropout.eval()

    def tearDown(self):
        self.temp.cleanup()
        torch.set_num_threads(self.threads)

    def runtime(self, **kwargs):
        return FakeRuntime(self.legacy.objectives, **kwargs)

    def modes(self):
        return {p: [m.training for m in h.modules()] for p, h in self.heads.items()}

    def run_evaluation(self, runtime=None, split="val", **kwargs):
        runtime = runtime or self.runtime()
        names = runtime.val_names if split == "val" else runtime.train_names
        with redirect_stdout(io.StringIO()):
            return evaluate.evaluate(runtime, self.heads, names, self.folder,
                step=1500, split=split, seed=3407, original_train=self.original,
                legacy=self.legacy, **kwargs)

    def test_real_helpers_synthetic_report_never_claims_cuda_or_formal(self):
        runtime = self.runtime()
        modes, rng = self.modes(), self.legacy.state.capture_rng()
        before = {p: self.legacy.state.state_fingerprint(h.state_dict()) for p, h in self.heads.items()}
        report = self.run_evaluation(runtime)
        for key in ("execution_complete", "smoke", "final_integrity_pending", "paired_counts_reopened", "rng_restored", "head_state_unchanged"):
            self.assertTrue(report[key], key)
        for key in ("complete", "full_validation_evidence", "whole_val450_processed", "test_accessed", "training", "prediction_arrays_saved"):
            self.assertFalse(report[key], key)
        self.assertEqual(report["image_count"], 2)
        self.assertEqual(report["valid_pixel_count"], 7)
        self.assertEqual(runtime.reference_calls, 6)  # exact inherited loss on EVERY path/image
        self.assertEqual(self.modes(), modes)
        self.assertEqual(self.legacy.state.state_fingerprint(rng), self.legacy.state.state_fingerprint(self.legacy.state.capture_rng()))
        self.assertEqual(before, {p: self.legacy.state.state_fingerprint(h.state_dict()) for p, h in self.heads.items()})
        self.assertFalse((self.folder / "predictions").exists())
        self.assertEqual(report["journal_sha256"], evaluate.digest(self.folder / "images.jsonl"))
        self.assertEqual((self.folder / "result.json.sha256").read_text().split()[0], evaluate.digest(self.folder / "result.json"))

    def test_objectives_decomposition_and_event_mean_are_consistent(self):
        report = self.run_evaluation()
        rows = [json.loads(s) for s in (self.folder / "images.jsonl").read_text().splitlines()]
        for path, metrics in report["metrics"].items():
            self.assertEqual(metrics["class_valid_counts"], [3, 1, 1, 2])
            self.assertAlmostEqual(metrics["mean_image_original_loss"], metrics["mean_image_nll"] + metrics["mean_image_lovasz"], places=6)
            self.assertAlmostEqual(metrics["mean_image_candidate_loss"], metrics["mean_image_original_loss"] + .1 * metrics["mean_image_anchor_kl"], places=6)
            self.assertAlmostEqual(metrics["mean_image_anchor_kl"], metrics["mean_image_anchor_localization_kl"] + metrics["mean_image_anchor_severity_kl_weighted"], places=6)
            self.assertAlmostEqual(metrics["mean_image_anchor_kl"], sum(metrics["mean_image_anchor_per_class_contribution"]), places=6)
            self.assertAlmostEqual(metrics["mean_image_anchor_kl"], metrics["mean_image_anchor_teacher_correct_kl_contribution"] + metrics["mean_image_anchor_teacher_wrong_kl_contribution"], places=6)
            for c in range(4):
                present = [r["objectives"][path]["anchor_per_class_conditional_kl"][c] for r in rows
                           if r["objectives"][path]["anchor_per_class_conditional_kl"][c] is not None]
                self.assertAlmostEqual(metrics["mean_present_image_anchor_per_class_conditional_kl"][c], sum(present) / len(present))
            event_mean = sum(report["per_event"][e][path]["mean_image_original_loss"] * n
                             for e, n in report["per_event_image_count"].items()) / 2
            self.assertAlmostEqual(metrics["mean_image_original_loss"], event_mean)
        self.assertAlmostEqual(report["metrics"]["grid_probq"]["mean_image_anchor_kl"], 0, places=6)
        self.assertEqual(report["metrics"]["anchor"], report["metrics"]["control"])

    def test_print_table_distinguishes_objective_from_performance(self):
        report = self.run_evaluation()
        output = io.StringIO()
        with redirect_stdout(output):
            evaluate.print_table(report)
        self.assertIn("lower KL or candidate loss alone is not a benefit", output.getvalue())
        self.assertIn("original+anchor loss", output.getvalue())

    def test_fixed_train_retains_actual_order_and_subset_scope(self):
        runtime = self.runtime()
        report = self.run_evaluation(runtime, split="fixed_train")
        self.assertEqual(runtime.seen, list(reversed(runtime.train_names)))
        self.assertNotEqual(report["order_sha256"], report["requested_order_sha256"])
        self.assertIn("not full Train2918", report["fixed_train_scope"])
        self.assertFalse(report["full_validation_evidence"])

    def test_exclusion_failure_restores_rng_and_modes(self):
        before, modes = self.legacy.state.capture_rng(), self.modes()
        with self.assertRaisesRegex(ValueError, "query image or event"):
            self.run_evaluation(self.runtime(unsafe=True))
        self.assertEqual(self.legacy.state.state_fingerprint(before), self.legacy.state.state_fingerprint(self.legacy.state.capture_rng()))
        self.assertEqual(self.modes(), modes)
        self.assertFalse((self.folder / "result.json").exists())

    def test_query_ground_truth_is_not_a_reader_input(self):
        with self.assertRaisesRegex(ValueError, "ground-truth separation"):
            self.run_evaluation(self.runtime(leak=True))

    def test_val_order_is_enforced(self):
        runtime = self.runtime()
        runtime.batches = lambda names, **kw: reversed(names)
        with self.assertRaisesRegex(ValueError, "Val loader"):
            self.run_evaluation(runtime)

    def test_test_split_and_cross_split_are_refused(self):
        runtime = self.runtime()
        for split, names in (("test", runtime.val_names), ("val", runtime.train_names)):
            with self.subTest(split=split), self.assertRaises(ValueError):
                evaluate.evaluate(runtime, self.heads, names, self.folder,
                    step=1500, split=split, seed=3407, original_train=self.original, legacy=self.legacy)
        self.assertFalse(self.folder.exists())

    def test_paired_counts_distinguish_rescue_from_net_change(self):
        target = np.array([2, 2, 2, 1, 255], dtype=np.uint8)
        before = np.array([0, 2, 1, 1, 0], dtype=np.uint8)
        after = np.array([2, 0, 1, 2, 3], dtype=np.uint8)
        counts = self.original._transition(target, before, after)
        report = evaluate._paired_report(counts, "before", "after", self.original)
        self.assertEqual(report["rescued_by_gt_class"][2], 1)
        self.assertEqual(report["harmed_by_gt_class"][2], 1)
        self.assertEqual(report["both_wrong_by_gt_class"][2], 1)
        self.assertEqual(report["before_wrong_by_gt_class"][2], 2)
        self.assertEqual(report["correct_intact_changed_to_damaged"], 1)
        self.assertEqual(report["axes"], ["GT", "before", "after"])

    def test_corrupted_compact_evidence_is_not_promoted(self):
        original = evaluate._check_journal
        def corrupt(path, *args):
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            rows[0]["paired_counts"]["control_vs_grid"][0][0][0] += 1
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            return original(path, *args)
        with patch.object(evaluate, "_check_journal", side_effect=corrupt), self.assertRaisesRegex(ValueError, "marginal confusion"):
            self.run_evaluation()
        self.assertFalse((self.folder / "result.json").exists())

    def test_original_loss_formula_disagreement_is_not_hidden(self):
        runtime = self.runtime()
        loss = runtime.loss
        runtime.loss = lambda p, y: loss(p, y) + .5
        with self.assertRaisesRegex(RuntimeError, "differs from inherited"):
            self.run_evaluation(runtime)

    def test_all_ignore_image_is_zero_and_not_dropped_from_image_average(self):
        runtime = self.runtime()
        prepare = runtime.prepare
        def with_ignore(name):
            example = prepare(name)
            if name.endswith("1"):
                example.target[:] = 255
            return example
        runtime.prepare = with_ignore
        report = self.run_evaluation(runtime)
        rows = [json.loads(s) for s in (self.folder / "images.jsonl").read_text().splitlines()]
        self.assertEqual(report["image_count"], 2)
        self.assertEqual(report["valid_pixel_count"], 3)
        for path in evaluate.PATHS:
            self.assertEqual(rows[1]["objectives"][path]["original_loss"], 0)
            self.assertEqual(rows[1]["objectives"][path]["anchor_kl"], 0)
            self.assertAlmostEqual(report["metrics"][path]["mean_image_original_loss"], rows[0]["objectives"][path]["original_loss"] / 2)
            self.assertIsNone(report["metrics"][path]["mean_present_image_anchor_per_class_conditional_kl"][3])


if __name__ == "__main__":
    unittest.main()
