"""CPU fixtures test evidence accounting, never BRIGHT/CUDA effectiveness."""
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

import evaluation


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_functions(path, names, scope):
    """Execute the exact pinned helpers without importing their CLI/model."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise RuntimeError("Missing inherited helper definition")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv2d(3, 4, 1)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, *, features, shift):
        if any(module.training for module in self.modules()):
            raise AssertionError("All head modules must be in eval mode")
        logits = self.projection(self.dropout(features)) + shift
        return {"logits": logits, "probabilities": logits.softmax(1)}


class TinyFactory:
    def __init__(self):
        self.prepared, self.forwarded = [], []

    def prepare(self, example, arms):
        self.prepared.append(example.image)
        shifts = {
            "e": [0., 0., 0., 0.], "f": [0., 1., 0., 0.],
            "b": [0., 0., 1., 0.], "d": [0., 0., 0., 1.],
        }
        return {arm: {"shift": torch.tensor(shifts.get(arm, [0.] * 4)).reshape(1, 4, 1, 1)}
                for arm in arms}

    def forward(self, head, example, arm, support):
        self.forwarded.append((example.image, arm, id(example)))
        return head(features=example.common["features"], **support)


class TinyRuntime:
    def __init__(self, objectives, *, unsafe=False, leak=False, all_ignore=False):
        self.train_names = ["train0", "train1", "train2"]
        self.val_names = ["val0", "val1", "val2"]
        self.device = torch.device("cpu")
        self.protocol_info = {"synthetic": True}
        self.bank = {"events": torch.tensor([7]), "images": torch.tensor([7])}
        self.front = torch.nn.Sequential(torch.nn.Dropout(.1))
        self.front[0].eval()
        self.objectives, self.unsafe, self.leak = objectives, unsafe, leak
        self.all_ignore = all_ignore
        self.reference_calls, self.seen, self.batch_calls = 0, [], []

    def loss(self, probability, target):
        self.reference_calls += 1
        return self.objectives.objective_terms(probability, target)["loss"]

    def batches(self, names, *, train, seed):
        self.batch_calls.append((list(names), train, seed))
        random.random()
        np.random.uniform()
        torch.rand(1)
        yield from reversed(names) if train else names

    def prepare(self, name):
        if any(module.training for module in self.front.modules()):
            raise AssertionError("Frozen frontend must be in eval mode")
        self.seen.append(name)
        # Two images share one event; the final all-ignore image is another event.
        if name.endswith("0"):
            target = torch.tensor([[[0, 1], [2, 255]]])
        elif name.endswith("1"):
            target = torch.tensor([[[0, 0], [3, 3]]])
        else:
            target = torch.full((1, 2, 2), 255, dtype=torch.long)
        if self.all_ignore:
            target.fill_(255)
        return SimpleNamespace(
            image=name, event="event0" if not name.endswith("2") else "event1", target=target,
            common={"features": torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2) / 11,
                    "base_probability_full": torch.tensor([.4, .3, .2, .1]).reshape(
                        1, 4, 1, 1).expand(1, 4, 2, 2).clone()},
            supports={arm: {"support_ids": torch.tensor([[0]]),
                            "support_valid": torch.ones((1, 2), dtype=torch.bool),
                            "slot_valid": torch.ones((1, 1, 2), dtype=torch.bool)}
                      for arm in "abcd"},
            diagnostics={"query_GT_used_for_supports": self.leak,
                         "event_id": 7 if self.unsafe else 8, "image_id": 8})


class EvaluationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw_reference = os.environ.get("BRIGHT_ABLATION_REFERENCE")
        if not raw_reference:
            raise RuntimeError("Set BRIGHT_ABLATION_REFERENCE to the exact reference package for CPU tests")
        reference = Path(raw_reference) / "deps"
        original, helpers = reference / "original", reference / "helpers"
        cls.original = exact_functions(
            original / "train.py",
            ("_metrics", "_confusion", "_transition", "_transition_report", "_assert_example", "_coverage"),
            {"np": np, "torch": torch, "ARMS": tuple("abcd")})
        cls.legacy = SimpleNamespace(
            state=load_module("ablation_eval_state", helpers / "continuation.py"),
            objectives=load_module("ablation_eval_objectives", helpers / "objectives.py"),
            study=exact_functions(helpers / "study.py", ("validate_example",), {"torch": torch}))

    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name) / "val"
        head = TinyHead()
        self.heads = {arm: copy.deepcopy(head) for arm in ("e", "f", "b", "d")}
        self.heads["e"].dropout.eval()
        self.heads["d"].eval()
        self.factory = TinyFactory()

    def tearDown(self):
        self.temporary.cleanup()
        torch.set_num_threads(self.threads)

    def runtime(self, **kwargs):
        return TinyRuntime(self.legacy.objectives, **kwargs)

    def modes(self, runtime):
        return ([module.training for head in self.heads.values() for module in head.modules()],
                [module.training for module in runtime.front.modules()])

    def run_evaluation(self, runtime=None, *, split="val", names=None, **kwargs):
        runtime = runtime or self.runtime()
        names = names if names is not None else (
            runtime.val_names if split == "val" else runtime.train_names)
        with redirect_stdout(io.StringIO()):
            return evaluation.evaluate(
                runtime, self.heads, names, self.folder, step=500, split=split,
                seed=3407, original_train=self.original, legacy=self.legacy,
                factory=self.factory, **kwargs)

    def test_one_frozen_extract_per_image_complete_modes_rng_and_pending_status(self):
        runtime = self.runtime()
        modes, rng = self.modes(runtime), self.legacy.state.capture_rng()
        hashes = {arm: self.legacy.state.state_fingerprint(head.state_dict())
                  for arm, head in self.heads.items()}
        result = self.run_evaluation(runtime, primary_endpoint=True)
        self.assertEqual(runtime.seen, runtime.val_names)
        self.assertEqual(self.factory.prepared, runtime.val_names)
        self.assertEqual(len(self.factory.forwarded), 12)
        for name in runtime.val_names:
            self.assertEqual(len({row[2] for row in self.factory.forwarded if row[0] == name}), 1)
        self.assertEqual(runtime.batch_calls, [(runtime.val_names, False, 3407)])
        self.assertEqual(runtime.reference_calls, 15)
        self.assertEqual(result["image_count"], 3)
        self.assertEqual(result["valid_pixel_count"], 7)
        self.assertEqual(self.modes(runtime), modes)
        self.assertEqual(self.legacy.state.state_fingerprint(self.legacy.state.capture_rng()),
                         self.legacy.state.state_fingerprint(rng))
        self.assertEqual(result["head_hashes"], hashes)
        for key in ("execution_complete", "smoke", "final_integrity_pending", "paired_counts_reopened",
                    "objective_totals_reopened", "per_event_totals_reopened", "module_modes_restored"):
            self.assertTrue(result[key], key)
        for key in ("complete", "full_validation_evidence", "primary_endpoint", "whole_val450_processed",
                    "test_accessed", "training", "prediction_arrays_saved"):
            self.assertFalse(result[key], key)
        self.assertTrue(result["primary_endpoint_requested"])
        self.assertEqual(result["journal_sha256"], evaluation.digest(self.folder / "images.jsonl"))
        self.assertEqual((self.folder / "result.json.sha256").read_text().split()[0],
                         evaluation.digest(self.folder / "result.json"))

    def test_losses_are_image_means_and_event_confusions_and_pairs_sum(self):
        result = self.run_evaluation()
        rows = [json.loads(line) for line in (self.folder / "images.jsonl").read_text().splitlines()]
        for path, metrics in result["metrics"].items():
            self.assertEqual(metrics["class_valid_counts"], [3, 1, 1, 2])
            self.assertEqual(metrics["class_present_image_counts"], [2, 1, 1, 1])
            self.assertAlmostEqual(metrics["mean_image_original_loss"],
                                   metrics["mean_image_nll"] + metrics["mean_image_lovasz"], places=6)
            for key in evaluation.SCALAR_TERMS:
                self.assertAlmostEqual(metrics["mean_image_" + key],
                    sum(row["objectives"][path][key] for row in rows) / 3, places=7)
                self.assertAlmostEqual(metrics["mean_image_" + key],
                    sum(result["per_event"][event][path]["mean_image_" + key] * count
                        for event, count in result["per_event_image_count"].items()) / 3, places=7)
            for term in ("nll", "lovasz"):
                self.assertAlmostEqual(sum(metrics["mean_image_class_" + term + "_contribution"]),
                                       metrics["mean_image_" + term], places=6)
            np.testing.assert_array_equal(metrics["confusion"], sum(
                np.asarray(event[path]["confusion"]) for event in result["per_event"].values()))
            self.assertEqual(result["per_event"]["event1"][path]["iou"], [None] * 4)
            self.assertNotIn("mean_image_candidate_loss", metrics)
            self.assertFalse(any("kl" in key.lower() for key in metrics))
        for key, report in result["transitions"].items():
            counts = np.asarray(report["counts"])
            before, after = report["axes"][1:]
            np.testing.assert_array_equal(counts.sum(2), result["metrics"][before]["confusion"])
            np.testing.assert_array_equal(counts.sum(1), result["metrics"][after]["confusion"])
            np.testing.assert_array_equal(counts, sum(
                np.asarray(event[key]["counts"]) for event in result["per_event_transitions"].values()))

    def test_fixed_train_uses_evaluation_transform_and_exact_order(self):
        runtime = self.runtime()
        result = self.run_evaluation(runtime, split="fixed_train", names=runtime.train_names[:2])
        self.assertEqual(runtime.batch_calls, [(runtime.train_names[:2], False, 3407)])
        self.assertEqual(runtime.seen, runtime.train_names[:2])
        self.assertEqual(result["order_sha256"], result["requested_order_sha256"])
        self.assertFalse(result["full_validation_evidence"])
        self.assertIn("not full Train2918", result["fixed_train_scope"])

    def test_evaluation_batches_hook_is_preferred(self):
        runtime = self.runtime()
        calls = []
        def loader(names, *, seed):
            calls.append((list(names), seed))
            yield from names
        runtime.evaluation_batches = loader
        self.run_evaluation(runtime, split="fixed_train")
        self.assertEqual(calls, [(runtime.train_names, 3407)])
        self.assertEqual(runtime.batch_calls, [])

    def test_all_ignore_absent_classes_are_null_with_finite_zero_loss(self):
        result = self.run_evaluation(self.runtime(all_ignore=True))
        for metric in result["metrics"].values():
            self.assertEqual(metric["iou"], [None] * 4)
            self.assertEqual(metric["precision"], [None] * 4)
            self.assertEqual(metric["recall"], [None] * 4)
            self.assertEqual(metric["mean_image_original_loss"], 0)
        self.assertEqual(result["valid_pixel_count"], 0)

    def test_leak_or_same_event_rejected_with_modes_rng_restored(self):
        for kwargs, message in (({"unsafe": True}, "query image or event"),
                                ({"leak": True}, "ground-truth separation")):
            runtime = self.runtime(**kwargs)
            rng, modes = self.legacy.state.capture_rng(), self.modes(runtime)
            self.folder = Path(self.temporary.name) / next(iter(kwargs))
            with self.assertRaisesRegex(ValueError, message):
                self.run_evaluation(runtime)
            self.assertEqual(modes, self.modes(runtime))
            self.assertEqual(self.legacy.state.state_fingerprint(rng),
                             self.legacy.state.state_fingerprint(self.legacy.state.capture_rng()))
            self.assertFalse((self.folder / "result.json").exists())

    def test_wrong_order_incomplete_duplicate_test_and_cross_split_rejected(self):
        runtime = self.runtime()
        for split, names in (("test", runtime.val_names), ("val", runtime.train_names),
                             ("val", [runtime.val_names[0]] * 2)):
            with self.assertRaises(ValueError):
                self.run_evaluation(runtime, split=split, names=names)
        for label, selected in (("order", list(reversed(runtime.val_names))),
                                ("incomplete", runtime.val_names[:-1]),
                                ("duplicate", [runtime.val_names[0]] * 3)):
            runtime = self.runtime()
            runtime.evaluation_batches = lambda names, seed, picked=selected: iter(picked)
            self.folder = Path(self.temporary.name) / label
            with self.assertRaisesRegex(ValueError, "requested identities"):
                self.run_evaluation(runtime)
            self.assertFalse((self.folder / "result.json").exists())

    def test_reopened_confusion_objective_and_event_corruption_rejected(self):
        original = evaluation._check_journal
        for kind in ("confusion", "objective", "event", "paired"):
            self.folder = Path(self.temporary.name) / kind
            def corrupt(path, *args, kind=kind):
                rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
                if kind == "confusion":
                    rows[0]["confusion"]["e"][0][0] += 1
                elif kind == "objective":
                    rows[0]["objectives"]["e"]["nll"] += .1
                elif kind == "event":
                    rows[0]["event"] = "event1"
                else:
                    key = next(iter(rows[0]["paired_counts"]))
                    rows[0]["paired_counts"][key][0][0][0] += 1
                Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))
                return original(path, *args)
            with patch.object(evaluation, "_check_journal", side_effect=corrupt):
                with self.assertRaises(ValueError):
                    self.run_evaluation()
            self.assertFalse((self.folder / "result.json").exists())

    def test_head_mutation_detected_and_modes_rng_still_restored(self):
        runtime = self.runtime()
        rng, modes = self.legacy.state.capture_rng(), self.modes(runtime)
        original = self.factory.forward
        def mutate(head, example, arm, support):
            head.projection.bias.add_(1)
            return original(head, example, arm, support)
        self.factory.forward = mutate
        with self.assertRaisesRegex(RuntimeError, "head parameters/buffers"):
            self.run_evaluation(runtime)
        self.assertEqual(modes, self.modes(runtime))
        self.assertEqual(self.legacy.state.state_fingerprint(rng),
                         self.legacy.state.state_fingerprint(self.legacy.state.capture_rng()))
        self.assertFalse((self.folder / "result.json").exists())

    def test_exact_causal_contrasts_are_available_only_for_present_arms(self):
        values = {"grid_probq": .30, "e": .40, "f": .42, "b": .43, "d": .48,
                  "a": .41, "c": .44, "r": .45, "l": .46, "n": .47}
        metrics = {arm: {"foreground_miou": value} for arm, value in values.items()}
        pairs = evaluation.comparison_pairs(evaluation.ARMS)
        contrast = evaluation.metric_contrasts(metrics, pairs)
        self.assertEqual(len(contrast), 19)
        self.assertAlmostEqual(contrast["b_minus_e"], 3.)
        self.assertAlmostEqual(contrast["d_minus_f"], 6.)
        self.assertAlmostEqual(contrast["d_minus_n"], 1.)
        self.assertAlmostEqual(contrast["e_minus_grid_probq"], 10.)
        interactions = evaluation.interaction_contrasts(metrics)
        self.assertAlmostEqual(interactions["(d-b)-(f-e)"], 3.)
        self.assertAlmostEqual(interactions["(d-c)-(b-a)"], 2.)
        self.assertEqual(set(evaluation.comparison_pairs(("d",))), {"d_minus_grid_probq"})
        self.assertEqual(evaluation.interaction_contrasts({"d": metrics["d"]}), {})

    def test_paired_rescue_and_harm_survive_equal_net_correct_count(self):
        target = np.array([2, 2, 2, 1, 255])
        before = np.array([0, 2, 1, 1, 0])
        after = np.array([2, 0, 1, 2, 3])
        counts = self.original._transition(target, before, after)
        report = evaluation._paired_report(counts, "e", "b", self.original)
        self.assertEqual(report["rescued_by_gt_class"][2], 1)
        self.assertEqual(report["harmed_by_gt_class"][2], 1)
        self.assertEqual(report["net_correct_change_by_gt_class"][2], 0)
        self.assertEqual(report["correct_damaged_changed_to_background"], 1)
        self.assertEqual(report["axes"], ["GT", "e", "b"])
        self.assertEqual(int(counts.sum()), 4)

    def test_mechanism_summary_streams_saved_json_and_restores_rng(self):
        runtime = self.runtime()
        runtime.last_extract = {"fixture": True}
        rng = self.legacy.state.capture_rng()
        originals = []
        def audit_example(example, current_runtime, support):
            value = {"image": example.image, "count": len(support)}
            originals.append(value)
            return value
        def summarize_audits(audits):
            self.assertNotIsInstance(audits, (list, tuple))
            rows = []
            for index, audit in enumerate(audits):
                self.assertIsNot(audit, originals[index])
                rows.append(audit)
            random.random()
            np.random.uniform()
            torch.rand(1)
            return {"image_count": len(rows), "count": sum(row["count"] for row in rows)}
        module = SimpleNamespace(audit_example=audit_example, summarize_audits=summarize_audits)
        with patch.dict("sys.modules", {"diagnostics": module}):
            result = self.run_evaluation(runtime)
        self.assertEqual(result["mechanism_audit"], {"image_count": 3, "count": 12})
        self.assertEqual(result["mechanism_audit_image_count"], 3)
        self.assertTrue(result["mechanism_audits_reopened"])
        self.assertEqual(self.legacy.state.state_fingerprint(rng),
                         self.legacy.state.state_fingerprint(self.legacy.state.capture_rng()))

    def test_streamed_mechanism_missing_or_wrong_identity_is_rejected(self):
        path = Path(self.temporary.name) / "audits.jsonl"
        examples = [
            {"image": "val0", "mechanism_audit": {"image": "val0"}},
            {"image": "val1", "mechanism_audit": {"image": "val1"}}]
        for corruption in ("missing_key", "wrong_image", "wrong_order"):
            rows = copy.deepcopy(examples)
            if corruption == "missing_key":
                rows[0]["mechanism_audits"] = rows[0].pop("mechanism_audit")
            elif corruption == "wrong_image":
                rows[0]["mechanism_audit"]["image"] = "train0"
            else:
                rows.reverse()
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.subTest(corruption=corruption), self.assertRaisesRegex(ValueError, "mechanism audit"):
                list(evaluation._journal_audits(path, ["val0", "val1"]))

    def test_actual_joint_model_all_arms_update_then_evaluate_without_changing_state(self):
        """Exercise the real model/support factory across the runner boundary."""
        from ablation_runner import update_arms
        from supports import SupportFactory
        from test_supports import make_fixture, _model

        runtime, example = make_fixture()
        example.diagnostics["query_GT_used_for_supports"] = False
        example.target = torch.arange(example.target.numel()).reshape_as(example.target) % 4
        example.image = "event0/train.png"
        runtime.train_names = [example.image]
        runtime.val_names = ["event0/val.png"]
        runtime.device = torch.device("cpu")
        runtime.protocol_info = {"synthetic": True}
        runtime.loss = lambda probability, target: self.legacy.objectives.objective_terms(
            probability, target)["loss"]
        runtime.deps = SimpleNamespace(region_nll=lambda probability, target:
            self.legacy.objectives.objective_terms(probability, target)["nll"])
        prepared = []
        def prepare(name):
            prepared.append(name)
            value = copy.copy(example)
            value.image = name
            return value
        runtime.prepare = prepare
        runtime.batches = lambda names, train, seed: iter(names)
        prototype = _model()
        self.heads = {arm: copy.deepcopy(prototype) for arm in evaluation.ARMS}
        self.factory = SupportFactory(runtime, seed=3407)
        optimizers = {arm: torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.005)
                      for arm, head in self.heads.items()}
        before = {arm: self.legacy.state.state_fingerprint(head.state_dict())
                  for arm, head in self.heads.items()}
        updates = update_arms(self.heads, optimizers, example, runtime, self.factory,
                              self.legacy.state, step=1, multiplier=1.)
        for arm, row in updates.items():
            self.assertEqual(row["optimizer_step"], 1)
            self.assertGreater(row["parameter_update_l2"], 0)
            self.assertGreater(row["reader_grad_abs_sum"], 0)
            self.assertAlmostEqual(row["original_loss"], row["nll"] + row["lovasz"], places=6)
            self.assertNotEqual(before[arm], self.legacy.state.state_fingerprint(
                self.heads[arm].state_dict()))
        optimizer_hash = self.legacy.state.state_fingerprint(
            {arm: optimizer.state_dict() for arm, optimizer in optimizers.items()})
        post_update_hashes = {arm: self.legacy.state.state_fingerprint(head.state_dict())
                              for arm, head in self.heads.items()}
        result = self.run_evaluation(runtime)
        self.assertEqual(prepared, runtime.val_names)
        self.assertEqual(set(result["metrics"]), set(evaluation.ARMS) | {"grid_probq"})
        self.assertEqual(len(result["transitions"]), 19)
        self.assertEqual(result["head_hashes"], post_update_hashes)
        self.assertEqual(optimizer_hash, self.legacy.state.state_fingerprint(
            {arm: optimizer.state_dict() for arm, optimizer in optimizers.items()}))
        self.assertEqual(result["valid_pixel_count"], example.target.numel())
        rows = [json.loads(line) for line in (self.folder / "images.jsonl").read_text().splitlines()]
        for arm in ("f", "c", "d", "r", "l", "n"):
            self.assertTrue(rows[0]["support_diagnostics"][arm]["selection_legal"])
        self.assertEqual(set(rows[0]["support_selection_sha256"]), set(evaluation.ARMS))
        self.assertFalse(result["full_validation_evidence"])
        self.assertTrue(result["final_integrity_pending"])


if __name__ == "__main__":
    unittest.main()
