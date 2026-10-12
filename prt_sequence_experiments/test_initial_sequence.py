"""CPU contracts for the fixed-attempt gate and actual-output scene observer."""
import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

spec = importlib.util.spec_from_file_location("sequence_initial", Path(__file__).with_name("initial_sequence.py"))
initial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(initial)


def passed_report():
    return dict(format="SRAC_REGION_HEAD_ADAPT_EVAL_V1", complete=True,
                full_validation_evidence=True, initial=True, smoke=False, test_accessed=False,
                training=False, image_count=450, expected_image_count=450,
                initial_parameters={"passed": True}, initial_checks_passed=True,
                initial_probability_checks={arm: dict(passed=True, checked_images=450, failed_images=0)
                                            for arm in initial.ARMS},
                reproduction={"passed": True}, e0_metric_comparison={"passed": True},
                native_a1_replay_ok=True, immutable={"files": True, "frozen_models": True},
                pool_guard={"restored": True, "failures": []})


def failed_report():
    report = passed_report()
    report.update(complete=False, full_validation_evidence=False, initial_checks_passed=False)
    report["initial_probability_checks"]["p"].update(passed=False, failed_images=1)
    return report


class GateTests(unittest.TestCase):
    def test_strict_full_initial_pass(self):
        self.assertTrue(initial.full_initial_pass(passed_report()))

    def test_prefix_cannot_pass_or_trigger_capture(self):
        for count in (2, 449):
            report = failed_report()
            report["image_count"] = count
            self.assertFalse(initial.isolated_initial_probability_failure(report))
            self.assertFalse(initial.full_initial_pass(report))

    def test_metric_native_pool_integrity_failures_are_other(self):
        variants = []
        for field in ("reproduction", "e0_metric_comparison"):
            r = failed_report()
            r[field]["passed"] = False
            variants.append(r)
        for field in ("native_a1_replay_ok", "initial"):
            r = failed_report()
            r[field] = False
            variants.append(r)
        r = failed_report()
        r["immutable"]["files"] = False
        variants.append(r)
        r = failed_report()
        r["pool_guard"]["failures"] = ["device failure"]
        variants.append(r)
        for report in variants:
            self.assertFalse(initial.isolated_initial_probability_failure(report))

    def test_t_failure_does_not_authorize_p_r_capture(self):
        r = failed_report()
        r["initial_probability_checks"]["t"]["passed"] = False
        self.assertFalse(initial.isolated_initial_probability_failure(r))

    def test_failed_initial_cannot_be_full_pass(self):
        self.assertTrue(initial.isolated_initial_probability_failure(failed_report()))
        self.assertFalse(initial.full_initial_pass(failed_report()))

    def test_all_image_counts_and_explicit_bool_required(self):
        report = passed_report()
        report["initial_probability_checks"]["p"]["checked_images"] = 449
        self.assertFalse(initial.full_initial_pass(report))
        report = passed_report()
        report["immutable"]["files"] = 1
        self.assertFalse(initial.full_initial_pass(report))


class EvidenceTests(unittest.TestCase):
    def fixture(self, folder):
        root, repo = Path(folder) / "failed", Path(folder) / "repo"
        (root / "fresh_e0").mkdir(parents=True)
        (root / "prt_fit/initial_val").mkdir(parents=True)
        (repo / "bda_benchmark").mkdir(parents=True)
        names = [str(i) for i in range(450)]
        protocol = root / "fresh_e0/protocol.json"
        protocol.write_text(json.dumps({"selected_images": names}))
        header = dict(format="SRAC_REGION_HEAD_ADAPT_V1", smoke=False, test_accessed=False,
                      protocol_sha256=initial.digest(protocol), source_sha256="a" * 64)
        (root / "prt_fit/manifest.json").write_text(json.dumps(header))
        (root / "prt_fit/summary.json").write_text(json.dumps({**header, "complete": False,
           "phase": "initial_Val", "updates_per_arm": {arm: 0 for arm in initial.ARMS}}))
        report = {**failed_report(), "protocol_sha256": header["protocol_sha256"],
                  "source_sha256": header["source_sha256"]}
        (root / "prt_fit/initial_val/result.json").write_text(json.dumps(report))
        journal = root / "prt_fit/initial_val/images.jsonl"
        journal.write_text("\n".join(json.dumps({"image": name, "index": i}) for i, name in enumerate(names)))
        return root, repo, protocol

    def test_exact_failed_protocol_and_val_order_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root, repo, protocol = self.fixture(folder)
            paths, hashes, source, names = initial.bind_failed_evidence(root, protocol, Path(folder) / "new", repo)
            self.assertEqual(hashes["protocol"], initial.digest(protocol))
            self.assertEqual(len(names), 450)
            self.assertEqual(source, "a" * 64)

    def test_refuse_other_protocol_even_same_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root, repo, protocol = self.fixture(folder)
            other = Path(folder) / "other.json"
            other.write_bytes(protocol.read_bytes())
            with self.assertRaisesRegex(ValueError, "Protocol must"):
                initial.bind_failed_evidence(root, other, Path(folder) / "new", repo)

    def test_refuse_output_inside_source_or_old_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root, repo, protocol = self.fixture(folder)
            for output in (root / "new", repo / "new"):
                with self.assertRaisesRegex(ValueError, "outside"):
                    initial.bind_failed_evidence(root, protocol, output, repo)

    def test_refuse_changed_source_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root, repo, protocol = self.fixture(folder)
            path = root / "prt_fit/summary.json"
            content = json.loads(path.read_text())
            content["source_sha256"] = "b" * 64
            path.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, "identity"):
                initial.bind_failed_evidence(root, protocol, Path(folder) / "new", repo)

    def test_refuse_already_trained_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root, repo, protocol = self.fixture(folder)
            path = root / "prt_fit/summary.json"
            content = json.loads(path.read_text())
            content["updates_per_arm"]["p"] = 1
            path.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, "zero optimizer"):
                initial.bind_failed_evidence(root, protocol, Path(folder) / "new", repo)

    def test_refuse_reordered_journal(self):
        with tempfile.TemporaryDirectory() as folder:
            root, repo, protocol = self.fixture(folder)
            path = root / "prt_fit/initial_val/images.jsonl"
            lines = path.read_text().splitlines()
            lines[0], lines[1] = lines[1], lines[0]
            path.write_text("\n".join(lines))
            with self.assertRaisesRegex(ValueError, "order"):
                initial.bind_failed_evidence(root, protocol, Path(folder) / "new", repo)


class ObserverTests(unittest.TestCase):
    def setup_functions(self, fail_index):
        parent = nn.Conv2d(2, 4, 1)
        front = SimpleNamespace(base=SimpleNamespace(clf_2=parent))
        heads = nn.ModuleDict({"p": copy.deepcopy(parent), "r": copy.deepcopy(parent), "t": nn.Linear(2, 4)})
        calls = []

        def feature_branches(feature, q, ids, count, head, output_size):
            calls.append("feature_branches")
            token = feature.flatten(2).transpose(1, 2)
            mass = torch.ones(1, 1)
            direct_logits = head(feature)
            token_logits = torch.nn.functional.linear(token, head.weight[:, :, 0, 0], head.bias)
            return direct_logits.softmax(1), token_logits.softmax(-1)

        def head_probability(arm, head, inputs, output_size):
            calls.append(arm)
            if arm == "p":
                return head(inputs["feature"]).softmax(1)
            logits = torch.nn.functional.linear(inputs["mean_tokens"], head.weight[:, :, 0, 0], head.bias)
            return logits.softmax(-1)

        current_index = [0]

        def initial_closeness(probability):
            return {arm: {"passed": not (current_index[0] == fail_index and arm == "p"),
                           "max_abs": .02 if arm == "p" else 0.0} for arm in initial.ARMS}

        def evaluate_three():
            for index in range(3):
                current_index[0] = index
                current_image = "scene" + str(index)
                feature = torch.ones(1, 2, 1, 1) * (index + 1)
                result = {"q": torch.ones(1, 1, 1), "ids": torch.zeros(1, 1, dtype=torch.long),
                          "count": 1, "shape": (1, 1)}
                inputs = {"feature": feature, "mean_tokens": feature.flatten(2).transpose(1, 2),
                          "mass": torch.ones(1, 1)}
                target = torch.full((1, 2, 2), 255, dtype=torch.long)
                low_logits = parent(feature)
                pixel, region = feature_branches(feature, result["q"], result["ids"], 1, parent, (2, 2))
                probability = {"pixel_grid": pixel, "region_mean": region,
                               "p": head_probability("p", heads["p"], inputs, (2, 2)),
                               "r": head_probability("r", heads["r"], inputs, (2, 2))}
                initial_closeness(probability)
            return calls

        return front, heads, calls, feature_branches, head_probability, initial_closeness, evaluate_three

    def test_actual_first_failure_snapshot_and_profile_hook_restored(self):
        with tempfile.TemporaryDirectory() as folder:
            front, heads, calls, branches, headprob, close, evaluate = self.setup_functions(1)
            record = {"protocol_sha256": "a" * 64, "original_source_sha256": "b" * 64,
                      "runtime_settings": {}, "val_order_sha256": "c" * 64,
                      "evidence_sha256": {}, "source_failed_run": "old", "helper_sha256": "d" * 64}
            observer = initial.SceneObserver(branches, headprob, close, heads, front, folder, record,
                                             ["scene0", "scene1", "scene2"])
            previous = sys.getprofile()
            before = {key: value.detach().clone() for key, value in heads.state_dict().items()}
            with self.assertRaises(initial.CapturedSceneStop):
                with observer.installed():
                    evaluate()
            self.assertIs(sys.getprofile(), previous)
            self.assertEqual(len(heads["p"]._forward_hooks), 0)
            self.assertEqual(calls, ["feature_branches", "p", "r"] * 2)
            scene = torch.load(Path(folder) / "scene.pt", weights_only=False)
            self.assertEqual(scene["image"], "scene1")
            self.assertEqual(scene["ordered_prefix_images"], ["scene0", "scene1"])
            self.assertFalse(scene["full_validation_evidence"])
            self.assertEqual(scene["failed_arms"], ["p"])
            self.assertFalse(scene["labels_saved"])
            self.assertEqual(scene["index_one_based"], 2)
            self.assertEqual(scene["provenance"]["helper_sha256"], "d" * 64)
            self.assertEqual(scene["tensor_sha256"]["feature"], initial.cpu_tensor_digest(scene["tensors"]["feature"]))
            self.assertTrue(torch.equal(scene["tensors"]["feature"], torch.full((1, 2, 1, 1), 2.0)))
            self.assertTrue(all(tensor.device.type == "cpu" for tensor in scene["tensors"].values()))
            self.assertTrue(all(torch.equal(value, heads.state_dict()[key]) for key, value in before.items()))

    def test_no_failure_is_not_fabricated(self):
        with tempfile.TemporaryDirectory() as folder:
            front, heads, calls, branches, headprob, close, evaluate = self.setup_functions(99)
            observer = initial.SceneObserver(branches, headprob, close, heads, front, folder, {},
                                             ["scene0", "scene1", "scene2"])
            with observer.installed():
                evaluate()
            self.assertEqual(len(calls), 9)
            self.assertFalse((Path(folder) / "scene.pt").exists())
            self.assertFalse(observer.record.get("scene_captured", False))

    def test_observer_restores_previous_profiler_on_original_exception(self):
        with tempfile.TemporaryDirectory() as folder:
            front, heads, calls, branches, headprob, close, evaluate = self.setup_functions(99)
            observer = initial.SceneObserver(branches, headprob, close, heads, front, folder, {}, [])
            events = []
            previous = sys.getprofile()
            callback = lambda frame, event, arg: events.append(event)
            sys.setprofile(callback)
            try:
                with self.assertRaisesRegex(RuntimeError, "device-style failure"):
                    with observer.installed():
                        raise RuntimeError("device-style failure")
                self.assertIs(sys.getprofile(), callback)
                self.assertEqual(len(heads["p"]._forward_hooks), 0)
            finally:
                sys.setprofile(previous)


class FullExecutionTests(unittest.TestCase):
    fixture = EvidenceTests.fixture
    def execute_mocked_full(self, folder, report, evaluator_error=False):
        root, repo, protocol = self.fixture(folder)
        names = json.loads(protocol.read_text())["selected_images"]
        events = []
        front = nn.Linear(2, 4)
        metric = nn.Linear(2, 2)
        decoder = nn.Linear(2, 4)
        for module in (front, metric, decoder):
            module.requires_grad_(False)

        class RecordingHeads(nn.ModuleDict):
            def to(self, device):
                events.append("heads_to")
                return super().to(device)

        class FakeOptimizer:
            def __init__(self, parameters, **kwargs):
                list(parameters)
                self.kwargs = kwargs
                events.append("optimizer")

            def step(self):
                raise AssertionError("Initialization must never perform an optimizer step")

        bundle = dict(front=front, metric=metric, decoders={"j1": decoder},
                      device=torch.device("cpu"), protocol_sha256=initial.digest(protocol),
                      val_names=names, runtime_settings={})
        core, evaluate, train = ModuleType("fakecore"), ModuleType("fakeevaluate"), ModuleType("faketrain")
        core.TRAIN_ARMS = initial.ARMS
        core.source_hash = lambda: "a" * 64

        def load_source(*args):
            events.append("load_source")
            return bundle

        def make_heads(front):
            events.append("make_heads")
            return RecordingHeads({arm: nn.Linear(2, 4) for arm in initial.ARMS})

        def parameter_checks(*args):
            events.append("initial_parameter_checks")
            return {"passed": True}

        def eval_heads(bundle, heads, output, limit, initial):
            self.assertEqual(limit, 0)
            self.assertTrue(initial)
            events.append("evaluate_heads")
            self.assertIsNone(sys.getprofile())
            self.assertFalse(heads["p"]._forward_hooks)
            output.mkdir()
            (output / "result.json").write_text(json.dumps(report))
            if evaluator_error:
                raise RuntimeError("original formal gate failed")
            return report

        def formal_assert(report, smoke):
            events.append("assert_formal_report")
            self.assertFalse(smoke)

        core.load_source, core.make_heads = load_source, make_heads
        core.initial_parameter_checks, core.assert_formal_report = parameter_checks, formal_assert
        evaluate.evaluate_heads = eval_heads
        train.FORMAL_STEPS = 5000
        train.validate_training_config = lambda *args: 5000
        train.validate_trainable_heads = lambda *args: events.append("validate_trainable_heads")
        pkg = ModuleType("bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1")
        pkg.core, pkg.evaluate, pkg.train = core, evaluate, train
        modules = {"bda_benchmark": ModuleType("bda_benchmark"),
                   "bda_benchmark.experiments": ModuleType("bda_benchmark.experiments"),
                   pkg.__name__: pkg}
        args = SimpleNamespace(repo=str(repo), failed_run=str(root), protocol=str(protocol),
                               output=str(Path(folder) / "full"), mode="full")
        with mock.patch.dict(sys.modules, modules), mock.patch.object(torch.optim, "AdamW", FakeOptimizer):
            result = initial.run(args)
        return result, events

    def test_full_original_gpu_operation_order_and_no_instrumentation(self):
        with tempfile.TemporaryDirectory() as folder:
            result, events = self.execute_mocked_full(folder, passed_report())
            self.assertEqual(events, ["load_source", "make_heads", "heads_to", "optimizer", "optimizer", "optimizer",
                                      "validate_trainable_heads", "initial_parameter_checks",
                                      "evaluate_heads", "assert_formal_report"])
            self.assertTrue(result["complete"])
            self.assertTrue(result["full_order_passed"])
            self.assertFalse(result["full_validation_evidence"])
            self.assertEqual(result["optimizer_steps"], 0)

    def test_full_failed_original_initial_gate_is_categorized_for_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            result, events = self.execute_mocked_full(folder, failed_report(), evaluator_error=True)
            self.assertFalse(result["complete"])
            self.assertFalse(result["full_order_passed"])
            self.assertEqual(result["failure_kind"], "initial_probability_gate")
            self.assertNotIn("assert_formal_report", events)

    def test_full_other_fail_does_not_authorize_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            report = failed_report()
            report["reproduction"]["passed"] = False
            result, events = self.execute_mocked_full(folder, report, evaluator_error=True)
            self.assertEqual(result["failure_kind"], "other")
            self.assertFalse(result["full_order_passed"])


if __name__ == "__main__":
    unittest.main()
