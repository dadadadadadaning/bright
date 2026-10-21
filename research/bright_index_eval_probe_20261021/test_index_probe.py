"""CPU/synthetic contracts, not reproduction of the server CUDA failure."""
import copy
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "bright_index_eval2000_v1.pyz"
sys.path.insert(0, str(ARCHIVE))
sys.path.append(os.environ["BRIGHT_ABLATION_TEST_SOURCE"])
import torch
import index_eval_probe as driver
from index_observer import IndexTrace, IndexContractError, compact_cpu, track_evaluation


class IndexTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.position = dict(index=30, image="target", active_arm="b", stage="forward")
        self.trace = IndexTrace(self.root, self.position)

    def test_expanded_gather_preserves_bytes_rng_and_output(self):
        source = torch.arange(24.).reshape(1, 6, 4)
        index = torch.tensor([[[0], [5], [2]]]).expand(1, 3, 4)
        expected, rng = source.gather(1, index), torch.get_rng_state()
        before = source.clone(), index.clone()
        with self.trace:
            actual = source.gather(1, index)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(source, before[0]) and torch.equal(index, before[1]))
        self.assertEqual(compact_cpu(index).shape, (1, 3, 1))
        record = json.loads((self.root / "target_indices.jsonl").read_text().splitlines()[0])
        self.assertEqual(record["indices"][0]["maximum"], 5)
        self.assertTrue(record["callsites"])

    def test_invalid_gather_stops_before_operator_and_saves_owned_cpu_case(self):
        source, index = torch.ones(1, 4), torch.tensor([[4]])
        with self.assertRaises(IndexContractError):
            with self.trace:
                source.gather(1, index)
        self.assertFalse(self.trace.first_failure["operator_called"])
        index.zero_()
        with mock.patch("index_observer.compact_cpu", side_effect=AssertionError("late read")):
            self.trace.save()
        case = torch.load(self.root / "index_failure_cpu.pt", weights_only=False)
        self.assertEqual(case["indices"][0]["compact"].item(), 4)
        self.assertEqual(case["source_cpu"].device.type, "cpu")

    def test_negative_gather_is_rejected(self):
        with self.assertRaises(IndexContractError):
            with self.trace:
                torch.ones(2).gather(0, torch.tensor([-1]))

    def test_advanced_negative_index_is_legal(self):
        source, index = torch.arange(6.), torch.tensor([-6, -1, 2])
        with self.trace:
            result = source[index]
        self.assertTrue(torch.equal(result, torch.tensor([0., 5., 2.])))
        self.assertIsNone(self.trace.first_failure)

    def test_index_select_int32_and_scatter_add(self):
        source = torch.arange(5.)
        with self.trace:
            selected = torch.index_select(source, 0, torch.tensor([0, 4], dtype=torch.int32))
            result = torch.zeros(5).scatter_add_(0, torch.tensor([1, 1, 3]), torch.ones(3))
        self.assertTrue(torch.equal(selected, torch.tensor([0., 4.])))
        self.assertTrue(torch.equal(result, torch.tensor([0., 2., 0., 1., 0.])))

    def test_prefix_does_not_copy_indices(self):
        self.position["index"] = 0
        source, index = torch.ones(2), torch.tensor([1])
        with mock.patch("index_observer.compact_cpu", side_effect=AssertionError("prefix CPU copy")):
            with self.trace:
                source.gather(0, index)
        self.assertEqual(self.trace.snapshot_count, 0)

    def test_original_operator_error_is_preserved(self):
        failure = RuntimeError("synthetic CUDA failure")
        fake = mock.Mock(_schema=torch.ops.aten.gather.default._schema, side_effect=failure)
        with self.assertRaises(RuntimeError) as caught:
            self.trace.__torch_dispatch__(fake, (), (torch.ones(2), 0, torch.tensor([1])))
        self.assertIs(caught.exception, failure)
        self.assertTrue(self.trace.first_failure["operator_called"])
        self.assertIsNotNone(self.trace.failure_case)

    def test_real_classifier_output_and_state_unchanged(self):
        package = types.ModuleType("_index_probe_reference")
        package.__path__ = [str(Path(os.environ["BRIGHT_ABLATION_REFERENCE"]) / "deps/original")]
        sys.modules[package.__name__] = package
        module = importlib.import_module(package.__name__ + ".test_model")
        inputs = module.fixture()
        model = module.JointRegionMemoryClassifier(hidden=16, heads=4, chunk_size=2)
        with torch.no_grad():
            model.joint_head[-1].weight.normal_(0, .1)
        before = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters())
        original_opt = copy.deepcopy(optimizer.state_dict())
        with torch.no_grad():
            reference = model(**inputs)["probabilities"]
            with self.trace:
                observed = model(**inputs)["probabilities"]
        self.assertTrue(torch.equal(reference, observed))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertEqual(original_opt, optimizer.state_dict())
        self.assertGreater(self.trace.index_count, 0)


class EvaluatorTests(unittest.TestCase):
    def run_case(self, fail):
        import test_evaluation as fixtures
        fixtures.EvaluationContracts.setUpClass()
        fixture = fixtures.EvaluationContracts()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.heads.update(r=copy.deepcopy(fixture.heads["d"]), l=copy.deepcopy(fixture.heads["d"]))
        runtime = fixture.runtime()
        with tempfile.TemporaryDirectory() as temp:
            position = dict(index=-1)
            trace = IndexTrace(temp, position, 0)
            original_forward = fixture.factory.forward
            error = RuntimeError("synthetic B failure")
            def forward(head, example, arm, support):
                if fail and arm == "b":
                    raise error
                return original_forward(head, example, arm, support)
            fixture.factory.forward = forward
            state = fixture.legacy.state
            original_restore = state.restore_rng
            with mock.patch.object(state, "restore_rng", wraps=original_restore) as restore:
                with track_evaluation(runtime, fixture.factory, fixtures.evaluation,
                                      runtime.val_names, state, trace):
                    if fail:
                        with self.assertRaises(RuntimeError) as caught:
                            fixture.run_evaluation(runtime)
                        self.assertIs(caught.exception, error)
                        self.assertEqual(restore.call_count, 0)
                    else:
                        fixture.run_evaluation(runtime)
                        self.assertEqual(restore.call_count, 1)
                self.assertIs(state.restore_rng, restore)
            self.assertIs(state.restore_rng, original_restore)
            self.assertNotIn("prepare", vars(runtime))
            self.assertIs(fixture.factory.forward, forward)
            self.assertEqual(trace.rng_restore_skipped, fail)
            if fail:
                self.assertEqual(trace.first_failure["position"]["active_arm"], "b")
                self.assertEqual(trace.first_failure["position"]["stage"], "forward")

    def test_original_evaluator_success_restores_rng_and_hooks(self):
        self.run_case(False)

    def test_original_evaluator_failure_keeps_first_error_and_skips_rng(self):
        self.run_case(True)


class ProtocolTests(unittest.TestCase):
    def test_launcher_pins_worker_and_sets_blocking_before_process_start(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            with (base / "lock").open("a+") as stream:
                with mock.patch.object(driver, "BASE", base), \
                     mock.patch.object(driver, "leases", return_value=contextlib.nullcontext([stream])), \
                     mock.patch.object(driver, "prepared", return_value=dict(evidence_files={})), \
                     mock.patch.object(sys, "argv", [str(ARCHIVE)]), \
                     mock.patch.object(driver.subprocess, "Popen", return_value=types.SimpleNamespace(pid=123)) as popen:
                    self.assertEqual(driver.launch(), 0)
                argv = popen.call_args.args[0]
                options = popen.call_args.kwargs
                self.assertEqual(options["env"]["CUDA_LAUNCH_BLOCKING"], "1")
                self.assertEqual(argv[4], "worker")
                copied = Path(argv[3])
                self.assertEqual(copied.read_bytes(), ARCHIVE.read_bytes())
                self.assertTrue(copied.parent.name.startswith("index_eval2000_"))
                self.assertEqual(json.loads((copied.parent / "summary.json").read_text())["optimizer_updates"], 0)

    def test_only_launch_blocking_difference_is_allowed(self):
        before = dict(execution_settings=dict(cuda_launch_blocking=None, precision="fp32"), source="same")
        actual = copy.deepcopy(before)
        actual["execution_settings"]["cuda_launch_blocking"] = "1"
        self.assertEqual(driver.validate_protocol(actual, before)["diagnostic"], "1")
        actual["source"] = "different"
        with self.assertRaises(ValueError):
            driver.validate_protocol(actual, before)
        with self.assertRaises(ValueError):
            driver.validate_protocol(before, before)

    def test_bundle_preserves_recorded_continuation_helpers(self):
        with zipfile.ZipFile(ARCHIVE) as archive:
            ledger = json.loads(archive.read("INDEX_PROBE_BASE.json"))
            for name, digest in ledger["files"].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest, name)

    def test_preflight_binds_checkpoint2000_failure_and_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = ["image_%s" % i for i in range(450)]
            names[30] = driver.TARGET_IMAGE
            context = dict(manifest=dict(val_names=names), parent=dict(step=2000), evidence_files={})
            (root / "trainer.pyz").write_bytes(b"synthetic")
            (root / "traceback.txt").write_text("synthetic traceback")
            driver.write_json(root / "route_observer_summary.json", dict(calls=1, failures=0))
            driver.write_json(root / "evaluation_failure.json", dict(index=30, image=driver.TARGET_IMAGE,
                              path="b", error="RuntimeError: device-side assert triggered"))
            status = dict(phase="failed", last_complete_step=2000, failure_phase="evaluate_val_2000",
                          new_updates_per_arm={a: 500 for a in driver.ARMS})
            driver.write_json(root / "summary.json", status)
            driver.write_json(root / "manifest.json", dict(val_names=names))
            (root / "val_step_002000").mkdir()
            journal = root / "val_step_002000/images.jsonl"
            journal.write_text("".join(json.dumps(dict(image=n)) + "\n" for n in names[:30]))
            with mock.patch.object(driver, "FAILURE_TRAINER_SHA", driver.sha256(root / "trainer.pyz")):
                driver.check_failure(context, root)
                context["parent"]["step"] = 1500
                with self.assertRaises(ValueError):
                    driver.check_failure(context, root)
                context["parent"]["step"] = 2000
                journal.write_text("")
                with self.assertRaisesRegex(ValueError, "first30"):
                    driver.check_failure(context, root)


if __name__ == "__main__":
    unittest.main()
