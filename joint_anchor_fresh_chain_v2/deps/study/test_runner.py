"""Synthetic lifecycle/worker tests. These are not real BRIGHT/CUDA evidence."""
from contextlib import contextmanager, redirect_stdout
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import engine
import legacy as bridge
import test_evaluate as evaluation_tests

FakeRuntime, TinyHead, _functions = evaluation_tests.FakeRuntime, evaluation_tests.TinyHead, evaluation_tests._functions


_spec = importlib.util.spec_from_file_location("_bright_anchor_runner_under_test", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def _json_write(path, data):
    Path(path).write_text(json.dumps(data, allow_nan=False), encoding="utf-8")


@contextmanager
def _lock(_path):
    yield


class PipelineContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "new_run"
        self.args = runner.parser().parse_args(["full", "--start", "fresh", "--config", "config.json", "--output", str(self.root)])
        self.legacy = SimpleNamespace(study=SimpleNamespace(
            exclusive_pipeline_lock=_lock, write_json=_json_write,
            read_json=lambda p: json.loads(Path(p).read_text())))
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def subprocess(self, command, **kwargs):
        self.calls.append(command)
        mode = command[command.index("--mode") + 1] if "--mode" in command else "cpu_contracts"
        if mode != "cpu_contracts":
            out = Path(command[command.index("--output") + 1]); out.mkdir()
            report = dict(complete=True, execution_complete=True, training_complete=True,
                          final_integrity_pending=False, smoke=mode == "smoke", test_accessed=False,
                          endpoint_step=22 if mode == "smoke" else 1500,
                          full_validation_evidence=mode == "train")
            _json_write(out / "summary.json", report)
        return SimpleNamespace(pid=123, stdout=iter([mode + " synthetic\n"]), wait=lambda: 0)

    def run_pipeline(self, popen=None):
        with patch.object(runner, "release_hash", return_value="test"), \
             patch.object(runner, "dependencies", return_value=(self.legacy, {})), \
             patch.object(runner.subprocess, "Popen", side_effect=popen or self.subprocess), \
             patch.object(runner, "print_report"), redirect_stdout(io.StringIO()):
            runner.pipeline(self.args)
        return json.loads((self.root / "workflow_state.json").read_text())

    def test_full_pipeline_requires_smoke_before_formal(self):
        state = self.run_pipeline()
        self.assertEqual(list(state["stages"]), ["cpu_contracts", "smoke", "train"])
        self.assertTrue(state["complete"] and state["full_validation_evidence"])
        self.assertFalse(state["test_accessed"])
        self.assertEqual(len(self.calls), 3)

    def test_check_does_not_launch_formal_train(self):
        self.args.command = "check"
        state = self.run_pipeline()
        self.assertEqual(list(state["stages"]), ["cpu_contracts", "smoke"])
        self.assertFalse(state["training_complete"] or state["full_validation_evidence"])

    def test_failure_stops_later_stages_and_keeps_traceback(self):
        def fail(command, **kw):
            value = self.subprocess(command, **kw)
            if "--mode" in command:
                value.wait = lambda: 1
            return value
        with self.assertRaisesRegex(RuntimeError, "smoke failed"):
            self.run_pipeline(fail)
        state = json.loads((self.root / "workflow_state.json").read_text())
        self.assertFalse(state["complete"])
        self.assertNotIn("train", state["stages"])
        self.assertEqual(state["stages"]["smoke"]["status"], "failed")
        self.assertTrue((self.root / "traceback.txt").is_file())

    def test_zero_exit_without_complete_evidence_is_not_success(self):
        def wrong(command, **kw):
            value = self.subprocess(command, **kw)
            if "--mode" in command:
                out = Path(command[command.index("--output") + 1])
                report = json.loads((out / "summary.json").read_text())
                report["final_integrity_pending"] = True
                _json_write(out / "summary.json", report)
            return value
        with self.assertRaisesRegex(ValueError, "lacks complete matching evidence"):
            self.run_pipeline(wrong)
        state = json.loads((self.root / "workflow_state.json").read_text())
        self.assertEqual(state["stages"]["smoke"]["status"], "failed")
        self.assertNotIn("train", state["stages"])

    def test_existing_run_is_preserved(self):
        self.root.mkdir(); (self.root / "keep.txt").write_text("original")
        with self.assertRaises(FileExistsError):
            self.run_pipeline()
        self.assertEqual((self.root / "keep.txt").read_text(), "original")
        self.assertFalse(self.calls)


class _TrainingHead(TinyHead):
    """Use the evaluator's tiny model, allowing train/eval and Reader naming."""
    def __init__(self, chunk_size=32):
        super().__init__()
        self.query_projection = self.projection
        del self.projection

    def forward(self, *, features, **unused):
        logits = self.query_projection(self.dropout(features))
        return {"logits": logits, "probabilities": logits.softmax(1)}


class _CPUState:
    """Explicit test-only adapter; production CUDA checks are not weakened."""
    def __init__(self, wrapped):
        self.wrapped = wrapped
    def __getattr__(self, name):
        return getattr(self.wrapped, name)
    def restore_rng(self, value, **ignored):
        return self.wrapped.restore_rng(value, require_cuda=False)


class _TorchWorkerProxy:
    # Only the worker's availability check is simulated. engine/evaluate use
    # real CPU torch, so no CUDA call or tensor allocation happens in tests.
    cuda = SimpleNamespace(is_available=lambda: True)
    def __getattr__(self, name):
        return getattr(torch, name)


class WorkerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        evaluation_tests.EvaluationContracts.setUpClass()
        cls.original = evaluation_tests.EvaluationContracts.original
        cls.helpers = evaluation_tests.EvaluationContracts.legacy
        original_path = Path(os.environ.get("BRIGHT_ORIGINAL_TOOL", "/workspace/bright-followup/lr_study_source/original_fix2"))
        cls.functions = _functions(original_path / "train.py", ("_lr", "_seed", "_checked_verify", "_restored_guard", "_names"),
            {"torch": torch, "np": np, "random": __import__("random"), "math": __import__("math")})

    def setUp(self):
        self.threads = torch.get_num_threads(); torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name) / "worker"
        self.args = runner.parser().parse_args(["worker", "--mode", "smoke", "--start", "fresh", "--output", str(self.root)])
        self.config = dict(chunk_size=32, seed=3407, lr=1e-4, weight_decay=.005, workers=0,
                           device="cuda:0", key_name="t128", topk=64)
        self.static = dict(start="fresh", start_step=0, config=self.config, old_files={},
                           checkpoint=None, source_sha256="synthetic", tool="synthetic")
        self.saved = None
        helpers = self.helpers
        self.legacy = SimpleNamespace(state=_CPUState(helpers.state), objectives=helpers.objectives, files={},
            study=SimpleNamespace(write_json=_json_write, read_json=lambda p: json.loads(Path(p).read_text()),
                                  digest=__import__("evaluate").digest, validate_example=helpers.study.validate_example,
                                  check_old_files=lambda static: None,
                                  check_inputs_against_checkpoint=lambda runtime, saved: None))
        self.original_train = SimpleNamespace(**vars(self.original), **vars(self.functions),
            JointRegionMemoryClassifier=_TrainingHead, _formal_runtime=lambda runtime, smoke: None)
        objectives = self.helpers.objectives
        class Runtime(FakeRuntime):
            def __init__(self, args, output):
                super().__init__(objectives)
                self.train_names = ["train%04d" % i for i in range(2918)]
                self.val_names = ["val%04d" % i for i in range(450)]
                self.input_ledger = {}
                self.guard_report = {"restored": True, "restore_conflicts": []}
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def verify(self): return {"passed": True}
            def batches(self, names, train, seed):
                for name in names:
                    yield (None, None, None, None, [name])
            def prepare(self, batch):
                return super().prepare(batch[4][0])
        self.Runtime = Runtime
        self.original_run = SimpleNamespace(check_inputs=lambda args: None, verify_release=lambda: True)
        self.common = SimpleNamespace(source_hash=lambda: "synthetic")

    def tearDown(self):
        self.tmp.cleanup(); torch.set_num_threads(self.threads)

    def invoke(self):
        with patch.object(runner, "release_hash", return_value="synthetic-release"), \
             patch.object(runner, "dependencies", return_value=(self.legacy, self.static)), \
             patch.object(bridge, "import_runtime", return_value=(_TorchWorkerProxy(), self.common,
                         self.original_train, self.original_run, self.Runtime)), \
             patch.object(bridge, "load_checkpoint", return_value=self.saved), redirect_stdout(io.StringIO()):
            return runner.worker(self.args)

    def checkpoint_source(self):
        """Synthetic Adam state with a known update counter, not historical evidence."""
        head = _TrainingHead(); opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.005)
        for parameter in head.parameters():
            parameter.grad = torch.ones_like(parameter)
        opt.step(); opt.zero_grad(set_to_none=True)
        for value in opt.state.values():
            value["step"].fill_(500)
        self.saved = dict(heads={"d": head.state_dict()}, optimizers={"d": opt.state_dict()},
                          protocol_info={"synthetic": True}, rng=self.helpers.state.capture_rng())
        self.args.start = "checkpoint500"
        runtime = self.Runtime(SimpleNamespace(), self.root)
        self.static.update(start="checkpoint500", start_step=500, checkpoint="synthetic.pt",
            manifest={"train_names": runtime.train_names, "val_names": runtime.val_names},
            rows=[{"image": name, "event": "event" + name[-1]} for name in runtime.train_names])

    def test_fresh_smoke_shared_prefix_then_twenty_matched_updates(self):
        report = self.invoke()
        self.assertTrue(report["complete"] and report["execution_complete"])
        self.assertTrue(report["smoke"])
        self.assertFalse(report["full_validation_evidence"])
        self.assertEqual(report["branch_step"], 2)
        self.assertEqual(report["endpoint_step"], 22)
        self.assertEqual(report["new_updates"], {"control": 20, "anchor": 20})
        self.assertFalse(report["test_accessed"])
        self.assertEqual(len((self.root / "prefix_train.jsonl").read_text().splitlines()), 2)
        self.assertEqual(len((self.root / "train.jsonl").read_text().splitlines()), 20)
        prefix = torch.load(self.root / "checkpoints/checkpoint_000002.pt", map_location="cpu", weights_only=False)
        final = torch.load(self.root / "checkpoints/checkpoint_000022.pt", map_location="cpu", weights_only=False)
        self.assertEqual(set(prefix["heads"]), {"control"})
        self.assertEqual(set(final["heads"]), {"control", "anchor"})
        for key in final["optimizers"]:
            for value in final["optimizers"][key]["state"].values():
                self.assertEqual(float(value["step"]), 22)
        for path in report["evaluations"].values():
            result = json.loads(Path(path).read_text())
            self.assertFalse(result["full_validation_evidence"] or result["primary_endpoint"])
            self.assertFalse(result["final_integrity_pending"])
            self.assertTrue(result["complete"] and result["smoke"])
            self.assertEqual(Path(path + ".sha256").read_text().split()[0], self.legacy.study.digest(path))

    def test_zero_anchor_weight_gives_identical_final_heads(self):
        self.args.weight = 0
        self.invoke()
        final = torch.load(self.root / "checkpoints/checkpoint_000022.pt", map_location="cpu", weights_only=False)
        state = self.legacy.state
        self.assertEqual(state.state_fingerprint(final["heads"]["control"]), state.state_fingerprint(final["heads"]["anchor"]))
        self.assertEqual(state.state_fingerprint(final["optimizers"]["control"]), state.state_fingerprint(final["optimizers"]["anchor"]))

    def test_checkpoint_smoke_restores_optimizer_counts_and_skips_prefix_forward(self):
        self.checkpoint_source()
        report = self.invoke()
        self.assertEqual(report["branch_step"], 500)
        self.assertEqual(report["endpoint_step"], 520)
        self.assertFalse((self.root / "prefix_train.jsonl").exists())
        rows = [json.loads(line) for line in (self.root / "train.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0]["image"], "train0500")
        self.assertEqual(rows[-1]["image"], "train0519")
        self.assertAlmostEqual(rows[0]["heads"]["control"]["lr"], self.helpers.state.lr_at_step(501, multiplier=.2))
        final = torch.load(self.root / "checkpoints/checkpoint_000520.pt", map_location="cpu", weights_only=False)
        for name in final["optimizers"]:
            self.assertTrue(all(float(value["step"]) == 520 for value in final["optimizers"][name]["state"].values()))

    def test_historical_loader_mismatch_stops_before_new_updates(self):
        self.checkpoint_source()
        self.static["rows"][0]["image"] = "unexpected-image"
        with self.assertRaisesRegex(ValueError, "Historical loader mismatch"):
            self.invoke()
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["new_updates"], {"control": 0, "anchor": 0})
        self.assertFalse((self.root / "train.jsonl").exists())

    def test_source_integrity_failure_leaves_no_completed_evidence(self):
        self.common.source_hash = lambda: "changed"
        with self.assertRaisesRegex(ValueError, "Study or original code changed"):
            self.invoke()
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertFalse(summary["complete"] or summary["full_validation_evidence"])
        self.assertTrue(summary["final_integrity_pending"])
        initial = json.loads((self.root / "initial_val/result.json").read_text())
        self.assertFalse(initial["complete"])
        self.assertTrue(initial["final_integrity_pending"])
        self.assertTrue((self.root / "traceback.txt").is_file())


if __name__ == "__main__":
    unittest.main()
