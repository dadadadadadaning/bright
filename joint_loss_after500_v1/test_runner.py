"""Synthetic runner lifecycle tests; no BRIGHT data, weights or CUDA validation.

Production provenance/CUDA checks are mocked only inside these isolated tests.
The real runner still requires the original complete, hash-bound experiment.
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import continuation
import evaluation
import study


class PipelineLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.args = SimpleNamespace(command="full", output=str(self.base / "run"),
                                    train_root="synthetic-source", tool_dir="synthetic-tool")
        self.called = []
        self.codes = {}
        self.overrides = {}

    def tearDown(self):
        self.temp.cleanup()

    def fake_child(self, command, **kwargs):
        mode = command[command.index("--mode") + 1] if "--mode" in command else "cpu_contracts"
        self.called.append(mode)
        if mode != "cpu_contracts" and self.codes.get(mode, 0) == 0:
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            smoke = mode == "smoke"
            data = dict(complete=True, execution_complete=True, training_complete=True,
                        final_integrity_pending=False, smoke=smoke,
                        phase="complete_smoke" if smoke else "complete",
                        full_validation_evidence=not smoke, endpoint_step=520 if smoke else 1500,
                        test_accessed=False)
            data.update(self.overrides.get(mode, {}))
            study.write_json(output / "summary.json", data)
        return SimpleNamespace(pid=1234, stdout=iter(["synthetic child output\n"]),
                               wait=lambda: self.codes.get(mode, 0))

    def run_pipeline(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(study, "verify_release", return_value="synthetic"))
            stack.enter_context(patch.object(study, "validate_static", return_value={"meta": {"sha256": "source"}, "manifest": {"config": {}}}))
            stack.enter_context(patch.object(study.subprocess, "Popen", side_effect=self.fake_child))
            stack.enter_context(patch.object(study, "print_report"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return study.pipeline(self.args)

    def test_smoke_failure_stops_formal_and_records_failure(self):
        self.codes["smoke"] = 1
        with self.assertRaisesRegex(RuntimeError, "smoke failed"):
            self.run_pipeline()
        self.assertEqual(self.called, ["cpu_contracts", "smoke"])
        state = study.read_json(Path(self.args.output) / "workflow_state.json")
        self.assertFalse(state["complete"])
        self.assertEqual(state["stages"]["smoke"]["status"], "failed")
        self.assertNotIn("train", state["stages"])

    def test_zero_exit_without_complete_evidence_stops_formal(self):
        self.overrides["smoke"] = {"complete": False}
        with self.assertRaises(ValueError):
            self.run_pipeline()
        self.assertNotIn("train", self.called)

    def test_check_mode_is_not_formal_validation(self):
        self.args.command = "check"
        self.run_pipeline()
        state = study.read_json(Path(self.args.output) / "workflow_state.json")
        self.assertTrue(state["complete"])
        self.assertFalse(state["full_validation_evidence"])
        self.assertFalse(state["training_complete"])
        self.assertEqual(self.called, ["cpu_contracts", "smoke"])

    def test_full_mode_requires_and_finishes_both_separate_children(self):
        self.run_pipeline()
        self.assertEqual(self.called, ["cpu_contracts", "smoke", "train"])
        state = study.read_json(Path(self.args.output) / "workflow_state.json")
        self.assertTrue(state["complete"])
        self.assertTrue(state["training_complete"])
        self.assertTrue(state["full_validation_evidence"])
        self.assertFalse(state["test_accessed"])
        smoke = study.read_json(Path(self.args.output) / "smoke/summary.json")
        train = study.read_json(Path(self.args.output) / "train/summary.json")
        self.assertEqual((smoke["endpoint_step"], train["endpoint_step"]), (520, 1500))

    def test_formal_pipeline_rejects_smoke_mislabeled_as_formal(self):
        self.overrides["train"] = {"smoke": True, "endpoint_step": 520}
        with self.assertRaises(ValueError):
            self.run_pipeline()
        state = study.read_json(Path(self.args.output) / "workflow_state.json")
        self.assertFalse(state["complete"])
        self.assertFalse(state["full_validation_evidence"])

    def test_formal_pipeline_rejects_pending_integrity(self):
        self.overrides["train"] = {"final_integrity_pending": True}
        with self.assertRaises(ValueError):
            self.run_pipeline()

    def test_duplicate_active_pipeline_is_refused_before_child(self):
        with study.exclusive_pipeline_lock(self.base / "active_pipeline.lock"):
            with self.assertRaisesRegex(RuntimeError, "holds the run lock"):
                self.run_pipeline()
        self.assertEqual(self.called, [])

    def test_existing_output_is_not_overwritten(self):
        root = Path(self.args.output)
        root.mkdir()
        sentinel = root / "preserve.txt"
        sentinel.write_text("saved experiment")
        with self.assertRaises(FileExistsError):
            self.run_pipeline()
        self.assertEqual(sentinel.read_text(), "saved experiment")
        self.assertEqual(self.called, [])

    def test_bad_source_aborts_before_outputs_or_subprocesses(self):
        with patch.object(study, "verify_release", return_value="synthetic"), \
                patch.object(study, "validate_static", side_effect=ValueError("source mismatch")), \
                patch.object(study.subprocess, "Popen") as child:
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                study.pipeline(self.args)
        child.assert_not_called()
        self.assertFalse(Path(self.args.output).exists())

    def test_report_handles_pipeline_before_child_report_exists(self):
        root = Path(self.args.output)
        root.mkdir()
        study.write_json(root / "workflow_state.json", {"current_stage": "cpu_contracts", "complete": False})
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            study.print_report(root)
        self.assertIn("尚无训练子阶段报告", buffer.getvalue())

    def test_report_marks_missing_evaluation_without_claiming_result(self):
        root = Path(self.args.output)
        (root / "train").mkdir(parents=True)
        study.write_json(root / "train/summary.json", {
            "phase": "initial_val", "complete": False,
            "evaluations": {"500": str(root / "not_written_yet/result.json")}})
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            study.print_report(root)
        self.assertIn("500", buffer.getvalue())
        self.assertIn("not_written_yet", buffer.getvalue())


class WorkerLifecycle(unittest.TestCase):
    """Exercise runner sequencing with tiny CPU fixtures and mocked evaluations."""

    def test_smoke_restores_rng_and_loader_then_never_claims_formal(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "smoke"
            rows = [{"image": "train_%04d" % i, "event": "event", "step": i + 1} for i in range(1500)]
            names = [row["image"] for row in rows]
            static = dict(root="synthetic-root", checkpoint="synthetic500", old_files={}, rows=rows,
                          manifest={"train_names": names, "val_names": ["v0", "v1"]},
                          meta={"sha256": "synthetic-checkpoint", "head_hashes": {"d": "fixed-head"}},
                          reference={})
            initial_random = random.Random(1234).getstate()
            expected_first_random = random.Random(1234).random()
            saved = dict(config={"lr": 1e-4, "seed": 3407}, protocol_info={"synthetic": True},
                         rng={"python": initial_random})
            heads = {name: torch.nn.Linear(1, 1, bias=False) for name in continuation.BINDINGS}
            optimizers = {name: SimpleNamespace(state_dict=lambda: {}) for name in heads}
            restores, training, evaluations = [], [], []
            consumed = []

            class Runtime:
                def __init__(self, args, result_path):
                    self.protocol_info = saved["protocol_info"]
                    self.train_names, self.val_names = names, ["v0", "v1"]
                    self.input_ledger, self.device = {}, "cpu"

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def batches(self, image_names, **kwargs):
                    for name in image_names:
                        consumed.append(name)
                        yield name

                def prepare(self, batch):
                    return SimpleNamespace(image=batch, event="event")

            common = SimpleNamespace(module_hash=lambda _: "fixed-head", source_hash=lambda: study.EXPECTED_SOURCE)
            original_train = SimpleNamespace(
                _formal_runtime=lambda *args: None, _checked_verify=lambda _: {"passed": True},
                _restored_guard=lambda _: {"restored": True},
                _lr=lambda step, horizon, maximum: continuation.lr_at_step(step, maximum, horizon))
            original_run = SimpleNamespace(check_inputs=lambda _: None)

            def restore(state, **kwargs):
                restores.append(len(training))
                random.setstate(state["python"])

            def evaluate(runtime, active_heads, image_names, path, **kwargs):
                # Deliberately disturb the RNG: runner must restore after initial
                # Val + fixed Train even if an evaluator implementation did so.
                random.random()
                evaluations.append((kwargs["step"], kwargs["split"], list(image_names)))
                path.mkdir()
                metrics = {name: {"foreground_miou": .5} for name in ("grid_probq", *heads)}
                record = dict(step=kwargs["step"], split=kwargs["split"], metrics=metrics,
                              names_sha256="same-fixed-list", order_sha256="same-fixed-order")
                study.write_json(path / "result.json", record)
                return record

            def train_step(active_heads, active_optimizers, example, runtime, step, lr):
                training.append((step, example.image, random.random(), len(consumed)))
                return {name: {"reader_grad_abs_sum": 1., "grad_norm": .1} for name in heads}

            def save_checkpoint(path, active_heads, *args):
                step = args[1]
                target = path / ("checkpoint_%06d.pt" % step)
                torch.save({"heads": {name: head.state_dict() for name, head in active_heads.items()}}, target)
                return target, {"sha256": study.digest(target), "head_hashes": {name: "fixed-head" for name in heads}}

            args = SimpleNamespace(mode="smoke", output=str(output))
            with ExitStack() as stack:
                patches = [
                    patch.object(study, "verify_release", return_value="synthetic"),
                    patch.object(study, "validate_static", return_value=static),
                    patch.object(study, "import_original", return_value=(torch, common, original_train, original_run, Runtime)),
                    patch.object(study, "load_saved", return_value=saved),
                    patch.object(study, "check_inputs_against_checkpoint"),
                    patch.object(study, "check_old_files"),
                    patch.object(study, "package_fingerprint", return_value="fixed-driver"),
                    patch.object(study, "validate_example", return_value={"d": {"supported_slot_queries": 1}}),
                    patch.object(study, "save_checkpoint", side_effect=save_checkpoint),
                    patch.object(torch.cuda, "is_available", return_value=True),
                    patch.object(continuation, "fork_heads_and_optimizers", return_value=(heads, optimizers, {})),
                    patch.object(continuation, "raw_batch_identity", side_effect=lambda batch: batch),
                    patch.object(continuation, "restore_rng", side_effect=restore),
                    patch.object(continuation, "train_matched_step", side_effect=train_step),
                    patch.object(evaluation, "evaluate", side_effect=evaluate),
                    patch.object(evaluation, "print_table"),
                    patch.object(evaluation, "metriccontrast", return_value={}),
                ]
                for item in patches:
                    stack.enter_context(item)
                stack.enter_context(redirect_stdout(io.StringIO()))
                result = study.worker(args)
            self.assertEqual(restores[:2], [0, 0])
            self.assertEqual(training[0], (501, names[500], expected_first_random, 501))
            self.assertEqual([row[0] for row in training], list(range(501, 521)))
            self.assertEqual(evaluations[0][:2], (500, "val"))
            self.assertEqual(evaluations[1][:2], (500, "fixed_train"))
            self.assertEqual(evaluations[-1][:2], (520, "fixed_train"))
            self.assertEqual(evaluations[1][2], evaluations[-1][2])
            self.assertTrue(result["complete"])
            self.assertTrue(result["smoke"])
            self.assertFalse(result["full_validation_evidence"])
            self.assertEqual(result["suffix_updates_per_branch"], {name: 20 for name in heads})
            for path in result["evaluations"].values():
                report = study.read_json(path)
                self.assertFalse(report["primary_endpoint"])
                self.assertFalse(report["full_validation_evidence"])


if __name__ == "__main__":
    unittest.main()
