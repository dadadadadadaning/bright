"""CPU and synthetic tests only; no BRIGHT/GPU effectiveness evidence."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import torch
import evaluate_checkpoint as program


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resume_source = Path(os.environ["BRIGHT_RESUME_TEST_SOURCE"])
ablation_source = Path(os.environ["BRIGHT_ABLATION_TEST_SOURCE"])
archive = resume_source.parent / "bright_slot_memory_resume_v1.pyz"
if program.sha256(archive) != program.TRAINER_SHA256:
    raise RuntimeError("Tests require the exact pinned trainer archive")
sys.path.insert(0, str(archive))
import evidence
fixtures = load_module("synthetic_report_fixtures", resume_source / "test_evidence.py")
sys.path.insert(0, str(ablation_source))
import evaluation
eval_fixtures = load_module("cpu_evaluation_fixtures", ablation_source / "test_evaluation.py")


class ObserverContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.markers = io.StringIO()
        self.target = torch.zeros(1, 2, 2, dtype=torch.long)
        self.probability = torch.full((1, 4, 2, 2), .25)
        self.original_validator = evaluation._validate_probability
        class Runtime:
            def prepare(self, name):
                return SimpleNamespace(image=name)
        self.runtime = Runtime()

    def observe(self):
        return program.observed_evaluation(torch, self.runtime, evaluation,
                                            ["image98"], self.output, self.markers)

    def test_valid_original_validator_preserves_tensor_and_rng_and_restores_hooks(self):
        before = self.probability.clone()
        rng = torch.get_rng_state().clone()
        with patch.object(program, "probability_details", side_effect=AssertionError("extra read")):
            with self.observe() as state:
                self.runtime.prepare("image98")
                evaluation._validate_probability(self.probability, self.target, "d")
                self.assertEqual(state["image"], "image98")
                self.assertEqual(state["path"], "d")
        self.assertIs(evaluation._validate_probability, self.original_validator)
        self.assertNotIn("prepare", vars(self.runtime))
        self.assertTrue(torch.equal(before, self.probability))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse((self.output / "probability_failure.json").exists())

    def test_original_rejection_captures_exact_overshoot_and_reraises(self):
        self.probability.zero_()
        self.probability[:, 0] = 1
        self.probability[0, 0, 0, 0] += torch.finfo(torch.float32).eps
        before = self.probability.clone()
        with self.assertRaisesRegex(ValueError, "Invalid full-image probabilities"):
            with self.observe():
                self.runtime.prepare("image98")
                evaluation._validate_probability(self.probability, self.target, "grid_probq")
        failure = program.read_json(self.output / "probability_failure.json")
        self.assertEqual(failure["probability"]["above_one_count"], 1)
        self.assertEqual(failure["probability"]["above_one_excess"], torch.finfo(torch.float32).eps)
        self.assertEqual(failure["image"], "image98")
        self.assertEqual(failure["path"], "grid_probq")
        case = torch.load(self.output / "probability_failure.pt", weights_only=False)
        self.assertTrue(torch.equal(case["probability"], before))
        self.assertTrue(torch.equal(before, self.probability))
        self.assertIs(evaluation._validate_probability, self.original_validator)
        self.assertNotIn("prepare", vars(self.runtime))

    def test_nonfinite_case_is_json_safe_without_sanitizing_tensor(self):
        self.probability[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            with self.observe():
                self.runtime.prepare("image98")
                evaluation._validate_probability(self.probability, self.target, "f")
        failure = program.read_json(self.output / "probability_failure.json")
        self.assertEqual(failure["probability"]["nan_count"], 1)
        json.dumps(failure, allow_nan=False)
        self.assertTrue(torch.isnan(self.probability[0, 0, 0, 0]))

    def test_runtime_error_does_not_trigger_more_tensor_reads(self):
        def crash(*args):
            raise RuntimeError("simulated CUDA assert")
        with patch.object(evaluation, "_validate_probability", crash):
            with patch.object(program, "probability_details", side_effect=AssertionError("unsafe read")):
                with patch.object(torch, "save", side_effect=AssertionError("unsafe save")):
                    with self.assertRaisesRegex(RuntimeError, "simulated CUDA"):
                        with self.observe():
                            self.runtime.prepare("image98")
                            evaluation._validate_probability(self.probability, self.target, "b")
            self.assertIs(evaluation._validate_probability, crash)
        failure = program.read_json(self.output / "probability_failure.json")
        self.assertNotIn("probability", failure)
        self.assertNotIn("diagnostic_error", failure)
        self.assertIs(evaluation._validate_probability, self.original_validator)

    def test_loader_order_mismatch_restores_hooks_and_preserves_failure_location(self):
        with self.assertRaisesRegex(ValueError, "order differs"):
            with self.observe() as state:
                self.runtime.prepare("wrong_image")
        self.assertEqual(state["stage"], "prepare_begin")
        self.assertNotIn("prepare", vars(self.runtime))
        self.assertIs(evaluation._validate_probability, self.original_validator)

    def test_existing_instance_override_is_restored_after_failure(self):
        def previous(name):
            raise ValueError("failed prepare")
        self.runtime.prepare = previous
        with self.assertRaisesRegex(ValueError, "failed prepare"):
            with self.observe():
                self.runtime.prepare("image98")
        self.assertIs(self.runtime.prepare, previous)


class FinalizationContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.output = self.base / "eval1500_synthetic"
        self.names = [f"val_{i:04d}" for i in range(450)]
        self.report = fixtures.make_report(1500, "val", self.names)
        self.path, self.report = fixtures.write_report(
            self.output / "val_step_001500", self.report, self.names)
        self.source = self.base / "failed_source"
        self.source.mkdir()
        (self.source / "summary.json").write_text('{"phase":"failed","complete":false}\n')
        self.source_before = (self.source / "summary.json").read_bytes()
        self.checks = dict.fromkeys(program.REQUIRED_CHECKS, True)
        self.provenance = dict(source_root=str(self.source))

    def finalize(self):
        return program.finalize_fresh_report(self.path, self.report, output=self.output,
                                              names=self.names, evidence=evidence,
                                              checks=self.checks, provenance=self.provenance)

    def test_all_gates_finalize_only_fresh_report_and_preserve_pending_bytes(self):
        pending_bytes = self.path.read_bytes()
        finished = self.finalize()
        self.assertTrue(finished["complete"])
        self.assertEqual(finished["phase"], "standalone_evaluation_complete")
        self.assertFalse(finished["source_training_run_complete"])
        self.assertFalse(finished["primary_endpoint"])
        self.assertEqual(finished["optimizer_updates"], 0)
        self.assertEqual(self.path.with_name("result.pending.json").read_bytes(), pending_bytes)
        self.assertEqual((self.source / "summary.json").read_bytes(), self.source_before)
        evidence.validate_saved_report(self.path, finished, self.names, program.sha256, self.output)

    def test_each_integrity_gate_is_required_without_result_changes(self):
        before = self.path.read_bytes()
        for key in program.REQUIRED_CHECKS:
            with self.subTest(key=key):
                self.checks[key] = False
                with self.assertRaisesRegex(ValueError, "integrity checks"):
                    self.finalize()
                self.checks[key] = True
                self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(self.path.with_name("result.pending.json").exists())

    def test_partial_image_count_is_not_completed(self):
        self.report["image_count"] = 97
        fixtures.rehash_report(self.path, self.report)
        with self.assertRaisesRegex(ValueError, "image count"):
            self.finalize()
        self.assertFalse(program.read_json(self.path)["complete"])

    def test_wrong_image_order_in_journal_is_not_accepted_even_if_rehashed(self):
        journal = self.path.with_name("images.jsonl")
        lines = journal.read_text().splitlines(keepends=True)
        first = json.loads(lines[0])
        first["image"] = "val_wrong"
        lines[0] = json.dumps(first) + "\n"
        journal.write_text("".join(lines))
        fixtures.rehash_report(self.path, self.report)
        with self.assertRaises(ValueError):
            self.finalize()
        self.assertFalse(program.read_json(self.path)["complete"])

    def test_changed_objective_is_not_accepted_even_if_rehashed(self):
        journal = self.path.with_name("images.jsonl")
        lines = journal.read_text().splitlines(keepends=True)
        first = json.loads(lines[0])
        first["objectives"]["d"]["nll"] += .01
        lines[0] = json.dumps(first) + "\n"
        journal.write_text("".join(lines))
        fixtures.rehash_report(self.path, self.report)
        with self.assertRaises(ValueError):
            self.finalize()
        self.assertFalse(program.read_json(self.path)["complete"])

    def test_reject_output_within_old_run(self):
        self.provenance["source_root"] = str(self.base)
        with self.assertRaisesRegex(ValueError, "outside its source"):
            self.finalize()

    def test_reject_missing_report_checksum(self):
        self.path.with_name("result.json.sha256").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.finalize()

    def test_reject_test_or_smoke_as_formal_val(self):
        for field in ("test_accessed", "smoke"):
            with self.subTest(field=field):
                self.report[field] = True
                fixtures.rehash_report(self.path, self.report)
                with self.assertRaises(ValueError):
                    self.finalize()
                self.report[field] = False

    def test_metric_summary_preserves_undefined_iou(self):
        self.report["metrics"]["e"]["iou"][2] = None
        self.assertIsNone(program.concise_metrics(self.report)["e"]["miou4_pct"])


class OriginalEvaluatorCPUIntegration(unittest.TestCase):
    def test_actual_evaluator_six_heads_450_tiny_images_remains_smoke(self):
        eval_fixtures.EvaluationContracts.setUpClass()
        original, legacy = eval_fixtures.EvaluationContracts.original, eval_fixtures.EvaluationContracts.legacy
        runtime = eval_fixtures.TinyRuntime(legacy.objectives)
        runtime.val_names = [f"val{i}" for i in range(450)]
        heads = {a: eval_fixtures.TinyHead() for a in program.ARMS}
        factory = eval_fixtures.TinyFactory()
        optimizers = {a: torch.optim.AdamW(head.parameters()) for a, head in heads.items()}
        hashes = {a: legacy.state.state_fingerprint(o.state_dict()) for a, o in optimizers.items()}
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                folder = output / "val_step_001500"
                with redirect_stdout(io.StringIO()), patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer update")):
                    with program.observed_evaluation(torch, runtime, evaluation, runtime.val_names, output, io.StringIO()):
                        result = evaluation.evaluate(runtime, heads, runtime.val_names, folder,
                            step=1500, split="val", seed=3407, original_train=original,
                            legacy=legacy, factory=factory, primary_endpoint=False, smoke=False)
                self.assertEqual(result["image_count"], 450)
                self.assertEqual(len(factory.forwarded), 450 * 6)
                self.assertEqual(runtime.reference_calls, 450 * 7)
                self.assertTrue(result["smoke"])
                self.assertFalse(result["whole_val450_processed"])
                self.assertNotIn("prepare", vars(runtime))
                self.assertEqual(hashes, {a: legacy.state.state_fingerprint(o.state_dict()) for a, o in optimizers.items()})
                with self.assertRaises(ValueError):
                    program.finalize_fresh_report(folder / "result.json", result, output=output,
                        names=runtime.val_names, evidence=evidence,
                        checks=dict.fromkeys(program.REQUIRED_CHECKS, True),
                        provenance=dict(source_root=str(output.parent / "source")))
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
