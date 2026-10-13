"""Independent standard-library tests for the four-Q to eight-soft chain.

No GPU, BRIGHT inputs or training are started. Scheduling tests use callbacks;
witness checks use concrete on-disk artifacts with the public witness schema.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import chain


TRAIN_MODULE = "bda_benchmark.experiments.damageformer_srac_q_factorial_v1.train_q"
GROUPS = ("a1", "adapter", "embedding", "region_classifier")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FullChainTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo with spaces"
        self.repo.mkdir()
        self.q_root = self.root / "q"
        self.q_root.mkdir()
        self.witness = self.root / "SAR_FREEZE_WITNESS_V1.py"
        self.witness.write_text("# synthetic witness release\n")
        self.python = "/example/python"
        self.train_source = self.repo / Path(*TRAIN_MODULE.split("."))
        self.train_source = self.train_source.with_suffix(".py")
        self.train_source.parent.mkdir(parents=True)
        self.train_source.write_text("# synthetic original training source\n")
        self.protocol = self.root / "protocol.json"
        self.protocol.write_text("{}\n")

    def tearDown(self):
        self.temporary.cleanup()

    def stage(self, *, smoke=False, arm="baseline"):
        scope, steps = ("smoke", 2) if smoke else ("formal", 2000)
        output = self.q_root / scope / ("train_" + arm)
        command = [self.python, "-u", "-m", TRAIN_MODULE,
                   "--protocol", str(self.protocol), "--arm", arm,
                   "--output", str(output), "--steps", str(steps), "--eval_every", str(steps),
                   "--batch", "2", "--accum", "8", "--workers", "4", "--lr", "0.0001",
                   "--aux_weight", "0.1", "--cue_gain", "0.1", "--amp", "bf16"]
        if smoke:
            command.append("--smoke")
        return dict(name=scope + "_train_" + arm, kind="train", arm=arm,
                    smoke=smoke, command=command, output=str(output))

    def instrument(self, plan):
        return chain.instrument_plan(plan, self.repo, self.witness, self.q_root, self.python)

    def make_witness(self, stage):
        folder = self.q_root / "frozen_witness" / stage["name"]
        stage.update(frozen_witness_output=str(folder), frozen_witness_script=str(self.witness))
        folder.mkdir(parents=True)
        smoke, steps = stage["smoke"], 2 if stage["smoke"] else 2000
        immutable = {name: True for name in
                     ("e0_files", "protocol", "q_code", "runtime", "frozen_downstream")}
        original = dict(format="SRAC_Q_FACTORIAL_V1", complete=True, arm=stage["arm"],
                        steps=steps, initial_a1_reproduction=None if smoke else True,
                        full_validation_evidence=not smoke, a1_unchanged=True,
                        downstream_unchanged=True, immutable=immutable,
                        smoke=smoke, test_accessed=False, formal_checkpoint="last.pth")
        summary = dict(complete=True, exact_frozen_checks=True, updates=steps, smoke=smoke,
                       original_summary=original, diagnostic_only=smoke, test_accessed=False)
        arguments = dict(protocol=str(self.protocol), arm=stage["arm"], output=stage["output"],
                         steps=steps, eval_every=steps, batch=2, accum=8, workers=4,
                         lr=0.0001, aux_weight=0.1, cue_gain=0.1, amp="bf16", smoke=smoke)
        declaration = dict(format="SAR_FROZEN_WITNESS_V1", script_sha256=digest(self.witness),
                           original_train_sha256=digest(self.train_source), arguments=arguments,
                           check_interval=20, original_checks_enabled=True,
                           changes_training_math=False, diagnostic_only=smoke,
                           old_checkpoint_reused=False, test_accessed=False)
        rows = [dict(event="baseline_captured", name=name, step=0, phase="initialization",
                     sha256=hashlib.sha256(name.encode()).hexdigest(), parameters=1, time=1)
                for name in GROUPS]
        rows += [dict(event="all_frozen_states_exact", step=0, phase=phase, time=2)
                 for phase in ("before_Val", "after_Val")]
        ticks = sorted({1, steps} | set(range(20, steps + 1, 20)))
        rows += [dict(event="all_frozen_states_exact", step=step,
                      phase="after_optimizer_step", time=3 + step)
                 for step in ticks]
        rows += [dict(event="all_frozen_states_exact", step=steps, phase=phase, time=4 + steps)
                 for phase in ("before_Val", "after_Val", "original_training_checks_passed")]
        (folder / "summary.json").write_text(json.dumps(summary))
        (folder / "declaration.json").write_text(json.dumps(declaration))
        (folder / "witness.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        return folder, summary

    def rewrite_summary(self, folder, value):
        (folder / "summary.json").write_text(json.dumps(value))

    def valid_q(self):
        return dict(complete=True, full_validation_evidence=True, immutable=True,
                    test_accessed=False)

    def valid_soft(self):
        return dict(complete=True, full_validation_evidence=True, test_accessed=False)

    def test_instrument_preserves_tail_and_input_plan(self):
        train = self.stage(smoke=True)
        preflight = dict(name="preflight", kind="preflight", command=["bash", "run.sh", "preflight"])
        evaluate = dict(name="smoke_val_baseline", kind="evaluate",
                        command=[self.python, "-u", "-m", TRAIN_MODULE.replace("train_q", "evaluate_q"),
                                 "--checkpoint", str(Path(train["output"]) / "last.pth")])
        plan = [preflight, train, evaluate]
        before = copy.deepcopy(plan)
        result = self.instrument(plan)
        self.assertEqual(plan, before)
        self.assertEqual(result[0], preflight)
        self.assertEqual(result[2], evaluate)
        command = result[1]["command"]
        expected = [self.python, "-u", str(self.witness), "child", "--repo", str(self.repo),
                    "--evidence", str(self.q_root / "frozen_witness" / train["name"]), "--"]
        self.assertEqual(command[:len(expected)], expected)
        self.assertEqual(command[len(expected):], train["command"][4:])
        self.assertEqual(result[1]["output"], train["output"])
        evidence = Path(command[command.index("--evidence") + 1])
        self.assertNotIn(Path(train["output"]).resolve(), evidence.resolve().parents)

    def test_all_eight_q_smoke_and_formal_train_stages_are_instrumented(self):
        stages = [self.stage(smoke=smoke, arm=arm) for smoke in (True, False)
                  for arm in ("baseline", "rgb_capacity", "shifted_sar", "sar_cue")]
        result = self.instrument(stages)
        self.assertEqual(len(result), 8)
        for previous, updated in zip(stages, result):
            self.assertEqual(updated["command"][-len(previous["command"][4:]):], previous["command"][4:])
            self.assertEqual(updated["command"][2], str(self.witness))
        evidence_paths = {row["command"][row["command"].index("--evidence") + 1] for row in result}
        self.assertEqual(len(evidence_paths), 8)

    def test_wrong_train_commands_are_rejected(self):
        bad_commands = ([self.python, "-u", "-m", "other.train", "--output", "fit"],
                        [self.python, "-m", TRAIN_MODULE],
                        ["bash", "run.sh", "train"], [])
        for command in bad_commands:
            with self.subTest(command=command):
                stage = self.stage()
                stage["command"] = command
                with self.assertRaises(Exception):
                    self.instrument([stage])

    def test_evidence_inside_training_output_is_rejected(self):
        stage = self.stage()
        bad_output = self.q_root / "frozen_witness"
        old_output = stage["output"]
        stage["output"] = str(bad_output)
        stage["command"][stage["command"].index(old_output)] = str(bad_output)
        with self.assertRaises(Exception):
            self.instrument([stage])

    def test_valid_smoke_and_formal_witnesses_have_real_hash_bindings(self):
        for smoke in (True, False):
            with self.subTest(smoke=smoke):
                stage = self.stage(smoke=smoke)
                folder, _ = self.make_witness(stage)
                record = chain.validate_witness(stage, self.q_root)
                serialized = json.dumps(record)
                self.assertIn(str(folder), serialized)
                self.assertIn(digest(folder / "summary.json"), serialized)

    def test_witness_failure_and_wrong_budget_rejected(self):
        stage = self.stage()
        folder, original = self.make_witness(stage)
        for field, value in (("complete", False), ("exact_frozen_checks", False),
                             ("updates", 1999), ("smoke", True),
                             ("diagnostic_only", True), ("test_accessed", True)):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed[field] = value
                self.rewrite_summary(folder, changed)
                with self.assertRaises(Exception):
                    chain.validate_witness(stage, self.q_root)
        self.rewrite_summary(folder, original)

    def test_original_training_freeze_and_validation_flags_rejected(self):
        stage = self.stage()
        folder, original = self.make_witness(stage)
        for field, value in (("complete", False), ("steps", 1999),
                             ("a1_unchanged", False), ("downstream_unchanged", False),
                             ("full_validation_evidence", False), ("immutable", {}),
                             ("immutable", {"e0_files": True, "runtime": False})):
            with self.subTest(field=field, value=value):
                changed = copy.deepcopy(original)
                changed["original_summary"][field] = value
                self.rewrite_summary(folder, changed)
                with self.assertRaises(Exception):
                    chain.validate_witness(stage, self.q_root)

    def test_smoke_witness_cannot_certify_full_validation(self):
        stage = self.stage(smoke=True)
        folder, record = self.make_witness(stage)
        record["original_summary"]["full_validation_evidence"] = True
        self.rewrite_summary(folder, record)
        with self.assertRaises(Exception):
            chain.validate_witness(stage, self.q_root)

    def test_missing_witness_summary_rejected(self):
        with self.assertRaises(Exception):
            chain.validate_witness(self.stage(), self.q_root)

    def test_witness_final_journal_must_represent_declared_end(self):
        stage = self.stage()
        folder, _ = self.make_witness(stage)
        path = folder / "witness.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for last in ({**rows[-1], "step": 1999},
                     {**rows[-1], "phase": "after_Val"},
                     {**rows[-1], "event": "baseline_captured"}):
            with self.subTest(last=last):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1] + [last]))
                with self.assertRaises(Exception):
                    chain.validate_witness(stage, self.q_root)

    def test_declaration_must_keep_original_checks_and_arguments(self):
        stage = self.stage()
        folder, _ = self.make_witness(stage)
        path = folder / "declaration.json"
        original = json.loads(path.read_text())
        for field, value in (("format", "other"), ("original_checks_enabled", False),
                             ("changes_training_math", True), ("old_checkpoint_reused", True),
                             ("diagnostic_only", True), ("script_sha256", "0" * 64)):
            with self.subTest(field=field):
                record = {**original, field: value}
                path.write_text(json.dumps(record))
                with self.assertRaises(Exception):
                    chain.validate_witness(stage, self.q_root)
        for field, value in (("arm", "sar_cue"), ("steps", 2), ("smoke", True)):
            with self.subTest(argument=field):
                record = copy.deepcopy(original)
                record["arguments"][field] = value
                path.write_text(json.dumps(record))
                with self.assertRaises(Exception):
                    chain.validate_witness(stage, self.q_root)

    def test_environment_restoration_does_not_mutate_parent(self):
        protocol = {"runtime_settings": {"environment": {
            "CUDA_VISIBLE_DEVICES": None, "NVIDIA_TF32_OVERRIDE": "0",
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": None},
            "cublas_workspace_config": ":4096:8"}}
        inherited = {"CUDA_VISIBLE_DEVICES": "1", "NVIDIA_TF32_OVERRIDE": "1",
                     "OMP_NUM_THREADS": "128", "MKL_NUM_THREADS": "128",
                     "CUBLAS_WORKSPACE_CONFIG": "old", "CUDA_LAUNCH_BLOCKING": "0",
                     "KEEP_ME": "kept", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        old_protocol, old_environment = copy.deepcopy(protocol), dict(inherited)
        restored = chain.restore_environment(protocol, inherited)
        self.assertEqual(protocol, old_protocol)
        self.assertEqual(inherited, old_environment)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", restored)
        self.assertNotIn("MKL_NUM_THREADS", restored)
        self.assertEqual(restored["NVIDIA_TF32_OVERRIDE"], "0")
        self.assertEqual(restored["OMP_NUM_THREADS"], "4")
        self.assertEqual(restored["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(restored["CUDA_LAUNCH_BLOCKING"], "1")
        self.assertEqual(restored["KEEP_ME"], "kept")
        self.assertEqual(restored["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")

    def test_environment_workspace_none_removes_inherited_setting(self):
        protocol = {"runtime_settings": {"environment": {name: None for name in
                    ("CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                    "cublas_workspace_config": None}}
        result = chain.restore_environment(protocol, {"CUBLAS_WORKSPACE_CONFIG": "old"})
        self.assertNotIn("CUBLAS_WORKSPACE_CONFIG", result)

    def test_invalid_environment_values_are_rejected_before_child_launch(self):
        empty_environment = {name: None for name in
                             ("CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
        for runtime in ({"environment": {**empty_environment, "OMP_NUM_THREADS": 4}},
                        {"environment": empty_environment, "cublas_workspace_config": 4096}):
            with self.subTest(runtime=runtime), self.assertRaises(Exception):
                chain.restore_environment({"runtime_settings": runtime}, {})

    def test_q_then_soft_order_and_phase(self):
        calls, state = [], {}
        q, soft = self.valid_q(), self.valid_soft()

        def q_call():
            self.assertEqual(state["phase"], "sar_q")
            calls.append("q")
            return q

        def soft_call():
            self.assertEqual(state["phase"], "sar_soft")
            calls.append("soft")
            return soft

        result = chain.run_serial(q_call, soft_call, state)
        self.assertEqual(calls, ["q", "soft"])
        self.assertEqual(state["phase"], "complete")
        self.assertEqual(result, {"q": q, "soft": soft})

    def test_q_exception_never_launches_soft(self):
        calls, state = [], {}

        def fail_q():
            calls.append("q")
            raise RuntimeError("Q training failed")

        def soft():
            calls.append("soft")
            return self.valid_soft()

        with self.assertRaises(Exception):
            chain.run_serial(fail_q, soft, state)
        self.assertEqual(calls, ["q"])
        self.assertNotEqual(state.get("phase"), "complete")

    def test_q_incomplete_or_missing_contract_never_launches_soft(self):
        bad_results = [{}, None, {**self.valid_q(), "complete": False},
                       {**self.valid_q(), "full_validation_evidence": False},
                       {**self.valid_q(), "immutable": {}},
                       {**self.valid_q(), "immutable": False},
                       {**self.valid_q(), "test_accessed": True}]
        for bad in bad_results:
            with self.subTest(result=bad):
                calls, state = [], {}

                def q():
                    calls.append("q")
                    return bad

                def soft():
                    calls.append("soft")
                    return self.valid_soft()

                with self.assertRaises(Exception):
                    chain.run_serial(q, soft, state)
                self.assertEqual(calls, ["q"])
                self.assertNotEqual(state.get("phase"), "complete")

    def test_json_booleans_required_for_q_contract(self):
        for field, value in (("complete", 1), ("full_validation_evidence", 1),
                             ("immutable", 1), ("test_accessed", 0)):
            with self.subTest(field=field):
                calls = []
                wrong = {**self.valid_q(), field: value}
                with self.assertRaises(Exception):
                    chain.run_serial(lambda: wrong, lambda: calls.append("soft"), {})
                self.assertEqual(calls, [])

    def test_soft_failure_does_not_mark_chain_complete(self):
        for bad in ({}, None, {**self.valid_soft(), "complete": False},
                    {**self.valid_soft(), "full_validation_evidence": False},
                    {**self.valid_soft(), "test_accessed": True}):
            with self.subTest(result=bad):
                calls, state = [], {}

                def q():
                    calls.append("q")
                    return self.valid_q()

                def soft():
                    calls.append("soft")
                    return bad

                with self.assertRaises(Exception):
                    chain.run_serial(q, soft, state)
                self.assertEqual(calls, ["q", "soft"])
                self.assertNotEqual(state.get("phase"), "complete")

    def test_soft_exception_preserves_q_before_failure(self):
        calls, state = [], {}

        def q():
            calls.append("q")
            return self.valid_q()

        def soft():
            calls.append("soft")
            raise RuntimeError("Soft validation failed")

        with self.assertRaises(Exception):
            chain.run_serial(q, soft, state)
        self.assertEqual(calls, ["q", "soft"])
        self.assertNotEqual(state.get("phase"), "complete")


if __name__ == "__main__":
    unittest.main()
