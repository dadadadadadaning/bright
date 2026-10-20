"""CPU wiring tests for formal continuation; no CUDA/training-quality claim.

Dataset/runtime and evaluation computation are synthetic. Original arm updates,
AdamW/RNG restoration and continuation checkpoint serialization remain real.
"""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
ADDON = ROOT / "slot_memory_ablation_v1"
sys.path.append(str(ADDON))
import ablation_runner
import resume_train
from inspect_failure import sha256


class CPUState:
    def __init__(self):
        reference = os.environ.get("BRIGHT_ABLATION_REFERENCE")
        if not reference:
            raise RuntimeError("BRIGHT_ABLATION_REFERENCE is required")
        path = Path(reference) / "deps/helpers/continuation.py"
        spec = importlib.util.spec_from_file_location("_resume_integration_state", path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        self.state_fingerprint = original.state_fingerprint
        self.lr_at_step = original.lr_at_step

    def capture_rng(self):
        return dict(
            python=copy.deepcopy(random.getstate()),
            numpy=copy.deepcopy(np.random.get_state()),
            torch_cpu=torch.get_rng_state().clone(), torch_cuda=[],
        )

    def restore_rng(self, saved, *, require_cuda=False):
        if saved["torch_cuda"]:
            raise ValueError("Synthetic CPU fixture has no CUDA RNG")
        random.setstate(saved["python"])
        np.random.set_state(saved["numpy"])
        torch.set_rng_state(saved["torch_cpu"])


def pixel_nll(probability, target):
    labels = target.long().reshape(-1).clone()
    valid = labels != 255
    table = probability.permute(0, 2, 3, 1).reshape(-1, 4)
    return F.nll_loss(table[valid].clamp_min(1e-8).log(), labels[valid])


def auxiliary_loss(probability, target):
    return 0.01 * probability.square().mean()


def fixture_loader(*args, **kwargs):
    raise AssertionError("Fixture runtime owns the ordered loader")


def fixture_parse(*args, **kwargs):
    raise AssertionError("Fixture runtime replaces batch parsing")


class FailBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        raise RuntimeError("synthetic B backward failure")


class TinyHead(torch.nn.Module):
    def __init__(self, chunk_size=32):
        super().__init__()
        self.query_projection = torch.nn.Linear(3, 4)
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, features):
        return self.dropout(self.query_projection(features)).T.reshape(1, 4, 2, 2).softmax(1)


class TinyFactory:
    def __init__(self, runtime, seed=3407):
        self.runtime = runtime
        runtime.factory = self

    def prepare(self, example, arms):
        return {arm: None for arm in arms}

    def forward(self, head, example, arm, support):
        self.runtime.active_arm = arm
        features = example.features * (1 + 0.05 * self.runtime.fixture.arms.index(arm))
        return {"probabilities": head(features)}

    def provenance(self):
        return {"fixture": "synthetic CPU supports"}


class TinyRuntime:
    def __init__(self, fixture, args, output, fail=False):
        self.fixture, self.args, self.output, self.fail = fixture, args, Path(output), fail
        self.device = torch.device("cpu")
        self.protocol_info = copy.deepcopy(fixture.manifest["protocol_info"])
        self.input_ledger = {fixture.input_file: sha256(fixture.input_file)}
        self.deps = SimpleNamespace(
            region_nll=pixel_nll, lovasz_softmax=auxiliary_loss,
            full_image_loader=fixture_loader, parse_batch=fixture_parse,
        )
        self.train_names, self.val_names = fixture.names, fixture.val_names
        self.guard_report = dict(restored=True, failures=[], restore_conflicts=[])
        self.prepared, self.loader_requests = [], []
        self.active_arm = self.image = None
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.closed = True
        return False

    def evaluation_batches(self, names, seed):
        names = list(names)
        self.loader_requests.append(names)
        # Catch restoration before loader setup, which would shift training RNG.
        torch.rand(7)
        random.random()
        np.random.random()
        return [[name] for name in names]

    def prepare(self, batch):
        self.image = str(batch[0])
        self.prepared.append(self.image)
        return SimpleNamespace(
            image=self.image, event="synthetic",
            features=self.fixture.features[self.image],
            target=torch.tensor([[[0, 1], [2, 255]]]),
        )

    def loss(self, probability, target):
        value = self.deps.region_nll(probability, target) + auxiliary_loss(probability, target)
        if self.fail and self.image == self.fixture.order[1002] and self.active_arm == "b":
            return FailBackward.apply(value)
        return value


class ResumeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "original"
        self.train = self.root / "seed_3407/train"
        self.train.mkdir(parents=True)
        self.seed, self.arms = 3407, ("e", "f", "b", "d", "r", "l")
        self.state = CPUState()
        self.input_file = self.root / "synthetic_input.txt"
        self.input_file.write_text("Synthetic CPU evidence only.\n")
        self.policy_file = self.root / "source_policy.json"
        self.write(self.policy_file, {"numeric_policy": {
            "num_threads": 1, "num_interop_threads": torch.get_num_interop_threads(),
        }})
        self.names = ["synthetic_train_%04d" % n for n in range(1307)]
        self.val_names = ["synthetic_val_%04d" % n for n in range(450)]
        self.fixed_names = self.names[:2]
        self.order = [
            str(batch[0]) for batch in torch.utils.data.DataLoader(
                self.names, batch_size=1, shuffle=True, num_workers=0,
                generator=torch.Generator().manual_seed(self.seed),
            )
        ]
        generator = torch.Generator().manual_seed(81)
        self.features = {name: torch.randn(4, 3, generator=generator) for name in self.names}
        self.config = dict(
            repo=str(self.base / "repo"), protocol=str(self.policy_file),
            bank_root=str(self.base / "banks"), values=str(self.base / "values"),
            device="cpu", chunk_size=32, lr=1e-4, weight_decay=0.005,
            output=str(self.train),
        )
        self.plan = dict(
            format="BRIGHT_SLOT_MEMORY_ABLATION_V1", arms=list(self.arms),
            seeds=[self.seed], endpoint_step=1004, post500_lr_multiplier=0.2,
            schedule_horizon=5000, maximum_lr=1e-4, warmup_steps=500,
            weight_decay=0.005, optimizer="AdamW", grad_clip=1.0,
            validation_steps=[0, 500, 1000, 1002, 1004], fixed_train_count=2,
            addon_release_sha256="synthetic-addon",
            source=dict(config=self.config, package=str(self.base / "source_package"),
                        source_run=str(self.base / "source_run")),
        )
        self.manifest = dict(
            format=self.plan["format"], plan=self.plan, seed=self.seed,
            train_names=self.names, val_names=self.val_names,
            fixed_train_names=self.fixed_names, runtime_config=self.config,
            protocol_info={"execution_settings": {"cuda_launch_blocking": None},
                           "fixture": "synthetic CPU"},
        )
        self.original_train = SimpleNamespace(
            JointRegionMemoryClassifier=TinyHead,
            _formal_runtime=lambda *args: None,
            _names=lambda runtime, smoke: (self.names, self.val_names),
            _checked_verify=lambda runtime: {"synthetic_integrity": True},
            _assert_example=lambda example: None,
        )
        torch.manual_seed(93)
        random.seed(93)
        np.random.seed(93)
        heads, opts, _ = ablation_runner.model_copies(
            torch, self.original_train, self.state, self.arms, self.config, torch.device("cpu"))
        runtime = TinyRuntime(self, SimpleNamespace(**self.config), self.train)
        factory = TinyFactory(runtime)
        for step in range(1, 4):
            warm_rows = ablation_runner.update_arms(
                heads, opts, runtime.prepare([self.order[step - 1]]),
                runtime, factory, self.state, step=step, multiplier=0.2)
        # An artificial counter avoids pretending to run1000 real training
        # steps in this CPU wiring fixture; AdamW moments remain nonzero.
        for opt in opts.values():
            for values in opt.state.values():
                values["step"].fill_(1000)
        self.saved = copy.deepcopy(dict(
            format=self.plan["format"], step=1000,
            heads={a: h.state_dict() for a, h in heads.items()},
            optimizers={a: o.state_dict() for a, o in opts.items()},
            rng=self.state.capture_rng(), study_plan=self.plan,
            runtime_config=self.config, protocol_info=self.manifest["protocol_info"],
            input_file_sha256={str(self.input_file): sha256(self.input_file)},
        ))
        self.expected, self.expected_rows = {}, {}
        for step in range(1001, 1005):
            rows = ablation_runner.update_arms(
                heads, opts, runtime.prepare([self.order[step - 1]]),
                runtime, factory, self.state, step=step, multiplier=0.2)
            self.expected_rows[step] = rows
            self.expected[step] = copy.deepcopy(dict(
                heads={a: h.state_dict() for a, h in heads.items()},
                optimizers={a: o.state_dict() for a, o in opts.items()},
                rng=self.state.capture_rng(),
            ))
        self.write(self.root / "plan.json", self.plan)
        self.write(self.train / "manifest.json", self.manifest)
        self.write(self.train / "summary.json", dict(
            phase="failed", complete=False,
            updates_per_arm={a: 1003 for a in self.arms}))
        # Historical scalar rows bind sampling identities and counters only;
        # they are explicitly synthetic and are not quality/trajectory evidence.
        history = []
        for step in range(1, 1004):
            arms = copy.deepcopy(self.expected_rows.get(step, warm_rows))
            for row in arms.values():
                row["optimizer_step"] = step
                row["lr"] = self.state.lr_at_step(
                    step, maximum=1e-4, horizon=5000, multiplier=0.2)
            history.append(dict(
                step=step, epoch=0, epoch_seed=self.seed, image=self.order[step - 1],
                event="synthetic", arms=arms))
        self.journal = self.train / "train.jsonl"
        self.journal.write_text("".join(json.dumps(row) + "\n" for row in history))
        self.checkpoint = self.train / "checkpoints/checkpoint_001000.pt"
        self.checkpoint.parent.mkdir()
        torch.save(self.saved, self.checkpoint)
        self.write(self.checkpoint.with_suffix(".pt.json"), dict(
            step=1000, sha256=sha256(self.checkpoint), roundtrip_verified=True,
            head_hashes={a: self.state.state_fingerprint(self.saved["heads"][a]) for a in self.arms},
        ))
        self.lineage = dict(
            run_root=str(self.root), seed=self.seed,
            original_journal_sha256=sha256(self.journal),
            original_manifest_sha256=sha256(self.train / "manifest.json"),
            last_original_complete_step=1003,
            initial_parent_step=1000,
            initial_parent_checkpoint_sha256=sha256(self.checkpoint),
        )
        self.before = self.snapshot(self.root)
        self.runtime_instances, self.model_instances, self.evaluations = [], [], []

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value) + "\n")

    @staticmethod
    def snapshot(folder):
        return {str(p.relative_to(folder)): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in folder.rglob("*") if p.is_file()}

    def context(self, *, fail=False, parent=None, parent_journal=None):
        def runtime_constructor(args, output):
            runtime = TinyRuntime(self, args, output, fail=fail)
            self.runtime_instances.append(runtime)
            return runtime

        def model_copies(*args, **kwargs):
            result = ablation_runner.model_copies(*args, **kwargs)
            self.model_instances.append(result[:2])
            return result

        runner = SimpleNamespace(
            update_arms=ablation_runner.update_arms, model_copies=model_copies,
            schedule_steps=ablation_runner.schedule_steps)
        bindings = SimpleNamespace(
            torch=torch, Runtime=runtime_constructor, original_train=self.original_train,
            legacy=SimpleNamespace(state=self.state),
            legacy_bridge=SimpleNamespace(verify_dependencies=lambda legacy: None),
        )
        evaluator = SimpleNamespace(evaluate=self.evaluate, print_table=lambda report: None)
        support_module = SimpleNamespace(
            SupportFactory=TinyFactory, ARM_SPECS={a: {"fixture": True} for a in self.arms})

        def named(path, label):
            name = Path(path).name
            if name == "ablation_runner.py":
                return runner
            if name == "evaluation.py":
                return evaluator
            if name == "supports.py":
                return support_module
            raise AssertionError("Unexpected named module: " + name)

        bridge = SimpleNamespace(
            check_package=lambda package: "synthetic-addon",
            verify_ledgers=lambda source: None, load_runtime=lambda source: bindings,
            named=named,
            runtime_arguments=lambda source, *, seed, output: SimpleNamespace(
                **dict(self.config, output=str(output), seed=seed)),
        )
        return dict(
            root=self.root, train=self.train, plan=self.plan, manifest=self.manifest,
            addon=ADDON, bridge=bridge, seed=self.seed, lineage=self.lineage,
            parent=parent or dict(kind="original", path=self.checkpoint,
                                 step=1000, sha256=sha256(self.checkpoint)),
            parent_journal=parent_journal or self.journal,
            evidence_files={
                str(path): sha256(path)
                for path in self.root.rglob("*") if path.is_file()
            },
            checkpoint_every=2,
        )

    def run_branch(self, name, *, fail=False, parent=None, parent_journal=None):
        output = self.base / name
        output.mkdir()
        context = self.context(fail=fail, parent=parent, parent_journal=parent_journal)
        original_write = resume_train.write_json

        def checked_write(path, value):
            if Path(path).name == "result.json" and value.get("complete") is True:
                self.assertTrue(self.runtime_instances[-1].closed,
                                "Evaluation was promoted before runtime closure")
            return original_write(path, value)

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, {}, clear=False))
            stack.enter_context(mock.patch.dict(sys.modules, {}, clear=False))
            stack.enter_context(mock.patch.object(torch.cuda, "is_available", return_value=True))
            stack.enter_context(mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False))
            stack.enter_context(mock.patch.object(resume_train, "write_json", side_effect=checked_write))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            code = resume_train.run_training(context, output)
        self.assertEqual(self.snapshot(self.root), self.before)
        summary = json.loads((output / "summary.json").read_text())
        runtime = self.runtime_instances[-1]
        self.assertTrue(runtime.closed)
        self.assertNotIn("forward", runtime.factory.__dict__)
        self.assertNotIn("loss", runtime.__dict__)
        self.assertIs(runtime.deps.region_nll, pixel_nll)
        for optimizer in self.model_instances[-1][1].values():
            self.assertFalse(optimizer._optimizer_step_pre_hooks)
            self.assertFalse(optimizer._optimizer_step_post_hooks)
        return code, output, summary, runtime

    def evaluate(self, runtime, heads, names, folder, *, step, split, seed,
                 original_train, legacy, factory, primary_endpoint=False, smoke=False):
        from test_evidence import make_report, write_report
        self.assertFalse(smoke)
        self.assertFalse(runtime.closed)
        names = list(names)
        rng = self.state.capture_rng()
        rng_before = self.state.state_fingerprint(rng)
        head_before = self.state.state_fingerprint(
            {a: h.state_dict() for a, h in heads.items()})
        modes = [(module, module.training) for head in heads.values() for module in head.modules()]
        try:
            for module, _ in modes:
                module.training = False
            # Evaluation really perturbs each tested RNG before restoring it.
            torch.rand(11)
            random.random()
            np.random.random()
            result = make_report(step, split, names, self.arms, primary=primary_endpoint)
            write_report(Path(folder), result, names)
        finally:
            for module, mode in modes:
                module.training = mode
            self.state.restore_rng(rng)
        rng_after = self.state.state_fingerprint(self.state.capture_rng())
        self.assertEqual(rng_after, rng_before)
        self.assertEqual(
            self.state.state_fingerprint({a: h.state_dict() for a, h in heads.items()}),
            head_before,
        )
        self.assertFalse(result["complete"])
        self.assertTrue(result["final_integrity_pending"])
        self.evaluations.append(dict(
            folder=str(folder), step=step, split=split, names=names,
            primary=primary_endpoint, rng_before=rng_before, rng_after=rng_after,
        ))
        return result

    def branch_evaluations(self, output):
        return [call for call in self.evaluations
                if Path(call["folder"]).is_relative_to(output)]

    def assert_checkpoint(self, path, step):
        import checkpoint
        saved, record = checkpoint.load(
            torch, path, self.state, plan=self.plan, original_manifest=self.manifest,
            lineage=self.lineage, journal_path=path.parent.parent / "train.jsonl")
        self.assertEqual(saved["step"], step)
        self.assertEqual(record["last_data_position"]["step"], step)
        self.assertEqual(record["last_data_position"]["image"], self.order[step - 1])
        self.assertEqual(record["updates_per_arm"], {a: step for a in self.arms})
        for key in ("heads", "optimizers", "rng"):
            self.assertEqual(
                self.state.state_fingerprint(saved[key]),
                self.state.state_fingerprint(self.expected[step][key]),
                key,
            )
        return saved, record

    def assert_journal(self, output, last):
        rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
        self.assertEqual([row["step"] for row in rows], list(range(1, last + 1)))
        self.assertEqual([row["image"] for row in rows], self.order[:last])
        for row in rows[1000:]:
            self.assertEqual(row["arms"], self.expected_rows[row["step"]])
        return rows

    def test_success_preserves_schedule_rng_and_promotes_after_close(self):
        code, output, summary, runtime = self.run_branch("success")
        self.assertEqual(code, 0)
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["training_complete"])
        self.assertTrue(summary["full_validation_evidence"])
        self.assertFalse(summary["final_integrity_pending"])
        self.assertEqual(summary["updates_per_arm"], {a: 1004 for a in self.arms})
        self.assertEqual(summary["new_updates_per_arm"], {a: 4 for a in self.arms})
        self.assertEqual(summary["restored_step"], 1000)
        self.assertEqual(runtime.prepared, self.order[1000:1004])
        self.assert_journal(output, 1004)
        self.assertEqual(
            [(call["step"], call["split"]) for call in self.branch_evaluations(output)],
            [(1000, "val"), (1000, "fixed_train"), (1002, "val"),
             (1004, "val"), (1004, "fixed_train")],
        )
        checkpoints = sorted((output / "checkpoints").glob("checkpoint_*.pt"))
        self.assertEqual([p.name for p in checkpoints], ["checkpoint_001002.pt", "checkpoint_001004.pt"])
        self.assert_checkpoint(checkpoints[0], 1002)
        self.assert_checkpoint(checkpoints[1], 1004)
        for call in self.branch_evaluations(output):
            folder = Path(call["folder"])
            result = json.loads((folder / "result.json").read_text())
            self.assertTrue(result["complete"])
            self.assertFalse(result["final_integrity_pending"])
            self.assertEqual(result["primary_endpoint"], call["primary"])
            self.assertEqual(
                (folder / "result.json.sha256").read_text().split()[0],
                sha256(folder / "result.json"),
            )

    def test_partial_b_failure_keeps_pending_reports_and_restarts_complete_checkpoint(self):
        code, output, summary, runtime = self.run_branch("failed", fail=True)
        self.assertEqual(code, 1)
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["full_validation_evidence"])
        self.assertEqual(summary["updates_per_arm"], {a: 1002 for a in self.arms})
        self.assertEqual(summary["new_updates_per_arm"], {a: 2 for a in self.arms})
        self.assertEqual(summary["last_complete_step"], 1002)
        self.assertEqual(summary["last_marker"]["arm"], "b")
        self.assertEqual(summary["last_marker"]["phase"], "backward_begin")
        self.assertEqual(summary["optimizer_steps_returned_in_observed_step"], ["e", "f"])
        self.assertIn("synthetic B backward failure", summary["error"])
        self.assertEqual(runtime.prepared, self.order[1000:1003])
        self.assert_journal(output, 1002)
        self.assertEqual(
            [(call["step"], call["split"]) for call in self.branch_evaluations(output)],
            [(1000, "val"), (1000, "fixed_train"), (1002, "val")],
        )
        checkpoints = list((output / "checkpoints").glob("checkpoint_*.pt"))
        self.assertEqual([p.name for p in checkpoints], ["checkpoint_001002.pt"])
        _, record = self.assert_checkpoint(checkpoints[0], 1002)
        for call in self.branch_evaluations(output):
            report = json.loads((Path(call["folder"]) / "result.json").read_text())
            self.assertFalse(report["complete"])
            self.assertTrue(report["final_integrity_pending"])
            self.assertFalse(report["full_validation_evidence"])
            self.assertFalse(report["primary_endpoint"])

        failed_before = self.snapshot(output)
        parent = dict(kind="resume", path=checkpoints[0], step=1002, sha256=record["sha256"])
        code, resumed, resumed_summary, resumed_runtime = self.run_branch(
            "continued", parent=parent, parent_journal=output / "train.jsonl")
        self.assertEqual(code, 0)
        self.assertTrue(resumed_summary["complete"])
        self.assertEqual(resumed_summary["restored_step"], 1002)
        self.assertEqual(resumed_summary["new_updates_per_arm"], {a: 2 for a in self.arms})
        self.assertEqual(self.snapshot(output), failed_before)
        self.assertEqual(resumed_runtime.prepared, self.order[1002:1004])
        self.assert_journal(resumed, 1004)
        self.assertEqual(
            [(call["step"], call["split"]) for call in self.branch_evaluations(resumed)],
            [(1002, "val"), (1002, "fixed_train"), (1004, "val"), (1004, "fixed_train")],
        )
        self.assert_checkpoint(resumed / "checkpoints/checkpoint_001004.pt", 1004)

    def test_endpoint_checkpoint_recovers_validation_without_new_updates(self):
        original_evaluate = self.evaluate

        def fail_endpoint_validation(*args, **kwargs):
            result = original_evaluate(*args, **kwargs)
            if kwargs["step"] == 1004 and kwargs["split"] == "val":
                raise RuntimeError("synthetic endpoint validation failure")
            return result

        with mock.patch.object(self, "evaluate", fail_endpoint_validation):
            code, failed, summary, _ = self.run_branch("endpoint_validation_failed")
        self.assertEqual(code, 1)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["last_complete_step"], 1004)
        self.assertEqual(summary["updates_per_arm"], {a: 1004 for a in self.arms})
        self.assertIn("synthetic endpoint validation failure", summary["error"])
        endpoint = failed / "checkpoints/checkpoint_001004.pt"
        _, record = self.assert_checkpoint(endpoint, 1004)
        for path in failed.glob("*/result.json"):
            report = json.loads(path.read_text())
            self.assertFalse(report["complete"])
            self.assertTrue(report["final_integrity_pending"])
        failed_before = self.snapshot(failed)

        def forbidden_update(*args, **kwargs):
            self.fail("Endpoint validation recovery must not call the updater")

        parent = dict(kind="resume", path=endpoint, step=1004, sha256=record["sha256"])
        with mock.patch.object(ablation_runner, "update_arms", forbidden_update):
            code, recovered, summary, runtime = self.run_branch(
                "endpoint_validation_recovered", parent=parent,
                parent_journal=failed / "train.jsonl")
        self.assertEqual(code, 0)
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["training_complete"])
        self.assertTrue(summary["full_validation_evidence"])
        self.assertEqual(summary["restored_step"], 1004)
        self.assertEqual(summary["last_complete_step"], 1004)
        self.assertEqual(summary["new_updates_per_arm"], {a: 0 for a in self.arms})
        self.assertEqual(runtime.prepared, [])
        self.assertEqual(runtime.loader_requests, [])
        self.assertEqual(self.snapshot(failed), failed_before)
        self.assert_journal(recovered, 1004)
        self.assertEqual(
            [(call["step"], call["split"], call["primary"])
             for call in self.branch_evaluations(recovered)],
            [(1004, "val", True), (1004, "fixed_train", False)],
        )
        self.assertFalse((recovered / "checkpoints").exists())
        heads, optimizers = self.model_instances[-1]
        for key, actual in (
            ("heads", {a: h.state_dict() for a, h in heads.items()}),
            ("optimizers", {a: o.state_dict() for a, o in optimizers.items()}),
            ("rng", self.state.capture_rng()),
        ):
            self.assertEqual(self.state.state_fingerprint(actual),
                             self.state.state_fingerprint(self.expected[1004][key]), key)
        for call in self.branch_evaluations(recovered):
            folder = Path(call["folder"])
            report = json.loads((folder / "result.json").read_text())
            self.assertTrue(report["complete"])
            self.assertFalse(report["final_integrity_pending"])
            self.assertEqual(report["primary_endpoint"], call["primary"])
            self.assertEqual(
                (folder / "result.json.sha256").read_text().split()[0],
                sha256(folder / "result.json"),
            )


if __name__ == "__main__":
    unittest.main()
