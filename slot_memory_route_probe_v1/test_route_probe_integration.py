"""CPU integration of isolated route diagnosis; this is not a CUDA test.

The runtime/data boundary is synthetic. The original updater, actual route
validator/unpool, AdamW restoration, and diagnostic observers remain active.
"""
from __future__ import annotations

import argparse
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
import route_probe
from inspect_failure import checkpoint_info, sha256

# RouteTrace finds these exact globals through TinyRouteHead.forward.
validate_routes = None
sparse_unpool = None


def reference_module(relative, name):
    root = os.environ.get("BRIGHT_ABLATION_REFERENCE")
    if not root:
        raise RuntimeError("BRIGHT_ABLATION_REFERENCE is required")
    spec = importlib.util.spec_from_file_location(name, Path(root) / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CPUState:
    def __init__(self):
        original = reference_module(
            "deps/helpers/continuation.py", "_route_integration_state")
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
            raise ValueError("CPU fixture must not contain CUDA RNG")
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
    raise AssertionError("Use the fixture runtime loader")


def fixture_parse(*args, **kwargs):
    raise AssertionError("Fixture runtime replaces batch parsing")


class TinyRouteHead(torch.nn.Module):
    def __init__(self, chunk_size=32):
        super().__init__()
        self.query_projection = torch.nn.Linear(3, 4)
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, features, q, ids, **support):
        # This must raise the real implementation's error on mutated shared Q.
        validate_routes(q, ids, 2, 1, 4)
        encoded = self.dropout(self.query_projection(features))[None]
        grid = sparse_unpool(encoded, q, ids)
        return {"probabilities": grid.reshape(1, 4, 2, 2).softmax(1)}


class TinyFactory:
    def __init__(self, runtime, seed=3407):
        self.runtime = runtime
        runtime.factory = self

    def prepare(self, example, arms):
        return {arm: dict(example.supports["b"]) for arm in arms}

    def forward(self, head, example, arm, support):
        self.runtime.active_arm = arm
        common = dict(example.common)
        common["features"] = common["features"] * (1.0 + 0.05 * self.runtime.fixture.arms.index(arm))
        return head(**common, **support)

    def provenance(self):
        return {"fixture": "synthetic CPU route supports"}


class TinyRuntime:
    def __init__(self, fixture, args, output):
        self.fixture, self.args, self.output = fixture, args, Path(output)
        self.device = torch.device("cpu")
        self.protocol_info = copy.deepcopy(fixture.manifest["protocol_info"])
        self.protocol_info["execution_settings"]["cuda_launch_blocking"] = os.environ.get(
            "CUDA_LAUNCH_BLOCKING")
        self.input_ledger = {fixture.input_file: sha256(fixture.input_file)}
        self.deps = SimpleNamespace(
            region_nll=pixel_nll, lovasz_softmax=auxiliary_loss,
            full_image_loader=fixture_loader, parse_batch=fixture_parse,
        )
        self.guard_report = dict(restored=True, failures=[], restore_conflicts=[])
        self.closed = False
        self.active_arm = None
        self.example = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.closed = True
        return False

    def evaluation_batches(self, names, seed):
        # Verify RNG restoration occurs after loader setup.
        torch.rand(7)
        random.random()
        np.random.random()
        return [[name] for name in names]

    def prepare(self, batch):
        name = str(batch[0])
        q = torch.full((1, 9, 4), 1.0 / 9.0)
        ids = (torch.arange(9) % 2)[:, None].expand(9, 4).clone()
        route_q = torch.cat((q * 0.6, q * 0.4), 1)
        route_ids = torch.cat((2 * ids, 2 * ids + 1), 0)
        slot = dict(route_q=route_q, route_ids=route_ids)
        self.last_extract = dict(
            result=dict(q=q, ids=ids, count=2, shape=(2, 2)),
            building=torch.full((1, 1, 2, 2), 0.4),
            slots={"soft": slot, "capacity_control": slot},
        )
        self.example = SimpleNamespace(
            image=name, event="synthetic",
            common=dict(features=self.fixture.features, q=q.detach(), ids=ids),
            supports={"b": slot, "d": slot},
            target=torch.tensor([[[0, 1], [2, 255]]]),
        )
        return self.example

    def loss(self, probability, target):
        return self.deps.region_nll(probability, target) + auxiliary_loss(probability, target)


class RouteProbeIntegrationTests(unittest.TestCase):
    def setUp(self):
        global validate_routes, sparse_unpool
        self.slots = reference_module(
            "deps/original/slots.py", "_route_integration_slots")
        validate_routes, sparse_unpool = self.slots.validate_routes, self.slots.sparse_unpool
        self.original_validator = validate_routes
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
        self.image = "la_palma-volcano_00000167"
        self.names, self.val_names = [self.image], ["synthetic_val"]
        self.features = torch.randn(2, 3, generator=torch.Generator().manual_seed(81))
        self.config = dict(
            repo=str(self.base / "repo"), protocol=str(self.policy_file),
            bank_root=str(self.base / "banks"), values=str(self.base / "values"),
            device="cpu", chunk_size=32, lr=1e-4, weight_decay=0.005,
            output=str(self.train),
        )
        self.plan = dict(
            format="BRIGHT_SLOT_MEMORY_ABLATION_V1", arms=list(self.arms),
            seeds=[self.seed], endpoint_step=2500, post500_lr_multiplier=0.2,
            addon_release_sha256="synthetic-addon",
            source=dict(config=self.config, package=str(self.base / "source_package"),
                        source_run=str(self.base / "source_run")),
        )
        self.manifest = dict(
            format=self.plan["format"], plan=self.plan, seed=self.seed,
            train_names=self.names, val_names=self.val_names, runtime_config=self.config,
            protocol_info={"execution_settings": {"cuda_launch_blocking": None},
                           "fixture": "synthetic CPU"},
        )
        self.original_train = SimpleNamespace(
            JointRegionMemoryClassifier=TinyRouteHead,
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
            ablation_runner.update_arms(
                heads, opts, runtime.prepare([self.image]), runtime, factory, self.state,
                step=step, multiplier=0.2)
        # Artificial checkpoint counter, explicit to keep this CPU fixture small.
        # Nonzero moments and distinct arm states exercise actual restore.
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
        # Independent baseline uses the original full updater on the selected
        # image with the actual next optimizer step, not historical index1303.
        self.state.restore_rng(self.saved["rng"])
        self.expected_rows = ablation_runner.update_arms(
            heads, opts, runtime.prepare([self.image]), runtime, factory, self.state,
            step=1001, multiplier=0.2)
        self.expected = copy.deepcopy(dict(
            heads={a: h.state_dict() for a, h in heads.items()},
            optimizers={a: o.state_dict() for a, o in opts.items()},
            rng=self.state.capture_rng(),
        ))
        self.write(self.root / "plan.json", self.plan)
        self.write(self.train / "manifest.json", self.manifest)
        self.write(self.train / "summary.json", dict(phase="failed", complete=False))
        (self.train / "train.jsonl").write_text("Original history bound by mocked preflight.\n")
        self.checkpoint = self.train / "checkpoints/checkpoint_001000.pt"
        self.checkpoint.parent.mkdir()
        torch.save(self.saved, self.checkpoint)
        self.write(self.checkpoint.with_suffix(".pt.json"), dict(
            step=1000, sha256=sha256(self.checkpoint), roundtrip_verified=True,
            head_hashes={a: self.state.state_fingerprint(self.saved["heads"][a]) for a in self.arms},
        ))
        self.chosen = checkpoint_info(self.checkpoint.parent, self.arms)[0]
        self.before = self.original_snapshot()
        self.runtime_instances, self.model_instances = [], []

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value) + "\n")

    def original_snapshot(self):
        return {str(p.relative_to(self.root)): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in self.root.rglob("*") if p.is_file()}

    def run_probe(self, fail):
        def runtime_constructor(args, output):
            runtime = TinyRuntime(self, args, output)
            self.runtime_instances.append(runtime)
            return runtime

        original_copies = ablation_runner.model_copies

        def model_copies(*args, **kwargs):
            heads, optimizers, initial = original_copies(*args, **kwargs)
            if fail:
                def corrupt_shared_q_after_f(optimizer, args, kwargs):
                    runtime = self.runtime_instances[-1]
                    # Synthetic intervention only; the original B validator
                    # supplies the error and the driver must preserve it.
                    runtime.example.common["q"][0, 0, 0] = -0.001
                handle = optimizers["f"].register_step_post_hook(corrupt_shared_q_after_f)
                self.addCleanup(handle.remove)
            hooks = {
                a: (set(o._optimizer_step_pre_hooks), set(o._optimizer_step_post_hooks))
                for a, o in optimizers.items()
            }
            self.model_instances.append((heads, optimizers, hooks))
            return heads, optimizers, initial

        runner = SimpleNamespace(
            update_arms=ablation_runner.update_arms, model_copies=model_copies)
        bindings = SimpleNamespace(
            Runtime=runtime_constructor, original_train=self.original_train,
            legacy=SimpleNamespace(state=self.state),
        )
        bridge = SimpleNamespace(
            check_package=lambda package: "synthetic-addon",
            verify_ledgers=lambda source: None,
            load_runtime=lambda source: bindings,
            named=lambda path, label: (
                runner if Path(path).name == "ablation_runner.py"
                else SimpleNamespace(SupportFactory=TinyFactory)),
            runtime_arguments=lambda source, *, seed, output: SimpleNamespace(
                **dict(self.config, output=str(output), seed=seed)),
        )
        setup = dict(
            replay=self.base / "original_replay", root=self.root, train=self.train,
            seed=self.seed, plan=self.plan, manifest=self.manifest,
            addon=ADDON, bridge=bridge, chosen=self.chosen,
            lineage={"parent_step": 1000, "fixture": "synthetic CPU"},
            historical_step=1303, image=self.image,
            evidence_files={
                str(path): sha256(path)
                for path in self.root.rglob("*") if path.is_file()
            },
        )
        args = argparse.Namespace(
            replay_output=str(setup["replay"]), addon_dir=str(ADDON),
            output_base=str(self.base / "diagnostics"))
        historical_order = [self.image] * 1303
        reconstructed = (None, historical_order, {1303: {"image": self.image}})
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, {}, clear=False))
            stack.enter_context(mock.patch.dict(sys.modules, {}, clear=False))
            stack.enter_context(mock.patch.object(route_probe, "preflight", return_value=setup))
            stack.enter_context(mock.patch(
                "trajectory.reconstruct_order", return_value=reconstructed))
            stack.enter_context(mock.patch.object(torch.cuda, "is_available", return_value=True))
            stack.enter_context(mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=False))
            capture = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            code = route_probe.run_probe(args)
        outputs = list((self.base / "diagnostics").glob("route_probe_*"))
        self.assertEqual(len(outputs), 1)
        output = outputs[0]
        summary = json.loads((output / "summary.json").read_text())
        self.assertIn("BEGIN ABLATION ROUTE PROBE SUMMARY", capture.getvalue())
        self.assertEqual(self.original_snapshot(), self.before)
        self.assertFalse(any("checkpoint" in p.name for p in output.rglob("*")))
        self.assertFalse(summary["trained_checkpoint_saved"])
        self.assertFalse(summary["formal_experiment_complete"])
        self.assertFalse(summary["exact_historical_state_claim"])
        self.assertEqual(summary["historical_image_step"], 1303)
        self.assertEqual(summary["restored_checkpoint_step"], 1000)
        self.assertEqual(summary["diagnostic_optimizer_step"], 1001)
        self.assertEqual(len(self.runtime_instances), 1)
        self.assertEqual(len(self.model_instances), 1)
        runtime = self.runtime_instances[0]
        self.assertTrue(runtime.closed)
        self.assertNotIn("loss", runtime.__dict__)
        self.assertNotIn("forward", runtime.factory.__dict__)
        self.assertNotIn("prepare", runtime.factory.__dict__)
        self.assertIs(runtime.deps.region_nll, pixel_nll)
        self.assertIs(validate_routes, self.original_validator)
        self.assertIs(self.slots.validate_routes, self.original_validator)
        self.assertIs(sparse_unpool.__globals__["validate_routes"], self.original_validator)
        heads, optimizers, prior_hooks = self.model_instances[0]
        for arm, optimizer in optimizers.items():
            self.assertEqual(
                (set(optimizer._optimizer_step_pre_hooks),
                 set(optimizer._optimizer_step_post_hooks)),
                prior_hooks[arm],
            )
        self.assertEqual(summary["route_trace"]["diagnostic_error_count"], 0)
        self.assertEqual(summary["storage_audit"]["errors"], 0)
        self.assertEqual(summary["storage_audit"]["unsafe_alias_count"], 0)
        return code, output, summary, heads, optimizers

    def assert_state_matches(self, heads, optimizers, updated_arms, started_arms=()):
        for arm in self.arms:
            expected = self.expected if arm in updated_arms else self.saved
            self.assertEqual(
                self.state.state_fingerprint(heads[arm].state_dict()),
                self.state.state_fingerprint(expected["heads"][arm]),
                "head:" + arm,
            )
            expected_optimizer = copy.deepcopy(expected["optimizers"][arm])
            if arm in started_arms and arm not in updated_arms:
                # The original updater sets LR before entering this arm's forward.
                for group in expected_optimizer["param_groups"]:
                    group["lr"] = self.expected_rows[arm]["lr"]
            self.assertEqual(
                self.state.state_fingerprint(optimizers[arm].state_dict()),
                self.state.state_fingerprint(expected_optimizer),
                "optimizer:" + arm,
            )

    def test_success_uses_next_optimizer_step_and_matches_original_updater(self):
        code, output, summary, heads, optimizers = self.run_probe(fail=False)
        self.assertEqual(code, 0)
        self.assertEqual(summary["phase"], "isolated_update_completed")
        self.assertEqual(summary["optimizer_steps_returned"], list(self.arms))
        self.assertEqual(summary["new_complete_updates_per_arm"], {a: 1 for a in self.arms})
        self.assertTrue(summary["final_input_integrity_verified"])
        self.assertEqual(summary["route_trace"]["validator_failure_count"], 0)
        self.assertEqual(summary["route_trace"]["watched_change_count"], 0)
        self.assertGreaterEqual(summary["route_trace"]["validation_call_count"], 12)
        self.assert_state_matches(heads, optimizers, self.arms)
        self.assertEqual(
            self.state.state_fingerprint(self.state.capture_rng()),
            self.state.state_fingerprint(self.expected["rng"]),
        )
        result = json.loads((output / "isolated_update.json").read_text())
        self.assertEqual(result["arms"], self.expected_rows)
        self.assertFalse((output / "route_cpu_case.pt").exists())

    def test_b_original_validator_failure_preserves_partial_update_evidence(self):
        code, output, summary, heads, optimizers = self.run_probe(fail=True)
        self.assertEqual(code, 1)
        self.assertEqual(summary["phase"], "probe_failed")
        self.assertEqual(
            summary["error"], "ValueError: route weights must be nonnegative and sum to one")
        self.assertEqual(summary["failure_phase"], "isolated_original_update")
        self.assertEqual(summary["last_marker"]["arm"], "b")
        self.assertEqual(summary["last_marker"]["phase"], "forward_begin")
        self.assertEqual(summary["optimizer_steps_returned"], ["e", "f"])
        self.assertEqual(summary["new_complete_updates_per_arm"], {a: 0 for a in self.arms})
        self.assertEqual(summary["route_trace"]["validator_failure_count"], 1)
        self.assertGreater(summary["route_trace"]["watched_change_count"], 0)
        self.assert_state_matches(heads, optimizers, ("e", "f"), started_arms=("b",))
        self.assertFalse((output / "isolated_update.json").exists())
        self.assertTrue((output / "route_cpu_case.pt").is_file())
        failure = summary["route_trace"]["validators"]["classifier"]["first_failure"]
        self.assertEqual(failure["arm"], "b")
        self.assertEqual(failure["outcome"]["status"], "raised")
        self.assertEqual(failure["weights"]["negative_count"], 1)
        self.assertEqual(failure["callsite"]["function"], "forward")
        case = torch.load(output / "route_cpu_case.pt", map_location="cpu", weights_only=False)
        q = case["validators"]["classifier"]["first_failure_case"]["q"]
        self.assertEqual(int((q < 0).sum()), 1)
        self.assertIn("validate_routes", (output / "traceback.txt").read_text())


if __name__ == "__main__":
    unittest.main()
