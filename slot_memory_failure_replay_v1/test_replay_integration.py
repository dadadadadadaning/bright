"""CPU integration of the replay driver and its isolated worker handoff.

Only the dataset/runtime and process/GPU boundary are faked. Training updates,
AdamW restoration, observer hooks, trajectory comparison and checkpoint
serialization use the shipped implementations. This is not a CUDA test.
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
import replay
from inspect_failure import checkpoint_info, sha256


def reference_state():
    reference = os.environ.get("BRIGHT_ABLATION_REFERENCE")
    if not reference:
        raise RuntimeError("BRIGHT_ABLATION_REFERENCE is required for integration tests")
    path = Path(reference) / "deps/helpers/continuation.py"
    spec = importlib.util.spec_from_file_location("_replay_integration_reference_state", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CPUState:
    def __init__(self, reference):
        self.state_fingerprint = reference.state_fingerprint
        self.lr_at_step = reference.lr_at_step

    def capture_rng(self):
        return dict(
            python=copy.deepcopy(random.getstate()),
            numpy=copy.deepcopy(np.random.get_state()),
            torch_cpu=torch.get_rng_state().clone(),
            torch_cuda=[],
        )

    def restore_rng(self, saved, *, require_cuda=False):
        # The test fakes the CUDA boundary; actual randomness remains CPU.
        if saved["torch_cuda"]:
            raise ValueError("Synthetic integration must not contain CUDA RNG")
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
    raise AssertionError("Use the fixture runtime's ordered evaluation loader")


def fixture_parse(batch, device):
    raise AssertionError("Fixture preparation replaces repository parsing")


class FailBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        raise RuntimeError("synthetic candidate second-arm backward failure")


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

    def prepare(self, example, arms):
        return {arm: None for arm in arms}

    def forward(self, head, example, arm, support):
        self.runtime.active_arm = arm
        features = example.features * (1.05 if arm == "f" else 1.0)
        return {"probabilities": head(features)}

    def provenance(self):
        return {"fixture": "synthetic CPU supports"}


class TinyRuntime:
    def __init__(self, fixture, args, output, fail=False):
        self.fixture, self.args, self.output, self.fail = fixture, args, Path(output), fail
        self.device = torch.device("cpu")
        self.protocol_info = copy.deepcopy(fixture.manifest["protocol_info"])
        self.protocol_info["execution_settings"]["cuda_launch_blocking"] = os.environ.get("CUDA_LAUNCH_BLOCKING")
        self.input_ledger = {fixture.input_file: sha256(fixture.input_file)}
        self.deps = SimpleNamespace(
            region_nll=pixel_nll, lovasz_softmax=auxiliary_loss,
            full_image_loader=fixture_loader, parse_batch=fixture_parse,
        )
        self.guard_report = {"restored": True, "failures": [], "restore_conflicts": []}
        self.image = self.active_arm = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        return False

    def evaluation_batches(self, names, seed):
        # Deliberate setup RNG consumption catches restoration before loader setup.
        torch.rand(7)
        random.random()
        np.random.random()
        return torch.utils.data.DataLoader(
            list(names), batch_size=1, shuffle=False, num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )

    def prepare(self, batch):
        name = batch[0]
        self.image = name
        return SimpleNamespace(
            image=name, event="synthetic",
            features=self.fixture.features[name],
            target=torch.tensor([[[0, 1], [2, 255]]]),
        )

    def loss(self, probability, target):
        loss = self.deps.region_nll(probability, target) + auxiliary_loss(probability, target)
        if self.fail and self.image == self.fixture.order[4] and self.active_arm == "f":
            return FailBackward.apply(loss)
        return loss


class ReplayDriverIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "original"
        self.train = self.root / "seed_3407/train"
        self.train.mkdir(parents=True)
        self.seed, self.arms = 3407, ("e", "f")
        self.state = CPUState(reference_state())
        self.input_file = self.base / "synthetic_input.txt"
        self.input_file.write_text("Synthetic CPU evidence only.\n")
        self.policy_file = self.base / "source_policy.json"
        self.write(self.policy_file, {"numeric_policy": {
            "num_threads": 1, "num_interop_threads": torch.get_num_interop_threads(),
        }})
        self.names = ["synthetic_%02d" % n for n in range(7)]
        self.order = [
            batch[0] for batch in torch.utils.data.DataLoader(
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
            seeds=[self.seed], endpoint_step=5, post500_lr_multiplier=0.2,
            addon_release_sha256="synthetic-addon",
            source=dict(config=self.config, package=str(self.base / "source_package"),
                        source_run=str(self.base / "source_run")),
        )
        self.manifest = dict(
            format=self.plan["format"], plan=self.plan, seed=self.seed,
            train_names=self.names, val_names=["synthetic_val"],
            runtime_config=self.config,
            protocol_info={"execution_settings": {"cuda_launch_blocking": None},
                           "fixture": "synthetic CPU"},
        )
        self.original_train = SimpleNamespace(
            JointRegionMemoryClassifier=TinyHead,
            _formal_runtime=lambda *args: None,
            _names=lambda runtime, smoke: (self.names, ["synthetic_val"]),
            _checked_verify=lambda runtime: {"synthetic_integrity": True},
            _assert_example=lambda example: None,
        )
        torch.manual_seed(93)
        random.seed(93)
        np.random.seed(93)
        heads, optimizers, _ = ablation_runner.model_copies(
            torch, self.original_train, self.state, self.arms, self.config, torch.device("cpu"))
        runtime = TinyRuntime(self, SimpleNamespace(**self.config), self.train)
        factory = TinyFactory(runtime)
        self.rows, self.expected = [], {}
        for step in range(1, 6):
            example = runtime.prepare([self.order[step - 1]])
            rows = ablation_runner.update_arms(
                heads, optimizers, example, runtime, factory, self.state,
                step=step, multiplier=0.2)
            self.rows.append(dict(
                step=step, epoch=0, epoch_seed=self.seed,
                image=example.image, event=example.event, arms=rows))
            payload = copy.deepcopy(dict(
                format=self.plan["format"], step=step,
                heads={a: h.state_dict() for a, h in heads.items()},
                optimizers={a: o.state_dict() for a, o in optimizers.items()},
                rng=self.state.capture_rng(), study_plan=self.plan,
                runtime_config=self.config, protocol_info=self.manifest["protocol_info"],
                input_file_sha256={str(self.input_file): sha256(self.input_file)},
            ))
            if step in (2, 4, 5):
                self.expected[step] = payload
        self.write(self.root / "plan.json", self.plan)
        self.write(self.train / "manifest.json", self.manifest)
        self.write(self.train / "summary.json", dict(
            phase="failed", complete=False, updates_per_arm={a: 4 for a in self.arms}))
        (self.train / "train.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows[:4]))
        checkpoint = self.train / "checkpoints/checkpoint_000002.pt"
        checkpoint.parent.mkdir()
        torch.save(self.expected[2], checkpoint)
        self.write(checkpoint.with_suffix(".pt.json"), dict(
            step=2, sha256=sha256(checkpoint), roundtrip_verified=True,
            head_hashes={a: self.state.state_fingerprint(self.expected[2]["heads"][a]) for a in self.arms},
        ))
        self.chosen = checkpoint_info(checkpoint.parent, self.arms)[0]
        self.lineage = dict(
            run_root=str(self.root), seed=self.seed, parent_step=2,
            parent_checkpoint_sha256=self.chosen["sha256"],
            original_journal_sha256=sha256(self.train / "train.jsonl"),
            original_manifest_sha256=sha256(self.train / "manifest.json"),
            last_original_complete_step=4, candidate_step=5,
        )
        self.before = self.original_snapshot()
        self.runtime_instances = []

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value) + "\n")

    def original_snapshot(self):
        return {str(p.relative_to(self.root)): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in self.root.rglob("*") if p.is_file()}

    def run_supervisor(self, fail):
        def runtime_constructor(args, output):
            instance = TinyRuntime(
                self, args, output,
                fail=fail and Path(output).parent.name == "primary")
            self.runtime_instances.append(instance)
            return instance

        bindings = SimpleNamespace(
            Runtime=runtime_constructor, original_train=self.original_train,
            legacy=SimpleNamespace(state=self.state))

        def runtime_arguments(source, *, seed, output):
            config = dict(self.config, output=str(output), seed=seed)
            return SimpleNamespace(**config)

        bridge = SimpleNamespace(
            check_package=lambda package: "synthetic-addon",
            verify_ledgers=lambda source: None,
            load_runtime=lambda source: bindings,
            named=lambda path, label: (
                ablation_runner if Path(path).name == "ablation_runner.py"
                else SimpleNamespace(SupportFactory=TinyFactory)),
            runtime_arguments=runtime_arguments,
        )
        preflight = (self.root, self.train, self.plan, self.manifest, ADDON,
                     bridge, self.chosen, self.lineage)
        args = argparse.Namespace(
            run_root=str(self.root), seed=self.seed, addon_dir=str(ADDON),
            output_base=str(self.base / "diagnostics"), worker=None, worker_output=None)
        worker_calls = []

        def fake_subprocess(command, check):
            self.assertFalse(check)
            mode = command[command.index("--worker") + 1]
            worker_output = command[command.index("--worker-output") + 1]
            worker_calls.append(mode)
            worker_args = argparse.Namespace(
                run_root=command[command.index("--run-root") + 1],
                seed=int(command[command.index("--seed") + 1]),
                addon_dir=command[command.index("--addon-dir") + 1],
                worker=mode, worker_output=worker_output)
            # Separate fresh worker/runtime objects; CUDA/process boundaries are mocked.
            return SimpleNamespace(returncode=replay.worker(worker_args))

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, {}, clear=False))
            stack.enter_context(mock.patch.dict(sys.modules, {}, clear=False))
            stack.enter_context(mock.patch("replay.preflight", return_value=preflight))
            stack.enter_context(mock.patch("replay.subprocess.run", side_effect=fake_subprocess))
            stack.enter_context(mock.patch.object(torch.cuda, "is_available", return_value=True))
            stack.enter_context(mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=False))
            capture = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            code = replay.supervise(args)
        folders = list((self.base / "diagnostics").glob("replay_*"))
        self.assertEqual(len(folders), 1)
        output = folders[0]
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(self.original_snapshot(), self.before)
        self.assertIn("BEGIN ABLATION REPLAY SUMMARY", capture.getvalue())
        return code, output, summary, worker_calls

    def assert_checkpoint_matches_uninterrupted(self, path, step):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["step"], step)
        for key in ("heads", "optimizers", "rng"):
            self.assertEqual(self.state.state_fingerprint(payload[key]),
                             self.state.state_fingerprint(self.expected[step][key]), key)

    def test_primary_failure_hands_complete_prefix_to_fresh_dense_worker(self):
        code, output, summary, calls = self.run_supervisor(fail=True)
        self.assertEqual(code, 1)  # A diagnostic reproduction is not a formal success.
        self.assertEqual(calls, ["primary", "dense"])
        self.assertEqual(len(self.runtime_instances), 2)
        self.assertIsNot(self.runtime_instances[0], self.runtime_instances[1])
        primary, dense = summary["primary"], summary["dense"]
        self.assertEqual(primary["phase"], "worker_failed")
        self.assertEqual(primary["last_complete_step"], 4)
        self.assertEqual(primary["last_marker"]["arm"], "f")
        self.assertEqual(primary["last_marker"]["phase"], "backward_begin")
        self.assertEqual(primary["optimizer_steps_returned_in_failed_step"], ["e"])
        self.assertEqual(primary["new_updates_per_arm"], {"e": 2, "f": 2})
        committed = [json.loads(line) for line in (output / "primary/replay.jsonl").read_text().splitlines()]
        self.assertEqual([row["step"] for row in committed], [3, 4])
        self.assert_checkpoint_matches_uninterrupted(
            output / "primary/checkpoints/diagnostic_000004.pt", 4)
        self.assertFalse((output / "primary/checkpoints/diagnostic_000005.pt").exists())
        self.assertEqual(dense["phase"], "bounded_replay_passed")
        self.assertEqual(dense["restored_step"], 4)
        self.assertEqual(dense["last_complete_step"], 5)
        self.assertEqual(dense["new_updates_per_arm"], {"e": 1, "f": 1})
        self.assert_checkpoint_matches_uninterrupted(
            output / "dense/checkpoints/diagnostic_000005.pt", 5)
        self.assertTrue(primary["comparison"]["all_exact"])
        dense_markers = [json.loads(line) for line in (output / "dense/markers.jsonl").read_text().splitlines()]
        self.assertTrue(any(row["event"] == "nll_saved_target" for row in dense_markers))

    def test_primary_passes_bounded_endpoint_without_dense_worker(self):
        code, output, summary, calls = self.run_supervisor(fail=False)
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["primary"])
        self.assertIsNone(summary["dense"])
        primary = summary["primary"]
        self.assertEqual(primary["phase"], "bounded_replay_passed")
        self.assertEqual(primary["last_complete_step"], 5)
        self.assertEqual(primary["new_updates_per_arm"], {"e": 3, "f": 3})
        self.assertTrue(primary["comparison"]["all_exact"])
        self.assertEqual(primary["comparison"]["steps_compared"], 2)
        self.assert_checkpoint_matches_uninterrupted(
            output / "primary/checkpoints/diagnostic_000005.pt", 5)


if __name__ == "__main__":
    unittest.main()
