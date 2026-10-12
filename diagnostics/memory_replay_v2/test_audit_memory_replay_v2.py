"""CPU-only regression checks for the standalone replay audit.

Run next to AUDIT_MEMORY_REPLAY_V2.py with Python + PyTorch; no pytest or
BRIGHT import is needed. Archives are real temporary .pt files with SHA256
ledgers, so corruption checks exercise the actual disk-loading boundary.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch


SCRIPT = Path(__file__).with_name("AUDIT_MEMORY_REPLAY_V2.py")
SPEC = importlib.util.spec_from_file_location("audit_memory_replay_v2_under_test", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, allow_nan=False), encoding="utf-8")


class ArchiveFixture:
    """Small member archive; the protocol itself certifies formal Val450."""

    def __init__(self, root):
        self.root = Path(root)
        self.cache = self.root / "cache"
        (self.cache / "shards").mkdir(parents=True)
        self.bank_path = self.cache / "bank.pt"
        self.manifest_path = self.cache / "manifest.json"
        self.protocol_path = self.root / "protocol.json"
        self.evidence_path = self.root / "00001.pt"
        self.train_list_path = self.root / "train.txt"
        self.train_list_path.write_text("event_a_image_0\nevent_a_image_1\n", encoding="utf-8")
        tokens = torch.arange(4 * 128, dtype=torch.float32).reshape(4, 128) / 512
        distribution = torch.eye(4, dtype=torch.float32)
        geometry = torch.arange(4 * 6, dtype=torch.float32).reshape(4, 6).half() / 16
        self.shards = []
        records = []
        for image in range(2):
            shard = dict(tokens=(tokens + image).half(), distribution=distribution.clone(),
                         weights=torch.ones(4), geometry=geometry.clone(), mass=torch.ones(4),
                         event=0, image=image, name=f"event_a_image_{image}",
                         region_ids=torch.arange(4, dtype=torch.long))
            path = self.cache / "shards" / f"{image:05d}.pt"
            torch.save(shard, path)
            self.shards.append(shard)
            records.append(dict(path=path.relative_to(self.cache).as_posix(), sha256=sha256(path)))
        self.shard_path = self.cache / records[1]["path"]
        # Bank order is intentionally different from image order. Both support
        # members for image 1 retain their exact original bank row identities.
        images = torch.tensor([1, 0, 1], dtype=torch.long)
        regions = torch.tensor([1, 0, 3], dtype=torch.long)
        self.bank = dict(
            tokens=torch.stack([self.shards[i]["tokens"][r] for i, r in zip(images, regions)]),
            distribution=torch.stack([self.shards[i]["distribution"][r] for i, r in zip(images, regions)]),
            geometry=torch.stack([self.shards[i]["geometry"][r] for i, r in zip(images, regions)]),
            events=torch.zeros(3, dtype=torch.long), images=images, regions=regions,
            disasters=torch.zeros(3, dtype=torch.long), event_names=["event_a"],
            checkpoint_sha256="f" * 64,
        )
        torch.save(self.bank, self.bank_path)
        self.manifest = dict(format="SRAC_SNAPSHOT_MEMORY_V1", complete=True, split="train",
                             smoke=False, test_accessed=False, image_count=2, shards=records,
                             event_names=["event_a"], bank_sha256=sha256(self.bank_path),
                             checkpoint_sha256=self.bank["checkpoint_sha256"],
                             train_sha256=sha256(self.train_list_path),
                             source_weights_and_buffers_unchanged=True)
        write_json(self.manifest_path, self.manifest)
        self.protocol = dict(
            format="SRAC_HARDQUERY_PROTOCOL_V1", complete=True,
            full_validation_evidence=True, reproduction={"passed": True},
            image_count=450, scope="all", smoke=False, profile="legacy", pool_mode="native",
            test_accessed=False, input_files_unchanged=True, code_unchanged=True,
            frozen_evidence_unchanged=True, numerical_unchanged=True,
            selected_images=[f"val_{i:04d}" for i in range(450)],
            arguments=dict(cache=str(self.cache), train_list=str(self.train_list_path)),
            source_hashes=dict(bank_sha256=sha256(self.bank_path),
                               cache_manifest_sha256=sha256(self.manifest_path)),
            input_fingerprints=dict(bank=sha256(self.bank_path),
                                    cache_manifest=sha256(self.manifest_path),
                                    train_list=sha256(self.train_list_path)),
            input_paths=dict(bank=str(self.bank_path), cache_manifest=str(self.manifest_path),
                             train_list=str(self.train_list_path)),
        )
        write_json(self.protocol_path, self.protocol)
        rows = torch.tensor([0, 2], dtype=torch.long)
        self.evidence = dict(
            format="SRAC_MEMORY_KEY_REPLAY_EVIDENCE_V2", image="event_a_image_1",
            train_image_id=1, original_rows=rows.clone(), region_ids=regions[rows].clone(),
            replayed_fp32=self.bank["tokens"][rows].float() + 2e-6,
            archived_tokens=self.bank["tokens"][rows].clone(), report={"passed": True},
            source_sha256="a" * 64, protocol_sha256=sha256(self.protocol_path),
            original_bank_sha256=sha256(self.bank_path),
        )
        self.save_evidence()

    def save_evidence(self):
        torch.save(self.evidence, self.evidence_path)
        self.evidence_sha256 = sha256(self.evidence_path)

    def save_protocol(self):
        write_json(self.protocol_path, self.protocol)
        self.evidence["protocol_sha256"] = sha256(self.protocol_path)
        self.save_evidence()

    def rebind_shard_hash(self):
        """Keep hashes internally valid to exercise semantic cross-checks."""
        torch.save(self.shards[1], self.shard_path)
        self.manifest["shards"][1]["sha256"] = sha256(self.shard_path)
        write_json(self.manifest_path, self.manifest)
        self.protocol["source_hashes"]["cache_manifest_sha256"] = sha256(self.manifest_path)
        self.protocol["input_fingerprints"]["cache_manifest"] = sha256(self.manifest_path)
        self.save_protocol()

    def validate(self):
        return audit.validate_archive(self.protocol_path, self.evidence_path, self.evidence_sha256)


class ValidateArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fx = ArchiveFixture(self.temp.name)

    def reject(self):
        with self.assertRaises((ValueError, RuntimeError)):
            self.fx.validate()

    def test_valid_archive_preserves_identity_and_never_initializes_cuda(self):
        before = {path: sha256(path) for path in self.fx.root.rglob("*") if path.is_file()}
        with mock.patch.object(torch.cuda, "init", side_effect=AssertionError("CPU gate used CUDA")):
            result = self.fx.validate()
        self.assertEqual(result["image"], "event_a_image_1")
        self.assertEqual(result["train_image_id"], 1)
        self.assertTrue(torch.equal(result["original_rows"], torch.tensor([0, 2])))
        self.assertTrue(torch.equal(result["region_ids"], torch.tensor([1, 3])))
        self.assertEqual(result["bank"]["tokens"].device.type, "cpu")
        self.assertEqual(result["shard"]["tokens"].device.type, "cpu")
        self.assertEqual(before, {path: sha256(path) for path in self.fx.root.rglob("*") if path.is_file()})

    def test_cpu_only_run_is_observational_and_needs_no_bright_import_or_cuda(self):
        output = self.fx.root / "cpu_audit"
        cli = SimpleNamespace(protocol=str(self.fx.protocol_path),
                              evidence=str(self.fx.evidence_path),
                              evidence_sha256=self.fx.evidence_sha256,
                              output=str(output), repo=str(self.fx.root / "missing_repo"),
                              repeats=2, cpu_only=True)
        before = {path: sha256(path) for path in self.fx.root.rglob("*") if path.is_file()}
        with mock.patch.object(torch.cuda, "init", side_effect=AssertionError("CPU audit used CUDA")):
            report = audit.run(cli)
        self.assertIs(report["audit_complete"], True)
        self.assertIs(report["complete"], False)
        self.assertIs(report["full_validation_evidence"], False)
        self.assertIs(report["gpu_ran"], False)
        self.assertIs(report["training"], False)
        self.assertIs(report["test_accessed"], False)
        self.assertEqual(report["phase"], "CPU_only_complete")
        self.assertIs(report["archive_files_unchanged"], True)
        self.assertEqual(json.loads((output / "summary.json").read_text(encoding="utf-8")), report)
        self.assertTrue(all(sha256(path) == digest for path, digest in before.items()))

    def test_cpu_only_rechecks_archive_ledger_after_validation(self):
        output = self.fx.root / "cpu_audit_mutation"
        cli = SimpleNamespace(protocol=str(self.fx.protocol_path),
                              evidence=str(self.fx.evidence_path),
                              evidence_sha256=self.fx.evidence_sha256,
                              output=str(output), repo=str(self.fx.root / "missing_repo"),
                              repeats=2, cpu_only=True)
        validate = audit.validate_archive

        def mutate_after_validation(*args):
            archive = validate(*args)
            with self.fx.bank_path.open("ab") as handle:
                handle.write(b"mutation_after_CPU_validation")
            return archive

        with mock.patch.object(audit, "validate_archive", side_effect=mutate_after_validation):
            with self.assertRaisesRegex(RuntimeError, "changed during the audit"):
                audit.run(cli)
        report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertIs(report["audit_complete"], False)
        self.assertIs(report["complete"], False)
        self.assertIs(report["gpu_ran"], False)
        self.assertTrue((output / "traceback.txt").is_file())

    def test_evidence_byte_drift_rejected_before_payload_load(self):
        with self.fx.evidence_path.open("ab") as handle:
            handle.write(b"drift")
        self.reject()

    def test_bank_byte_drift_rejected(self):
        with self.fx.bank_path.open("ab") as handle:
            handle.write(b"drift")
        self.reject()

    def test_manifest_byte_drift_rejected(self):
        with self.fx.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        self.reject()

    def test_shard_byte_drift_rejected(self):
        with self.fx.shard_path.open("ab") as handle:
            handle.write(b"drift")
        self.reject()

    def test_existing_input_train_list_drift_rejected(self):
        self.fx.train_list_path.write_text("renamed\n", encoding="utf-8")
        self.reject()

    def test_protocol_hash_binding_cannot_be_replaced_with_another_protocol(self):
        self.fx.protocol["selected_images"][0] = "different_val_image"
        write_json(self.fx.protocol_path, self.fx.protocol)
        self.reject()

    def test_failed_or_incomplete_formal_protocol_rejected(self):
        for key, value in (("complete", False), ("smoke", True), ("image_count", 449),
                           ("profile", "modern"), ("pool_mode", "grouped"),
                           ("test_accessed", True), ("full_validation_evidence", False),
                           ("input_files_unchanged", False)):
            with self.subTest(key=key):
                old = self.fx.protocol[key]
                self.fx.protocol[key] = value
                self.fx.save_protocol()
                self.reject()
                self.fx.protocol[key] = old
                self.fx.save_protocol()

    def test_source_hash_and_input_fingerprint_must_both_bind_bank(self):
        for field in ("source_hashes", "input_fingerprints"):
            with self.subTest(field=field):
                key = "bank_sha256" if field == "source_hashes" else "bank"
                old = self.fx.protocol[field][key]
                self.fx.protocol[field][key] = "0" * 64
                self.fx.save_protocol()
                self.reject()
                self.fx.protocol[field][key] = old
                self.fx.save_protocol()

    def test_missing_archive_hash_binding_is_rejected(self):
        for field, key in (("source_hashes", "bank_sha256"),
                           ("source_hashes", "cache_manifest_sha256"),
                           ("input_fingerprints", "bank"),
                           ("input_fingerprints", "cache_manifest")):
            with self.subTest(field=field, key=key):
                old = self.fx.protocol[field].pop(key)
                self.fx.save_protocol()
                self.reject()
                self.fx.protocol[field][key] = old
                self.fx.save_protocol()

    def test_formal_protocol_requires_reproduction_and_unique_val450(self):
        self.fx.protocol["reproduction"]["passed"] = False
        self.fx.save_protocol()
        self.reject()
        self.fx.protocol["reproduction"]["passed"] = True
        self.fx.protocol["selected_images"][1] = self.fx.protocol["selected_images"][0]
        self.fx.save_protocol()
        self.reject()

    def test_evidence_archive_payload_cannot_drift_even_with_updated_evidence_hash(self):
        self.fx.evidence["archived_tokens"][0, 0] += 1
        self.fx.save_evidence()
        self.reject()

    def test_shard_and_bank_labels_must_agree_even_if_file_hashes_are_valid(self):
        self.fx.shards[1]["distribution"][1] = torch.tensor([0., 0., 1., 0.])
        self.fx.rebind_shard_hash()
        self.reject()

    def test_shard_and_bank_geometry_must_agree_even_if_file_hashes_are_valid(self):
        self.fx.shards[1]["geometry"][1, 0] += 1
        self.fx.rebind_shard_hash()
        self.reject()

    def test_evidence_cannot_select_another_image_or_name(self):
        for key, value in (("train_image_id", 0), ("image", "wrong_name")):
            with self.subTest(key=key):
                old = self.fx.evidence[key]
                self.fx.evidence[key] = value
                self.fx.save_evidence()
                self.reject()
                self.fx.evidence[key] = old
                self.fx.save_evidence()

    def test_row_and_region_identity_must_be_valid_unique_long(self):
        cases = (
            ("original_rows", torch.tensor([0, 0])),
            ("original_rows", torch.tensor([0, 3])),
            ("original_rows", torch.tensor([0., 2.])),
            ("region_ids", torch.tensor([1, 1])),
            ("region_ids", torch.tensor([1, 4])),
            ("region_ids", torch.tensor([1., 3.])),
        )
        for key, value in cases:
            with self.subTest(key=key, values=value.tolist()):
                old = self.fx.evidence[key]
                self.fx.evidence[key] = value
                self.fx.save_evidence()
                self.reject()
                self.fx.evidence[key] = old
                self.fx.save_evidence()

    def test_raw_replay_must_retain_fp32_not_upcast_archive(self):
        self.fx.evidence["replayed_fp32"] = self.fx.evidence["replayed_fp32"].half()
        self.fx.save_evidence()
        self.reject()

    def test_nonfinite_raw_replay_rejected(self):
        self.fx.evidence["replayed_fp32"][0, 0] = float("nan")
        self.fx.save_evidence()
        self.reject()


class ControlledOrchestrationTests(unittest.TestCase):
    """Exercise the real six-forward orchestration entirely on CPU.

    Only the repository loading and data dependencies are replaced. The
    actual StageCapture prehooks, replay math, comparisons, disk artifacts,
    input hashing and final immutability decisions execute normally. The
    fake device descriptor opens the CUDA-only orchestration branch while
    every actual tensor and model remains on CPU.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fx = ArchiveFixture(self.temp.name)
        self.fx.evidence["source_sha256"] = "v2-source-code"
        self.fx.save_evidence()
        self.pre = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2) / 12
        self.post = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2) / 24
        self.target = torch.arange(4, dtype=torch.long).reshape(1, 2, 2)
        self.input_paths = {}
        for name in ("pre", "post", "target"):
            path = self.fx.root / (name + ".synthetic_image")
            path.write_bytes((name + " unchanged source image bytes").encode())
            self.input_paths[name] = path

        class CpuPoolGuard:
            active = False
            instances = []

            def __init__(guard_self, *_args, **_kwargs):
                guard_self.image = None
                guard_self.restored = False
                guard_self.failures = []
                CpuPoolGuard.instances.append(guard_self)

            def __enter__(guard_self):
                CpuPoolGuard.active = True
                return guard_self

            def set_image(guard_self, name):
                guard_self.image = name

            def __exit__(guard_self, *_args):
                CpuPoolGuard.active = False
                guard_self.restored = True

            def report(guard_self):
                return dict(mode="native", restored=guard_self.restored,
                            failures=guard_self.failures, image=guard_self.image)

        shard = self.fx.shards[1]

        class TinyFront(torch.nn.Module):
            def __init__(front_self):
                super().__init__()
                front_self.base = torch.nn.Module()
                front_self.base.clf_2 = torch.nn.Identity()
                front_self.embedding = torch.nn.Module()
                front_self.embedding.main = torch.nn.Sequential(torch.nn.Identity())
                front_self.embedding.stats = torch.nn.Sequential(torch.nn.Identity())
                front_self.embedding.geometry = torch.nn.Sequential(torch.nn.Identity())
                front_self.register_buffer("token_values", shard["tokens"].float().unsqueeze(0))
                front_self.register_buffer("geometry_values", shard["geometry"].float().unsqueeze(0))
                front_self.gain = torch.nn.Parameter(torch.tensor(1.))
                front_self.forward_flags = []
                front_self.input_ids = []
                front_self.mutate_input = False

            def forward(front_self, pre, post):
                front_self.forward_flags.append((torch.is_inference_mode_enabled(),
                                                 torch.is_grad_enabled(), CpuPoolGuard.active))
                front_self.input_ids.append((id(pre), id(post)))
                if front_self.mutate_input:
                    pre.add_(1.)
                feature = (pre.sum(1, keepdim=True) + post.sum(1, keepdim=True)).repeat(1, 256, 1, 1)
                front_self.base.clf_2(feature)
                # These are real module invocations, exercising all four
                # prehooks with the same shapes as the front's stage outputs.
                content = front_self.token_values * front_self.gain
                main = front_self.embedding.main(content)
                stats = front_self.embedding.stats(torch.cat((content, content), -1))
                geometry = front_self.embedding.geometry(front_self.geometry_values * front_self.gain)
                tokens = main + 0. * stats[..., :128] + 0. * geometry.sum(-1, keepdim=True)
                return dict(tokens=tokens, q=torch.ones(1, 1, 4), ids=torch.arange(4).reshape(1, 4),
                            adapted_content=feature[:, :128], mass=torch.ones(1, 4))

        self.guard_type = CpuPoolGuard
        self.front = TinyFront().eval().requires_grad_(False)
        args = SimpleNamespace(pool_mode="native", dataset=str(self.fx.root), workers=0, seed=3407)
        self.bundle = dict(args=args, front=self.front,
                           device=SimpleNamespace(type="cuda"),
                           metric=torch.nn.Linear(128, 2).eval().requires_grad_(False),
                           decoders={arm: torch.nn.Linear(128, 4).eval().requires_grad_(False)
                                     for arm in ("j1", "j2", "j3")},
                           train_names=["event_a_image_0", "event_a_image_1"],
                           runtime_settings=dict(profile="legacy", deterministic=True))

        shared = ModuleType("bda_benchmark.experiments.damageformer_srac_memory_key_compare_v2.shared")
        shared.source_hash = mock.Mock(return_value="v2-source-code")
        shared.load_bundle = mock.Mock(return_value=self.bundle)

        def verify_all(_bundle, source, ledger):
            self.assertIs(_bundle, self.bundle)
            self.assertEqual(source, "v2-source-code")
            self.assertTrue(all(sha256(path) == expected for path, expected in ledger.items()))
            return dict(source=True, input_files=True)

        shared.verify_all = mock.Mock(side_effect=verify_all)
        self.shared = shared
        repo = Path(os.environ.get("BRIGHT_REPO", "/workspace/bright"))
        replay_path = repo / "bda_benchmark/experiments/damageformer_srac_memory_key_compare_v2/replay.py"
        replay_spec = importlib.util.spec_from_file_location("synthetic_audit_actual_replay", replay_path)
        replay = importlib.util.module_from_spec(replay_spec)
        replay_spec.loader.exec_module(replay)
        package_name = "bda_benchmark.experiments.damageformer_srac_memory_key_compare_v2"
        package = ModuleType(package_name)
        package.__path__ = [str(replay_path.parent)]
        package.shared, package.replay = shared, replay
        pool = ModuleType("bda_benchmark.experiments.damageformer_srac_evidence_followup_v1.native_pool")
        pool.PoolGuard = CpuPoolGuard
        memory = ModuleType("bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit")
        memory.full_image_loader = mock.Mock(return_value=[(self.pre, self.post, None, self.target,
                                                           ["event_a_image_1"])])
        memory.parse_batch = mock.Mock(side_effect=lambda batch, _device: batch)
        memory.region_targets = mock.Mock(return_value=(shard["distribution"].unsqueeze(0),
                                                       shard["weights"].unsqueeze(0)))
        core = ModuleType("bda_benchmark.experiments.damageformer_srac_region_test_v1.core")
        core.sample_paths = mock.Mock(return_value=self.input_paths)
        self.modules = {package_name: package, shared.__name__: shared,
                        package_name + ".replay": replay, pool.__name__: pool,
                        memory.__name__: memory, core.__name__: core}
        self.cli = SimpleNamespace(protocol=str(self.fx.protocol_path), evidence=str(self.fx.evidence_path),
                                   evidence_sha256=self.fx.evidence_sha256,
                                   output=str(self.fx.root / "controlled_audit"),
                                   repo=str(repo), repeats=2, cpu_only=False)

    def run_synthetic(self):
        with mock.patch.dict(sys.modules, self.modules), mock.patch.object(sys, "path", list(sys.path)):
            with mock.patch.object(torch.cuda, "init", side_effect=AssertionError("Synthetic CPU test initialized CUDA")):
                return audit.run(self.cli)

    def test_three_conditions_two_repeats_use_real_hooks_and_preserve_inputs(self):
        expected_tensor_hashes = dict(pre=audit.tensor_digest(self.pre), post=audit.tensor_digest(self.post))
        input_bytes = {key: tensor.numpy().tobytes() for key, tensor in
                       (("pre", self.pre), ("post", self.post))}
        before = {path: sha256(path) for path in self.fx.root.rglob("*") if path.is_file()}
        report = self.run_synthetic()
        self.assertIs(report["audit_complete"], True)
        self.assertIs(report["complete"], False)
        self.assertIs(report["full_validation_evidence"], False)
        self.assertEqual(self.front.forward_flags,
                         [(True, False, False)] * 2 + [(False, False, False)] * 2
                         + [(False, False, True)] * 2)
        self.assertEqual(self.front.input_ids, [(id(self.pre), id(self.post))] * 6)
        self.assertEqual(report["input_tensor_sha256"], expected_tensor_hashes)
        self.assertEqual(report["input_tensor_sha256_after"], expected_tensor_hashes)
        self.assertIs(report["input_tensor_bytes_unchanged"], True)
        self.assertEqual(self.pre.numpy().tobytes(), input_bytes["pre"])
        self.assertEqual(self.post.numpy().tobytes(), input_bytes["post"])
        self.assertEqual(self.shared.verify_all.call_count, 2)
        self.assertEqual(report["current_image_fingerprints"],
                         {str(path): sha256(path) for path in self.input_paths.values()})
        self.assertTrue(all(sha256(path) == digest for path, digest in before.items()))
        self.assertEqual(len(report["samples"]), 6)
        self.assertEqual(len(report["comparisons"]), 9)
        pairs = {(row["before"], row["after"]) for row in report["comparisons"]}
        self.assertEqual(len(pairs), 9)
        conditions = ("native_inference", "native_no_grad", "guarded_no_grad")
        expected_pairs = {(f"{condition}_1", f"{condition}_2") for condition in conditions}
        expected_pairs |= {(f"{conditions[left]}_{repeat}", f"{conditions[right]}_{repeat}")
                           for left, right in ((0, 1), (0, 2), (1, 2)) for repeat in (1, 2)}
        self.assertEqual(pairs, expected_pairs)
        for row in report["comparisons"]:
            self.assertEqual(set(row["stages"]), set(audit.STAGES))
            self.assertIsNone(row["first_recorded_exact_difference"])
            self.assertTrue(all(stage["exact_equal"] for stage in row["stages"].values()))
        artifacts = list(Path(self.cli.output).glob("*.pt"))
        self.assertEqual(len(artifacts), 6)
        expected = self.fx.evidence["archived_tokens"].float()
        for row in report["samples"]:
            path = Path(self.cli.output) / row["artifact"]
            self.assertEqual(sha256(path), row["artifact_sha256"])
            artifact = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(artifact["tokens"].dtype, torch.float32)
            self.assertTrue(torch.equal(artifact["tokens"], expected))
            self.assertIs(row["selected_storage_report"]["passed"], True)
        self.assertEqual(len(self.guard_type.instances), 2)
        self.assertTrue(all(guard.restored and guard.image == "event_a_image_1"
                            for guard in self.guard_type.instances))
        self.assertTrue(all(not module._forward_pre_hooks for module in self.front.modules()))

    def test_controlled_audit_rejects_input_tensor_mutation(self):
        self.front.mutate_input = True
        with self.assertRaisesRegex(RuntimeError, "input|Input"):
            self.run_synthetic()
        report = json.loads((Path(self.cli.output) / "summary.json").read_text(encoding="utf-8"))
        self.assertIs(report["audit_complete"], False)
        self.assertIs(report["complete"], False)
        self.assertIs(report["full_validation_evidence"], False)
        self.assertIs(report["input_tensor_bytes_unchanged"], False)


class CompareTensorTests(unittest.TestCase):
    def test_exact_fp32_replay(self):
        before = torch.tensor([[0., -1., 2.]])
        report = audit.compare_tensor(before, before.clone())
        for key in ("shape_equal", "dtype_equal", "finite", "exact_equal"):
            self.assertIs(report[key], True)
        self.assertEqual(report["max_abs"], 0.)
        self.assertEqual(report["differing_elements"], 0)

    def test_numeric_drift_is_measured_without_mutating_inputs(self):
        before = torch.tensor([1., 2., 3.])
        after = torch.tensor([1., 2.125, 2.75])
        first, second = before.clone(), after.clone()
        report = audit.compare_tensor(before, after)
        self.assertIs(report["exact_equal"], False)
        self.assertEqual(report["differing_elements"], 2)
        self.assertEqual(report["max_abs"], .25)
        self.assertTrue(torch.equal(before, first))
        self.assertTrue(torch.equal(after, second))

    def test_noncontiguous_tensor_is_compared_by_value(self):
        view = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
        self.assertFalse(view.is_contiguous())
        report = audit.compare_tensor(view, view.contiguous())
        self.assertIs(report["exact_equal"], True)
        self.assertEqual(report["differing_elements"], 0)

    def test_shape_mismatch_reports_failure_without_broadcast(self):
        report = audit.compare_tensor(torch.ones(1, 3), torch.ones(3))
        self.assertIs(report["shape_equal"], False)
        self.assertIs(report["exact_equal"], False)
        self.assertIsNone(report["max_abs"])

    def test_dtype_mismatch_does_not_claim_exact_replay(self):
        report = audit.compare_tensor(torch.ones(3), torch.ones(3, dtype=torch.float16))
        self.assertIs(report["shape_equal"], True)
        self.assertIs(report["dtype_equal"], False)
        self.assertIs(report["exact_equal"], False)

    def test_nonfinite_is_not_exact_replay_evidence(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                tensor = torch.tensor([value])
                report = audit.compare_tensor(tensor, tensor.clone())
                self.assertIs(report["finite"], False)
                self.assertIs(report["exact_equal"], False)
                self.assertIsNone(report["max_abs"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
