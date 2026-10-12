"""Source/protocol binding tests with synthetic evidence, no real weights."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from . import io_utils


class EvidenceBindingTests(unittest.TestCase):
    def setUp(self):
        self.settings = dict(deterministic_algorithms=False, cuda_matmul_allow_tf32=False,
                             device=None, torch_version="synthetic")
        self.bundle = dict(hashes={"front_sha256": "front", "a1_sha256": "a1"},
                           head_hashes={"j1": "one", "j2": "two", "j3": "three"},
                           reference_hashes={"validation.jsonl": "validation"},
                           inputs_before={"train_list": "train", "val_list": "val"})
        self.args = SimpleNamespace(smoke=False, profile="legacy", pool_mode="native", device="cpu",
                                    dataset=str(Path.cwd() / "synthetic_dataset"))
        self.record = dict(format="SRAC_HARDQUERY_PROTOCOL_V1", scope="all", complete=True,
                           smoke=False, full_validation_evidence=True, image_count=450,
                           reproduction=dict(passed=True, tolerance=1e-5),
                           source_hashes=deepcopy(self.bundle["hashes"]),
                           source_head_hashes=deepcopy(self.bundle["head_hashes"]),
                           reference_hashes=deepcopy(self.bundle["reference_hashes"]),
                           input_fingerprints=deepcopy(self.bundle["inputs_before"]),
                           source_sha256="code", runtime_settings=deepcopy(self.settings),
                           profile="legacy", pool_mode="native",
                           arguments=dict(device=self.args.device, dataset=self.args.dataset))
        self.code_patch = patch.object(io_utils, "code_hash", return_value="code")
        self.code_patch.start()
        self.addCleanup(self.code_patch.stop)

    def validate(self, record=None):
        io_utils.validate_protocol(self.record if record is None else record,
                                   self.bundle, self.args, self.settings)

    def test_formal_e0_record_matches_runtime_identity(self):
        self.validate()
        # The artifact goes through JSON before a subsequent process loads it.
        self.validate(json.loads(json.dumps(self.record)))

    def test_smoke_protocol_cannot_enter_formal_training(self):
        self.record.update(smoke=True, image_count=2, full_validation_evidence=False,
                           reproduction=dict(passed=None, tolerance=1e-5))
        with self.assertRaisesRegex(ValueError, "Formal work requires"):
            self.validate()
        self.args.smoke = True
        self.validate()

    def test_failed_protocol_cannot_be_relabelled_as_smoke(self):
        self.args.smoke = True
        self.record.update(complete=False, reproduction=dict(passed=False))
        with self.assertRaisesRegex(ValueError, "did not complete"):
            self.validate()
        # Even an inconsistent manually relabelled complete flag is rejected.
        self.record["complete"] = True
        with self.assertRaisesRegex(ValueError, "laundered"):
            self.validate()

    def test_a1_scope_and_wrong_format_cannot_gate_joint_training(self):
        for field, value in (("scope", "a1"), ("format", "OLD_AUDIT")):
            record = deepcopy(self.record)
            record[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "all-path"):
                self.validate(record)

    def test_formal_requires_450_and_successful_reproduction(self):
        for field, value in (("image_count", 449), ("full_validation_evidence", False),
                             ("reproduction", {"passed": None}), ("smoke", True)):
            record = deepcopy(self.record)
            record[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Formal work requires"):
                self.validate(record)

    def test_source_profile_pool_and_ledger_mismatch_fail(self):
        for field, value in (("source_hashes", {"front_sha256": "other"}),
                             ("source_head_hashes", {"j1": "other"}),
                             ("reference_hashes", {"validation.jsonl": "other"}),
                             ("source_sha256", "drifted-code"), ("profile", "strict"),
                             ("pool_mode", "grouped")):
            record = deepcopy(self.record)
            record[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity mismatch"):
                self.validate(record)
        self.record["input_fingerprints"]["val_list"] = "different-list"
        with self.assertRaisesRegex(ValueError, "ledger differs"):
            self.validate()

    def test_numeric_setting_drift_fails(self):
        self.record["runtime_settings"]["cuda_matmul_allow_tf32"] = True
        with self.assertRaisesRegex(ValueError, "Runtime differs"):
            self.validate()

    def test_compute_device_is_bound_and_cuda_alias_is_normalized(self):
        self.record["arguments"]["device"] = "cuda:0"
        self.args.device = "cuda"
        self.validate()
        self.args.device = "cuda:1"
        with self.assertRaises(ValueError):
            self.validate()
        self.args.device = "cpu"
        with self.assertRaises(ValueError):
            self.validate()
        self.record["arguments"]["device"] = "cpu:0"
        self.validate()

    def test_dataset_root_is_resolved_and_cannot_change(self):
        self.record["arguments"]["dataset"] = str(Path.cwd() / "intermediate" / ".." / "synthetic_dataset")
        self.validate()
        self.args.dataset = str(Path.cwd() / "other_synthetic_dataset")
        with self.assertRaises(ValueError):
            self.validate()

    def test_safe_child_rejects_parent_sibling_absolute_and_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "artifacts"
            root.mkdir()
            self.assertEqual(io_utils.safe_child(root, "shards/one.pt"), root / "shards" / "one.pt")
            for value in ("../outside.pt", "../artifacts_other/one.pt", str(Path(temp) / "outside.pt"), "."):
                with self.subTest(path=value), self.assertRaisesRegex(ValueError, "escapes"):
                    io_utils.safe_child(root, value)

    def test_artifacts_bind_protocol_digest_and_reject_smoke(self):
        self.bundle["protocol_sha256"] = "protocol-hash"
        record = dict(io_utils.artifact_identity(self.bundle), complete=True, test_accessed=False, smoke=False)
        io_utils.check_artifact(record, self.bundle)
        record["smoke"] = True
        with self.assertRaisesRegex(ValueError, "Smoke artifacts"):
            io_utils.check_artifact(record, self.bundle)
        io_utils.check_artifact(record, self.bundle, allow_smoke=True)
        record["protocol_sha256"] = "different-protocol"
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            io_utils.check_artifact(record, self.bundle, allow_smoke=True)


class HashDriftTests(unittest.TestCase):
    def test_code_hash_covers_local_runtime_and_inherited_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            package = Path(temp) / "new_package"
            package.mkdir()
            fake_io = package / "io_utils.py"
            fake_io.write_text("source", encoding="utf-8")
            (package / "numeric.py").write_text("runtime", encoding="utf-8")
            (package / "run.sh").write_text("runner", encoding="utf-8")
            dependencies = (
                ("damageformer_srac_evidence_followup_v1", "native_pool.py"),
                ("damageformer_srac_stage_diagnostics_v1", "grouped_pool.py"),
                ("damageformer_srac_stage_diagnostics_v1", "memory_core.py"),
                ("damageformer_srac_stage_diagnostics_v1", "memory_audit.py"),
                ("damageformer_srac_partition_probe_v1", "numeric_runtime.py"),
            )
            for folder, name in dependencies:
                dest = Path(temp) / folder / name
                dest.parent.mkdir(exist_ok=True)
                dest.write_text("dependency", encoding="utf-8")
            with patch.object(io_utils, "__file__", str(fake_io)):
                initial = io_utils.code_hash()
                (package / "test_added.py").write_text("test", encoding="utf-8")
                self.assertEqual(initial, io_utils.code_hash())
                (package / "numeric.py").write_text("changed runtime", encoding="utf-8")
                local_changed = io_utils.code_hash()
                self.assertNotEqual(initial, local_changed)
                inherited = Path(temp) / dependencies[-1][0] / dependencies[-1][1]
                inherited.write_text("changed inherited runtime", encoding="utf-8")
                self.assertNotEqual(local_changed, io_utils.code_hash())

    def test_verify_unchanged_detects_inputs_protocol_code_and_runtime(self):
        fake_name = "bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1.memory_audit"
        fake_audit = ModuleType(fake_name)
        fake_audit.fingerprint = lambda bundle: "frozen"
        fake_audit.input_fingerprints = lambda paths: {key: io_utils.digest(path) for key, path in paths.items()}
        fake_audit.diagnostic_code_fingerprint = lambda source_format: {"inherited": "same"}
        with tempfile.TemporaryDirectory() as temp:
            artifact, protocol = Path(temp) / "source.bin", Path(temp) / "protocol.json"
            artifact.write_bytes(b"source")
            protocol.write_text("{}", encoding="utf-8")
            bundle = dict(input_paths={"source": artifact}, inputs_before={"source": io_utils.digest(artifact)},
                          source_manifest={"format": "synthetic"}, code_before={"inherited": "same"},
                          own_source_sha256="code", frozen_before="frozen", runtime_settings={"tf32": False},
                          protocol_path=protocol, protocol_sha256=io_utils.digest(protocol))
            with patch.dict(sys.modules, {fake_name: fake_audit}), \
                    patch.object(io_utils, "code_hash", return_value="code") as code_mock, \
                    patch("bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1.numeric.runtime_settings",
                          return_value={"tf32": False}) as runtime_mock:
                self.assertTrue(all(io_utils.verify_unchanged(bundle).values()))
                artifact.write_bytes(b"changed source")
                with self.assertRaisesRegex(RuntimeError, "Immutable evidence"):
                    io_utils.verify_unchanged(bundle)
                artifact.write_bytes(b"source")
                protocol.write_text('{"changed": true}', encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "Immutable evidence"):
                    io_utils.verify_unchanged(bundle)
                protocol.write_text("{}", encoding="utf-8")
                code_mock.return_value = "changed code"
                with self.assertRaisesRegex(RuntimeError, "Immutable evidence"):
                    io_utils.verify_unchanged(bundle)
                code_mock.return_value = "code"
                runtime_mock.return_value = {"tf32": True}
                with self.assertRaisesRegex(RuntimeError, "Immutable evidence"):
                    io_utils.verify_unchanged(bundle)


if __name__ == "__main__":
    unittest.main()
