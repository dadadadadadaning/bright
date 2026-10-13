"""CPU metadata/binding contracts; the injected audit is not real BRIGHT evidence."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest

import prerequisites as p


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class Fixture:
    def __init__(self, base):
        self.base = Path(base)
        self.root = self.base / "sar_source"
        self.repo = self.base / "repo"
        (self.repo / "bda_benchmark").mkdir(parents=True)
        self.root.mkdir()
        self.output = self.base / "new/certificate.json"
        self.runtime = {"cublas_workspace_config": None, "torch_version": "fixture"}
        inputs, input_hashes = {}, {}
        for name in ("front", "a1", "train_list", "val_list"):
            path = self.base / (name + ".input")
            path.write_bytes(("immutable-" + name).encode())
            inputs[name], input_hashes[name] = str(path), p.digest(path)
        self.protocol = self.base / "e0/protocol.json"
        e0 = dict(format="SRAC_HARDQUERY_PROTOCOL_V1", scope="all", complete=True,
                  full_validation_evidence=True, smoke=False, image_count=450, test_accessed=False,
                  reproduction={"passed": True}, profile="legacy", pool_mode="native",
                  runtime_settings=self.runtime, input_paths=inputs, input_fingerprints=input_hashes)
        write(self.protocol, e0)
        settings = dict(steps=2000, eval_every=2000, batch=2, accum=8, workers=4,
                        lr=0.0001, aux_weight=0.1, cue_gain=0.1, amp="bf16")
        self.plan, self.stages = [], []
        for arm in p.ARMS:
            fit, val = self.root / "formal" / ("train_" + arm), self.root / "formal" / ("val_" + arm)
            fit.mkdir(parents=True)
            val.mkdir(parents=True)
            checkpoint = fit / "last.pth"
            checkpoint.write_bytes(("fixture-not-pickle-" + arm).encode())
            tm = dict(format="SRAC_Q_FACTORIAL_V1", arm=arm, arguments=dict(settings),
                source_hash="q-source", inherited_source_hash="inherited-source", initial_front_step=0,
                initial_front_complete=True, protocol_path=str(self.protocol), protocol_sha256=p.digest(self.protocol),
                front_sha256=input_hashes["front"], a1_sha256=input_hashes["a1"],
                train_sha256=input_hashes["train_list"], val_sha256=input_hashes["val_list"],
                e0_profile="legacy", e0_pool_mode="native", e0_runtime=self.runtime,
                frozen_a1=True, test_accessed=False, smoke=False)
            ts = dict(format="SRAC_Q_FACTORIAL_V1", complete=True, full_validation_evidence=True,
                arm=arm, steps=2000, formal_checkpoint="last.pth", smoke=False, test_accessed=False,
                a1_unchanged=True, downstream_unchanged=True, immutable={"files": True, "frozen_downstream": True})
            write(fit / "manifest.json", tm)
            write(fit / "summary.json", ts)
            report = dict(format="SRAC_Q_FACTORIAL_EVAL_V1", arm=arm, complete=True,
                full_validation_evidence=True, smoke=False, test_accessed=False, image_count=450,
                checkpoint_steps=2000, native_a1_replay_ok=True, reproduction={"passed": True},
                q_a1_reproduction={"passed": True}, protocol_sha256=p.digest(self.protocol),
                protocol_path=str(self.protocol), numerical_profile="legacy", pool_mode="native",
                immutable={"files": True}, training_hyperparameters=settings,
                checkpoint_path=str(checkpoint), checkpoint_sha256=p.digest(checkpoint), source_sha256="eval-source")
            write(val / "result.json", report)
            report.update(result_file="result.json", result_sha256=p.digest(val / "result.json"))
            write(val / "summary.json", report)
            rows = []
            for index in range(450):
                artifact = val / "predictions" / ("%04d.npz" % index)
                artifact.parent.mkdir(exist_ok=True)
                artifact.write_bytes(("fake-array-" + str(index)).encode())
                rows.append(dict(index=index, image="image_%04d" % index,
                                 prediction_file=str(artifact.relative_to(val)), prediction_sha256=p.digest(artifact)))
            (val / "images.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            for kind, folder in (("train", fit), ("val", val)):
                name = "formal_" + kind + "_" + arm
                gate = {"passed": True, "summary_sha256": p.digest(folder / "summary.json")}
                if kind == "train":
                    gate["checkpoint_sha256"] = p.digest(checkpoint)
                self.add_stage(name, gate)
        self.audit = dict(format="BRIGHT_EXTERNAL_SAR_VALIDATION_V1", complete=True, reports_only=False,
            training=False, test_accessed=False, status={key: {"passed": True, "artifact_audit": {
                "passed": True, "complete_audit": True, "image_count": 450}} for key in p.OPTIONS},
            direct_differences_pp={"q_grid_probq": {"sar_minus_baseline": {"foreground_miou_pp": -4.0}}})
        write(self.root / "sar_verification.json", self.audit)
        self.add_stage("verify_four_arm_artifacts", {"passed": True, "report_sha256": p.digest(self.root / "sar_verification.json")})
        self.manifest = dict(format=p.SOURCE_FORMAT, arguments={"repo_root": str(self.repo), "protocol": str(self.protocol)},
            test_accessed=False, unchanged_original_code=True, source_bindings={str(self.protocol): p.digest(self.protocol)},
            protocol_sha256=p.digest(self.protocol), source_runtime_settings=self.runtime,
            runtime_environment={"CUDA_LAUNCH_BLOCKING": "1", "CUBLAS_WORKSPACE_CONFIG": None}, stages=self.plan)
        self.state = dict(format=p.SOURCE_FORMAT, complete=True, full_validation_evidence=True,
            immutable=True, test_accessed=False, stages=self.stages, current_stage="complete")
        self.commit()
        self.calls = []

    def add_stage(self, name, gate):
        self.plan.append({"name": name})
        self.stages.append({"name": name, "complete": True, "returncode": 0, "gate": gate})

    def commit(self):
        write(self.root / "manifest.json", self.manifest)
        self.state["manifest_sha256"] = p.digest(self.root / "manifest.json")
        write(self.root / "workflow_state.json", self.state)

    def audit_call(self, args):
        self.calls.append(args)
        return copy.deepcopy(self.audit)

    def run(self, validator=None):
        return p.check(self.root, output=self.output, repo_root=self.repo, validator=validator or self.audit_call)

    def edit_stage_summary(self, arm, kind, **changes):
        path = self.root / "formal" / (kind + "_" + arm) / "summary.json"
        value = p.read_json(path)
        value.update(changes)
        write(path, value)
        stage_kind = "train" if kind == "train" else "val"
        stage = next(s for s in self.stages if s["name"] == "formal_" + stage_kind + "_" + arm)
        stage["gate"]["summary_sha256"] = p.digest(path)
        self.commit()


class PrerequisitesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.f = Fixture(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_negative_sar_still_certifies_source_without_training(self):
        cert = self.f.run()
        self.assertTrue(cert["complete"])
        self.assertEqual(cert["format"], "SAR_SOFT_SOURCE_CERTIFICATE_V1")
        self.assertFalse(cert["training"])
        self.assertFalse(cert["positive_sar_effect_required"])
        self.assertFalse(cert["new_cross_experiment_completed"])
        self.assertEqual(cert["protocol"], str(self.f.protocol))
        self.assertEqual(cert["baseline_checkpoint_sha256"], p.digest(cert["baseline_checkpoint"]))
        self.assertEqual(len(cert["artifact_bindings"]), 1800)
        self.assertFalse(any(name.endswith(".npz") for name in cert["source_bindings"]))
        self.assertTrue(self.f.output.is_file())
        self.assertEqual(len(self.f.calls), 1)
        self.assertFalse(self.f.calls[0].reports_only)
        self.assertIsNone(self.f.calls[0].output)
        for option in p.OPTIONS:
            self.assertTrue(Path(getattr(self.f.calls[0], option)).is_dir())

    def test_pending_source_does_not_write_any_certificate(self):
        self.f.state.update(complete=False, full_validation_evidence=False, current_stage="formal_train_sar_cue")
        self.f.commit()
        with self.assertRaises(p.PendingSourceError):
            self.f.run()
        self.assertFalse(self.f.output.exists())
        self.assertFalse(self.f.calls)

    def test_smoke_training_cannot_supply_formal_checkpoint(self):
        self.f.edit_stage_summary("sar_cue", "train", smoke=True, steps=2)
        with self.assertRaisesRegex(ValueError, "formal 2000-step"):
            self.f.run()
        self.assertFalse(self.f.output.exists())

    def test_changed_checkpoint_fails_before_array_audit(self):
        (self.f.root / "formal/train_baseline/last.pth").write_bytes(b"changed-checkpoint")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.f.run()
        self.assertFalse(self.f.calls)

    def test_eval_checkpoint_different_from_completed_last_fails(self):
        self.f.edit_stage_summary("sar_cue", "val", checkpoint_sha256="wrong")
        with self.assertRaisesRegex(ValueError, "completed last.pth"):
            self.f.run()

    def test_nonsynchronous_source_environment_is_rejected(self):
        self.f.manifest["runtime_environment"]["CUDA_LAUNCH_BLOCKING"] = None
        self.f.commit()
        with self.assertRaisesRegex(ValueError, "CUDA_LAUNCH_BLOCKING"):
            self.f.run()

    def test_no_unmatched_repo_override(self):
        other = Path(self.temp.name) / "other"
        (other / "bda_benchmark").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "repo-root differs"):
            p.check(self.f.root, output=self.f.output, repo_root=other, validator=self.f.audit_call)

    def test_failed_full_array_audit_does_not_create_certificate(self):
        def failed(args):
            report = self.f.audit_call(args)
            report["complete"] = False
            return report
        with self.assertRaisesRegex(ValueError, "full-array four-arm verification failed"):
            self.f.run(failed)
        self.assertFalse(self.f.output.exists())

    def test_post_audit_artifact_change_is_caught(self):
        def mutating(args):
            report = self.f.audit_call(args)
            (self.f.root / "formal/val_baseline/predictions/0001.npz").write_bytes(b"modified-during-audit")
            return report
        with self.assertRaisesRegex(ValueError, "Source changed during full-array"):
            self.f.run(mutating)
        self.assertFalse(self.f.output.exists())

    def test_existing_certificate_and_source_tree_are_never_overwritten(self):
        self.f.output.parent.mkdir()
        self.f.output.write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.f.run()
        self.assertEqual(self.f.output.read_text(), "preserve")
        with self.assertRaisesRegex(ValueError, "outside the preserved"):
            p.check(self.f.root, self.f.root / "new.json", self.f.audit_call)

    def test_q_runtime_drift_between_arms_fails(self):
        path = self.f.root / "formal/train_rgb_capacity/manifest.json"
        record = p.read_json(path)
        record["e0_runtime"] = {"cublas_workspace_config": ":4096:8"}
        write(path, record)
        with self.assertRaisesRegex(ValueError, "provenance differs from E0"):
            self.f.run()

    def test_arrayless_reports_only_never_certify(self):
        def reports_only(args):
            report = self.f.audit_call(args)
            report["reports_only"] = True
            return report
        with self.assertRaisesRegex(ValueError, "full-array four-arm verification failed"):
            self.f.run(reports_only)


if __name__ == "__main__":
    unittest.main()
