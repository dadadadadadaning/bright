"""Synthetic filesystem contracts; no Torch, weights or CUDA execution."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import source_plan as plan


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, (dict, list)) else value)
    return path


class SourcePlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        anchor = Path(os.environ.get("BRIGHT_ANCHOR_TOOL", str(
            Path(__file__).resolve().parents[1] / "joint_anchor_study_v1")))
        spec = importlib.util.spec_from_file_location("_source_plan_test_legacy", anchor / "legacy.py")
        self.bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bridge)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(plan, "_bridge", return_value=self.bridge).start()
        self.packages = mock.patch.object(self.bridge, "_package_files", return_value={}).start()
        self.original = self.root / "original"
        write(self.original / "SHA256SUMS.json", {})
        self.repo = self.root / "repo"
        for name in ("shared.py", "paired.py", "core.py"):
            write(self.repo / "bda_benchmark/experiments/damageformer_srac_memory_rebuild_v3" / name, "source")
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.train = write(self.root / "train.txt", "train")
        self.val = write(self.root / "val.txt", "val")
        self.e0 = write(self.root / "e0.json", {"arguments": {"dataset": str(self.dataset)},
                 "input_paths": {"train_list": str(self.train), "val_list": str(self.val)}})
        self.q = write(self.root / "formal/train_baseline/last.pth", "q bytes")
        self.a1 = write(self.root / "a1.pth", "a1 bytes")
        write(self.q.parent / "summary.json", {"complete": True, "smoke": False,
                 "arm": "baseline", "steps": 2000, "full_validation_evidence": True})
        write(self.q.parent / "manifest.json", {})
        evidence = write(self.q.parent / "validation.jsonl", "val evidence")
        self.protocol = write(self.root / "source.json", {
            "format": "SOFT_MEMORY_Q_SOURCE_V1", "test_accessed": False,
            "historical_e0_source_reverified": False, "device": "cuda:0", "repo": str(self.repo),
            "q_checkpoint": str(self.q), "q_checkpoint_sha256": plan._digest(self.q),
            "a1_checkpoint": str(self.a1), "a1_sha256": plan._digest(self.a1),
            "e0_metadata_protocol": str(self.e0), "e0_metadata_protocol_sha256": plan._digest(self.e0),
            "evidence_level": "training_final_val450", "evidence_sha256": plan._digest(evidence)})
        self.members = self.root / "membership"
        for filename in ("paired_bank.pt", "images.jsonl"):
            write(self.members / filename, "member " + filename)
        write(self.members / "protocol.json", {})
        membership = dict(format=plan.MEMBERSHIP_FORMAT, complete=True, smoke=False,
            train_only=True, test_accessed=False, training=False,
            paired_bank_sha256=plan._digest(self.members / "paired_bank.pt"),
            rebuild_protocol_sha256=plan._digest(self.members / "protocol.json"),
            images_journal_sha256=plan._digest(self.members / "images.jsonl"))
        write(self.members / "summary.json", membership)
        write(self.members / "manifest.json", membership)
        self.parent = self.root / "parent_old"
        self.values = self.root / "values_old"
        self.config = write(self.root / "fresh.json", dict(repo=str(self.repo), protocol=str(self.protocol),
            membership_root=str(self.members), bank_root=str(self.parent), values=str(self.values)))
        self.output = self.root / "new_run"

    def go(self):
        return plan.prepare_plan(self.config, self.root / "anchor", self.original, self.output)

    def make_parent(self, complete=True):
        for filename in ("paired_bank.pt", "images.jsonl"):
            write(self.parent / filename, "parent " + filename)
        write(self.parent / "protocol.json", {})
        record = dict(format=plan.PARENT_FORMAT, complete=complete, smoke=False,
            execution_complete=complete, train_only=True, test_accessed=False,
            source_protocol_sha256=plan._digest(self.protocol),
            paired_bank_sha256=plan._digest(self.parent / "paired_bank.pt"),
            rebuild_protocol_sha256=plan._digest(self.parent / "protocol.json"),
            images_journal_sha256=plan._digest(self.parent / "images.jsonl"))
        write(self.parent / "manifest.json", record)
        write(self.parent / "summary.json", record)

    def make_values(self):
        for filename in ("slot_bank.pt", "images.jsonl"):
            write(self.values / filename, "values " + filename)
        write(self.values / "input_fingerprints.json", {})
        record = dict(format=plan.VALUES_FORMAT, complete=True, smoke=False,
            execution_complete=True, train_only=True, test_accessed=False,
            source_protocol_sha256=plan._digest(self.protocol),
            paired_bank_sha256=plan._digest(self.parent / "paired_bank.pt"),
            file_sha256={f: plan._digest(self.values / f) for f in
                         ("slot_bank.pt", "images.jsonl", "input_fingerprints.json")})
        write(self.values / "manifest.json", record)
        record = dict(record, manifest_sha256=plan._digest(self.values / "manifest.json"),
                      slot_bank_sha256=plan._digest(self.values / "slot_bank.pt"))
        write(self.values / "summary.json", record)

    def test_missing_banks_plans_new_permanent_paths_without_writes(self):
        before = sorted(str(p) for p in self.root.rglob("*"))
        result = self.go()
        self.assertTrue(result["ready_for_build_or_reuse"])
        self.assertEqual(result["bank_actions"], ["parent", "build"])
        self.assertEqual(result["config"]["bank_root"], str(self.output / "parent"))
        self.assertNotIn("command", result["config"])
        self.assertEqual(before, sorted(str(p) for p in self.root.rglob("*")))

    def test_complete_banks_reused(self):
        self.make_parent(); self.make_values()
        result = self.go()
        self.assertTrue(result["ready_for_build_or_reuse"], result)
        self.assertEqual(result["bank_actions"], [])
        self.assertEqual(result["config"]["values"], str(self.values))

    def test_only_values_missing_does_not_rebuild_parent(self):
        self.make_parent()
        result = self.go()
        self.assertTrue(result["ready_for_build_or_reuse"])
        self.assertEqual(result["bank_actions"], ["build"])
        self.assertEqual(result["config"]["bank_root"], str(self.parent))

    def test_deleted_a1_and_q_both_reported(self):
        self.q.unlink(); self.a1.unlink()
        result = self.go()
        self.assertFalse(result["ready_for_build_or_reuse"])
        self.assertIn(str(self.q), result["foundation_missing"])
        self.assertIn(str(self.a1), result["foundation_missing"])

    def test_missing_membership_not_replaced_by_new_selection(self):
        (self.members / "paired_bank.pt").unlink()
        result = self.go()
        self.assertFalse(result["ready_for_build_or_reuse"])
        self.assertIn(str(self.members / "paired_bank.pt"), result["foundation_missing"])

    def test_corrupt_complete_parent_blocks_not_rebuild(self):
        self.make_parent()
        write(self.parent / "paired_bank.pt", "changed bytes")
        result = self.go()
        self.assertFalse(result["ready_for_build_or_reuse"])
        self.assertEqual(result["bank_inspection"]["parent"]["status"], "invalid")
        self.assertNotIn("parent", result["bank_actions"])

    def test_corruption_not_hidden_by_another_missing_artifact(self):
        self.make_parent()
        write(self.parent / "paired_bank.pt", "changed bytes")
        (self.parent / "images.jsonl").unlink()
        result = self.go()
        self.assertFalse(result["ready_for_build_or_reuse"])
        self.assertEqual(result["bank_inspection"]["parent"]["status"], "invalid")

    def test_incomplete_parent_rebuilt_without_overwrite(self):
        self.make_parent(complete=False)
        old = (self.parent / "manifest.json").read_bytes()
        result = self.go()
        self.assertTrue(result["ready_for_build_or_reuse"])
        self.assertEqual(result["bank_actions"], ["parent", "build"])
        self.assertEqual((self.parent / "manifest.json").read_bytes(), old)

    def test_changed_frozen_model_blocks(self):
        write(self.a1, "different model")
        result = self.go()
        self.assertFalse(result["ready_for_build_or_reuse"])
        self.assertTrue(any("SHA256 differs" in e for e in result["metadata_errors"]))

    def test_missing_formal_q_evidence_blocks(self):
        (self.q.parent / "validation.jsonl").unlink()
        result = self.go()
        self.assertFalse(result["ready_for_build_or_reuse"])
        self.assertIn(str(self.q.parent / "validation.jsonl"), result["foundation_missing"])

    def test_values_other_parent_blocks(self):
        self.make_parent(); self.make_values()
        manifest = plan._read(self.values / "manifest.json")
        manifest["paired_bank_sha256"] = "0" * 64
        write(self.values / "manifest.json", manifest)
        summary = plan._read(self.values / "summary.json")
        summary["manifest_sha256"] = plan._digest(self.values / "manifest.json")
        write(self.values / "summary.json", summary)
        result = self.go()
        self.assertFalse(result["ready_for_build_or_reuse"])
        self.assertIn("Existing slot bank binds a different parent bank", result["metadata_errors"])

    def test_protected_overlap_and_existing_destinations_block(self):
        (self.output / "parent").mkdir(parents=True)
        result = self.go()
        self.assertFalse(result["ready_for_build_or_reuse"])
        self.assertTrue(any("already exists" in e for e in result["metadata_errors"]))
        self.output = self.members
        result = self.go()
        self.assertFalse(result["ready_for_build_or_reuse"])
        self.assertTrue(any("overlaps protected" in e for e in result["metadata_errors"]))

    def test_reused_parent_missing_source_image_blocks(self):
        self.make_parent()
        manifest = plan._read(self.parent / "manifest.json")
        missing = self.root / "removed_image.png"
        manifest["source_image_fingerprints"] = {str(missing): "0" * 64}
        write(self.parent / "manifest.json", manifest)
        result = self.go()
        self.assertFalse(result["ready_for_build_or_reuse"])
        self.assertIn(str(missing), result["foundation_missing"])

    def test_unknown_bank_format_blocks(self):
        self.make_parent()
        manifest = plan._read(self.parent / "manifest.json")
        manifest["format"] = "some_other_bank"
        write(self.parent / "manifest.json", manifest)
        result = self.go()
        self.assertFalse(result["ready_for_build_or_reuse"])
        self.assertTrue(any("Unsupported bank format" in e for e in result["metadata_errors"]))

    def test_config_bad_value_not_silently_changed(self):
        config = plan._read(self.config)
        config["topk"] = 128
        write(self.config, config)
        with self.assertRaises(ValueError):
            self.go()


if __name__ == "__main__":
    unittest.main()
