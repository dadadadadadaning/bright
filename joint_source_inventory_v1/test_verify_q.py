"""CPU contracts with the real pinned formal-Q validator and synthetic files."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import verify_surviving_q as check

TOOL = Path(os.environ.get("BRIGHT_ORIGINAL_TOOL", "/workspace/bright-followup/lr_study_source/original_fix2"))


class VerifyQTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (TOOL / "test_fresh_source.py").is_file():
            raise RuntimeError("Set BRIGHT_ORIGINAL_TOOL to the exact fix2 package for validator tests")
        cls.original = check.load_verified_source(TOOL)
        # Reuse the original author's test fixture, not a copy of validation.
        spec = importlib.util.spec_from_file_location("_verify_q_original_fixture", TOOL / "test_fresh_source.py")
        cls.fixture_module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"fresh_source": cls.original}):
            spec.loader.exec_module(cls.fixture_module)

    def setUp(self):
        self.fixture = self.fixture_module.FreshQSourceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        split_dir = f.repo / "bda_benchmark" / "data"
        split_dir.mkdir()
        f.train_list = split_dir / "train_set.txt"
        f.val_list = split_dir / "val_set.txt"
        f.train_list.write_text("".join("train_%d\n" % i for i in range(2918)))
        f.val_list.write_text("".join("val_%d\n" % i for i in range(450)))
        for split in ("train", "val"):
            value = self.original.digest(getattr(f, split + "_list"))
            f.payload[split + "_sha256"] = value
            f.manifest[split + "_sha256"] = value
        f._training_val_journal()
        f._write(independent=False)
        f.e0.unlink()
        f.val_dir.mkdir(exist_ok=True)
        (f.val_dir / "summary.json").write_text(json.dumps({"complete": True}))
        self.args = SimpleNamespace(repo=str(f.repo), q_checkpoint=str(f.checkpoint), a1_checkpoint=str(f.a1),
                                    tool_dir=str(TOOL), train_list=None, val_list=None, dataset=None)
        self.addCleanup(mock.patch.stopall)
        self.loader = mock.patch.object(check, "load_verified_source", return_value=self.original).start()
        self.dependencies = mock.patch.object(self.original, "_dependencies", return_value=f.deps).start()
        self.validation = mock.patch.object(self.original, "_validate_formal", wraps=self.original._validate_formal).start()
        mock.patch.object(check, "AUDITED_A1_SHA256", self.original.digest(f.a1)).start()

    def run_check(self):
        return check.verify(self.args, emit=lambda _: None)

    def test_real_formal_validator_works_without_e0_or_independent_result(self):
        result = self.run_check()
        self.assertTrue(result["validation_evidence"]["formal_reused_validator_passed"], result["errors"])
        self.validation.assert_called_once()
        self.dependencies.assert_called_once_with(self.fixture.repo)
        self.assertEqual(self.validation.call_args.args[-1], self.original.TRAINING_EVIDENCE)
        self.assertTrue(result["validation_evidence"]["independent_summary_exists"])
        self.assertFalse(result["validation_evidence"]["independent_result_exists"])
        self.assertFalse(result["historical_e0_reverified"])
        self.assertFalse(result["source_ready_for_training"])
        self.assertFalse(result["cuda_forward"])
        self.assertFalse(result["test_accessed"])
        self.assertFalse(result["a1_weights_loaded"])
        self.assertEqual(len(result["splits"]["train"]["matching_paths"]), 1)

    def test_only_q_is_loaded_with_torch_cpu(self):
        torch = self.fixture_module.torch
        with mock.patch.object(torch, "load", wraps=torch.load) as load:
            result = self.run_check()
        self.assertEqual(load.call_count, 1)
        self.assertEqual(load.call_args.args[0], self.fixture.checkpoint)
        self.assertEqual(load.call_args.kwargs["map_location"], "cpu")
        self.assertFalse(result["a1_weights_loaded"])

    def test_changed_a1_blocks_formal_validation(self):
        self.fixture.a1.write_bytes(b"changed a1")
        result = self.run_check()
        self.assertFalse(result["validation_evidence"]["formal_reused_validator_passed"])
        self.assertTrue(any("A1" in error for error in result["errors"]))

    def test_wrong_split_hash_rejected_but_q_still_checked(self):
        self.fixture.train_list.write_text("changed\n")
        result = self.run_check()
        self.assertTrue(result["validation_evidence"]["formal_reused_validator_passed"])
        self.assertEqual(result["splits"]["train"]["matching_paths"], [])
        self.assertFalse(result["cpu_foundation_verified"])

    def test_changed_formal_journal_rejected_by_real_validator(self):
        path = self.fixture.train_dir / "validation.jsonl"
        row = json.loads(path.read_text())
        row["image_count"] = 2
        path.write_text(json.dumps(row) + "\n")
        result = self.run_check()
        self.assertFalse(result["validation_evidence"]["formal_reused_validator_passed"])
        self.assertTrue(any("formal Q validation" in error for error in result["errors"]))

    def test_missing_summary_does_not_stop_split_discovery(self):
        (self.fixture.train_dir / "summary.json").unlink()
        result = self.run_check()
        self.assertTrue(result["execution_complete"])
        self.assertTrue(result["splits"]["disjoint"])
        self.assertTrue(any("Formal Q summary" in error for error in result["errors"]))

    def test_nonfinite_adapter_reported_without_loading_a1(self):
        f = self.fixture
        self.fixture_module.torch.save(dict(f.payload, adapter={"bad": self.fixture_module.torch.tensor([float("nan")])}), f.checkpoint)
        result = self.run_check()
        self.assertFalse(result["adapter"]["all_tensors_finite"])
        self.assertFalse(result["cpu_foundation_verified"])

    def test_best_checkpoint_path_cannot_pass_as_formal_last(self):
        path = self.fixture.checkpoint.with_name("best.pth")
        path.write_bytes(self.fixture.checkpoint.read_bytes())
        self.args.q_checkpoint = str(path)
        result = self.run_check()
        self.assertFalse(result["q"]["formal_last_path"])
        self.assertFalse(result["validation_evidence"]["formal_reused_validator_passed"])
        self.validation.assert_not_called()

    def test_nonfinite_metadata_is_reportable_not_dropped(self):
        f = self.fixture
        payload = dict(f.payload, config={"width": float("nan")})
        self.fixture_module.torch.save(payload, f.checkpoint)
        result = self.run_check()
        self.assertIsNone(result["checkpoint_metadata"]["config"]["width"])
        self.assertTrue(any("Nonfinite metadata" in error for error in result["errors"]))
        json.dumps(result, allow_nan=False)
        rendered = check.render(result)
        self.assertIn("e0_runtime", rendered)
        self.assertIn(result["q"]["sha256"], rendered)

    def test_no_test_list_is_opened_even_if_explicit_override(self):
        test = self.fixture.repo / "bda_benchmark/test_set.txt"
        test.write_text("never_read\n")
        self.args.train_list = str(test)
        with mock.patch.object(self.original, "digest", wraps=self.original.digest) as digest:
            result = self.run_check()
        self.assertNotIn(test, [Path(call.args[0]) for call in digest.call_args_list])
        self.assertEqual(result["splits"]["train"]["matching_paths"], [])

    def test_complete_cpu_inventory_still_not_training_ready(self):
        f = self.fixture
        data = f.repo / "bda_benchmark/dataset_root"
        for directory in ("pre-event", "post-event", "target"):
            (data / directory).mkdir(parents=True, exist_ok=True)
        with mock.patch.object(check, "check_dataset", return_value=dict(path=str(data), complete=True)):
            result = self.run_check()
        self.assertTrue(result["cpu_foundation_verified"], result["errors"])
        self.assertFalse(result["source_ready_for_training"])
        self.assertFalse(result["memory_built"])


class DiscoveryTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_skips_pfl_conda_git_and_symlink_directories(self):
        bda = self.root / "bda_benchmark"
        bda.mkdir()
        for directory in ("PFLlib-master", "PFLlip-master", "conda-envs", ".git"):
            (bda / directory).mkdir()
            (bda / directory / "train_set.txt").write_text("hidden")
        outside = self.root / "outside"
        outside.mkdir(); (outside / "train_set.txt").write_text("hidden")
        (bda / "linked").symlink_to(outside, target_is_directory=True)
        (bda / "val_set.txt").write_text("visible")
        result = check.discover(self.root)
        self.assertEqual(result["train"], [])
        self.assertEqual(result["val"], [str(bda / "val_set.txt")])

    def test_dataset_checks_only_expected_paths_without_reading_images(self):
        names = dict(train=["train0", "train1"], val=["val0"])
        for name in names["train"] + names["val"]:
            for folder, suffix in (("pre-event", "_pre_disaster.tif"), ("post-event", "_post_disaster.tif"), ("target", "_building_damage.tif")):
                (self.root / folder).mkdir(exist_ok=True)
                (self.root / folder / (name + suffix)).write_bytes(b"not decoded")
        with mock.patch.object(Path, "open", side_effect=AssertionError("Image content opened")):
            row = check.check_dataset(self.root, names, emit=lambda _: None)
        self.assertTrue(row["complete"])
        self.assertFalse(row["image_contents_read"])
        (self.root / "target/val0_building_damage.tif").unlink()
        row = check.check_dataset(self.root, names, emit=lambda _: None)
        self.assertFalse(row["complete"])
        self.assertEqual(row["splits"]["val"]["missing_count"], 1)

    def test_changed_validator_blocked_before_import(self):
        (self.root / "fresh_source.py").write_text("raise AssertionError('must not execute')")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            check.load_verified_source(self.root)


if __name__ == "__main__":
    unittest.main()
