"""CPU contracts for diagnostic bounds, failure capture and old evidence."""
import argparse
import importlib.util
import json
import shutil
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

spec = importlib.util.spec_from_file_location("memory_debug_runner", Path(__file__).with_name("run_debug.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_relocated_tool_verifies_its_own_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(__file__).resolve().parent
            relocated = Path(folder) / "New" / "BRIGHT-new-tools_v1" / "memory_replay_debug"
            shutil.copytree(source, relocated)
            with patch.object(runner, "__file__", str(relocated / "run_debug.py")):
                ledger = runner.package_check()
            self.assertEqual(len(ledger), 5)
            self.assertTrue(all(Path(name).parent == relocated for name in ledger))

    def test_numeric_is_loaded_from_tool_directory_despite_module_collision(self):
        with patch.dict("sys.modules", {"numeric": SimpleNamespace(compare_tokens=lambda *args: "wrong")}):
            module = runner.diagnostic_numeric()
            self.assertEqual(Path(module.__file__).resolve(), Path(__file__).with_name("numeric.py").resolve())
            self.assertNotEqual(module.compare_tokens(torch.ones(1), torch.ones(1, dtype=torch.float16)), "wrong")

    def test_auto_find_failure_works_without_shell_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "run_old" / "keys_smoke"
            target.mkdir(parents=True)
            record = dict(complete=False, error="ValueError: " + runner.ERROR + " finite")
            runner.write(target / "summary.json", record)
            other = root / "run_new" / "keys_smoke"
            other.mkdir(parents=True)
            runner.write(other / "summary.json", dict(complete=True))
            self.assertEqual(runner.find_failure(root), target)
            self.assertEqual(runner.read(target / "summary.json"), record)

    def test_refuses_successful_explicit_record(self):
        with tempfile.TemporaryDirectory() as folder:
            runner.write(Path(folder) / "summary.json", dict(complete=True, error=runner.ERROR))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                runner.find_failure(folder, folder)

    def test_empty_record_tree_is_not_a_false_success(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "No failed"):
                runner.find_failure(folder)

    def test_real_traceback_captures_raw_locals_after_cleanup(self):
        marks = []
        def build(cli):
            raw = torch.tensor([[.25, .250244]], dtype=torch.float32)
            token_agreement = dict(finite=True, passed=False)
            try:
                raise ValueError(runner.ERROR + " failed")
            finally:
                marks.append("cleaned")
        builder = SimpleNamespace(run=build)
        try:
            builder.run(None)
        except ValueError as error:
            captured = runner.failed_frame(error, builder)
            self.assertEqual(marks, ["cleaned"])
            self.assertEqual(captured["raw"].dtype, torch.float32)
            self.assertTrue(captured["token_agreement"]["finite"])

    def test_outliers_preserve_original_member_identity(self):
        report = dict(outliers=[dict(coordinate=[1, 5]), dict(coordinate=[0, 9])])
        runner.annotate_outliers(report, torch.tensor([41, 900]), torch.tensor([7, 8]))
        self.assertEqual(report["outliers"][0]["original_bank_row"], 900)
        self.assertEqual(report["outliers"][0]["region_id"], 8)
        self.assertEqual(report["outliers"][1]["original_bank_row"], 41)

    def run_fake(self, build, folder, observe=None):
        old = Path(folder) / "old"
        old.mkdir()
        runner.write(old / "summary.json", dict(image="train_image"))
        shared = SimpleNamespace(source_hash=lambda: "source")
        prepared = (SimpleNamespace(run=build), shared, old, dict(image="train_image"), {}, {})
        cli = argparse.Namespace(output=str(Path(folder) / "new"), protocol="e0", plan_only=False)
        with patch.object(runner, "prepare", return_value=prepared), patch.object(runner, "observe_failure", side_effect=observe) as observed:
            result = runner.run(cli)
            return result, observed

    def test_one_successful_smoke_never_authorizes_training_or_fix_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = []
            result, observed = self.run_fake(lambda cli: calls.append(cli), folder)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].limit_images, 4)
            self.assertTrue(result["diagnostic_complete"])
            self.assertFalse(result["historical_failure_resolved"])
            self.assertFalse(result["downstream_authorized"])
            self.assertFalse(result["training"])
            self.assertFalse(result["test_accessed"])
            self.assertFalse(observed.called)

    def test_caught_finite_token_failure_observed_once_without_retry(self):
        def build(cli):
            token_agreement = dict(finite=True, passed=False)
            current_image = "train_image"
            raise ValueError(runner.ERROR + " failed")
        def observe(local, output, summary, shared, ledger):
            self.assertEqual(local["current_image"], "train_image")
            summary.update(diagnostic_complete=True, phase="observation_complete")
        with tempfile.TemporaryDirectory() as folder:
            result, observed = self.run_fake(build, folder, observe)
            self.assertEqual(observed.call_count, 1)
            self.assertFalse(result["original_attempt"]["passed"])
            self.assertFalse(result["downstream_authorized"])

    def test_unknown_value_error_stops_instead_of_reusing_context(self):
        def build(cli):
            raise ValueError("Unrelated broken source")
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Unrelated"):
                self.run_fake(build, folder)
            result = runner.read(Path(folder) / "new" / "summary.json")
            self.assertFalse(result["diagnostic_complete"])
            self.assertFalse(result["downstream_authorized"])

    def test_nonfinite_token_failure_stops_without_probes(self):
        def build(cli):
            token_agreement = dict(finite=False, passed=False)
            raise ValueError(runner.ERROR + " nonfinite")
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                self.run_fake(build, folder)
            result = runner.read(Path(folder) / "new" / "summary.json")
            self.assertFalse(result["diagnostic_complete"])

    def test_existing_output_refused_without_changing_its_data(self):
        with tempfile.TemporaryDirectory() as folder:
            new = Path(folder) / "new"
            new.mkdir()
            sentinel = new / "evidence.txt"
            sentinel.write_text("preserve")
            with self.assertRaises(FileExistsError):
                self.run_fake(lambda cli: None, folder)
            self.assertEqual(sentinel.read_text(), "preserve")

    def test_complete_cpu_scene_uses_real_hooks_and_pool_interfaces(self):
        # Synthetic tensors only; exercises actual project capture, Q helpers,
        # region labels and PoolGuard with the production front's return keys.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        class Front(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Module()
                self.base.clf_2 = nn.Conv2d(256, 4, 1)
            def forward(self, pre, post):
                feature = torch.ones(1, 256, 2, 2)
                pixel = F.interpolate(self.base.clf_2(feature), (8, 8), mode="bilinear", align_corners=False)
                tokens = torch.full((1, 1, 128), .25)
                tokens[0, 0, 0] = .250244140625
                return dict(tokens=tokens, q=torch.full((1, 9, 4), 1/9),
                            ids=torch.zeros(9, 4, dtype=torch.long), count=1, shape=(2, 2),
                            mass=torch.full((1, 1), 4.), geometry=torch.ones(1, 1, 5),
                            old_pixel=pixel, loc=torch.zeros(1, 2, 8, 8), adapted_content=feature)
        front = Front().eval().requires_grad_(False)
        pre, post = torch.ones(1, 3, 8, 8), torch.ones(1, 3, 8, 8)
        result, feature = front(pre, post), torch.ones(1, 256, 2, 2)
        old = dict(tokens=torch.full((1, 128), .25, dtype=torch.float16),
                   distribution=torch.tensor([[1., 0., 0., 0.]]),
                   images=torch.tensor([0]), regions=torch.tensor([0]), geometry=torch.ones(1, 5, dtype=torch.float16))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "cache"
            (source / "shards").mkdir(parents=True)
            shard = dict(tokens=old["tokens"], distribution=old["distribution"], geometry=old["geometry"],
                         region_ids=old["regions"], mass=torch.full((1,), 4.), name="train_image", image=0)
            path = source / "shards/00000.pt"
            torch.save(shard, path)
            bundle = dict(bank_cpu=old, train_names=["train_image"], front=front,
                          cache_root=source, cache=dict(shards=[dict(path="shards/00000.pt", sha256=runner.sha(path))]),
                          runtime_settings={}, hashes=dict(bank_sha256="original"), protocol_sha256="e0",
                          args=SimpleNamespace(pool_mode="native"))
            local = dict(bundle=bundle, result=result, feature=feature, region_ids=torch.tensor([0]),
                         selected_images=[0], index=0, current_image="train_image", target=torch.zeros(1, 8, 8, dtype=torch.long),
                         replay_tokens=result["tokens"][0].half().float(), pre=pre, post=post,
                         guard=SimpleNamespace(restored=True, failures=[]), own="source", image_ledger={})
            output = root / "output"
            output.mkdir()
            summary = dict(historical_failure_resolved=False, downstream_authorized=False)
            shared = SimpleNamespace(verify_all=lambda *args: dict(passed=True))
            runner.observe_failure(local, output, summary, shared, {})
            self.assertTrue(summary["diagnostic_complete"])
            self.assertEqual([x["name"] for x in summary["repeats"]], ["no_grad_1", "no_grad_2", "inference_mode_1"])
            self.assertTrue(all(x["native_a1"]["passed"] for x in summary["repeats"]))
            self.assertTrue(summary["baseline"]["cpu_vs_builder_gpu_cast_exact"])
            self.assertEqual(summary["builder_stored_outliers"]["outliers"][0]["coordinate"], [0, 0])
            self.assertEqual(summary["builder_stored_outliers"]["outliers"][0]["original_bank_row"], 0)
            self.assertTrue(summary["soft_label_replay"]["passed"])
            self.assertFalse(summary["downstream_authorized"])
            self.assertTrue((output / "failure_scene.pt").is_file())
            self.assertFalse(front.base.clf_2._forward_hooks)
            self.assertFalse(front.base.clf_2._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
