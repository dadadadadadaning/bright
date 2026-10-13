"""No GPU/model fixtures: fail binding, image selection and capture restoration."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("prt_initial_diagnose", Path(__file__).with_name("diagnose.py"))
diagnose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnose)


def row(name, p=0.0, r=0.0):
    return {"image": name, "initial_head_checks": {
        "p": {"passed": p == 0, "max_abs": p},
        "r": {"passed": r == 0, "max_abs": r},
        "t": {"passed": True, "max_abs": 0.0}}}


class SelectionTests(unittest.TestCase):
    def test_maximum_per_arm_not_two_biggest_p(self):
        rows = [row("p-worst", p=.024), row("p-next", p=.007), row("r-worst", r=.00028)]
        self.assertEqual(diagnose.select_images(rows, [r["image"] for r in rows], 2),
                         ["p-worst", "r-worst"])

    def test_shared_winner_deduplicated_and_filled(self):
        rows = [row("both", p=.02, r=.03), row("second", p=.001)]
        self.assertEqual(diagnose.select_images(rows, ["both", "second"], 2), ["both", "second"])

    def test_explicit_must_be_unique_val_images(self):
        rows = [row("a", p=.01)]
        with self.assertRaises(ValueError):
            diagnose.select_images(rows, ["a"], 2, ["a", "a"])
        with self.assertRaises(ValueError):
            diagnose.select_images(rows, ["a"], 2, ["train"])


class ObservationTests(unittest.TestCase):
    def test_actual_local_return_and_existing_hook_restored(self):
        def function(value):
            token = value + 3
            return token * 2
        calls = []
        previous = sys.getprofile()
        callback = lambda frame, event, arg: calls.append(event)
        sys.setprofile(callback)
        try:
            result, captured = diagnose.call_with_locals(function, (4,), ("token",))
            self.assertEqual(result, 14)
            self.assertEqual(captured, {"token": 7})
            self.assertIs(sys.getprofile(), callback)
            self.assertIn("return", calls)
        finally:
            sys.setprofile(previous)

    def test_exception_restores_observer_without_fabricating_capture(self):
        def function():
            raise RuntimeError("original failure")
        previous = sys.getprofile()
        with self.assertRaisesRegex(RuntimeError, "original failure"):
            diagnose.call_with_locals(function, (), ("token",))
        self.assertIs(sys.getprofile(), previous)

    def test_missing_local_is_error(self):
        with self.assertRaisesRegex(RuntimeError, "not captured"):
            diagnose.call_with_locals(lambda: 4, (), ("token",))


class FailureEvidenceTests(unittest.TestCase):
    def test_relocated_shell_plan_creates_no_outputs_or_gpu_process(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / "New" / "BRIGHT-new-tools_v1" / "prt_initial_debug"
            shutil.copytree(Path(__file__).parent, package, ignore=shutil.ignore_patterns("__pycache__"))
            repo = root / "BRIGHT-master"
            (repo / "bda_benchmark").mkdir(parents=True)
            failed = root / "failed"
            result = self.fixture(failed)
            result_sha256 = diagnose.digest(result)
            output_parent = root / "new_outputs"
            env = dict(os.environ, PYTHON=sys.executable, REPO=str(repo), FAILED_RUN=str(failed),
                       OUTPUT_PARENT=str(output_parent))
            completed = subprocess.run(["bash", str(package / "run.sh"), "--plan-only"],
                                       cwd=root, env=env, text=True, capture_output=True, check=True)
            self.assertFalse(output_parent.exists())
            self.assertIn("PLAN ONLY", completed.stdout)
            self.assertIn(str(package / "diagnose.py"), completed.stdout)
            self.assertEqual(diagnose.digest(result), result_sha256)

    def fixture(self, root):
        fit = root / "prt_fit"
        initial = fit / "initial_val"
        initial.mkdir(parents=True)
        (root / "fresh_e0").mkdir()
        protocol = root / "fresh_e0/protocol.json"
        protocol.write_text("{}")
        checksum = diagnose.digest(protocol)
        header = dict(format="SRAC_REGION_HEAD_ADAPT_V1", smoke=False, test_accessed=False,
                      source_sha256="a" * 64, protocol_sha256=checksum)
        (fit / "manifest.json").write_text(json.dumps(header))
        (fit / "summary.json").write_text(json.dumps({
            **header, "complete": False, "phase": "initial_Val",
            "updates_per_arm": {"p": 0, "r": 0, "t": 0}}))
        result = dict(**{**header, "format": "SRAC_REGION_HEAD_ADAPT_EVAL_V1"},
                      complete=False, initial=True, full_validation_evidence=False,
                      image_count=450, initial_parameters={"passed": True},
                      native_a1_replay_ok=True, reproduction={"passed": True},
                      e0_metric_comparison={"passed": True}, immutable={"files": True},
                      initial_checks_passed=False)
        (initial / "result.json").write_text(json.dumps(result))
        (initial / "images.jsonl").write_text("\n".join(json.dumps(row(str(i), p=.01)) for i in range(450)))
        return initial / "result.json"

    def test_bound_untrained_failure_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "failed"
            self.fixture(root)
            files, hashes, manifest, rows = diagnose.checked_failure(root, Path(folder) / "new")
            self.assertEqual(len(rows), 450)
            self.assertEqual(hashes["protocol"], manifest["protocol_sha256"])
            self.assertEqual(len(files), 5)

    def test_failure_of_original_reproduction_is_not_initialization_case(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "failed"
            result_path = self.fixture(root)
            result = json.loads(result_path.read_text())
            result["reproduction"]["passed"] = False
            result_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "isolated"):
                diagnose.checked_failure(root, Path(folder) / "new")

    def test_refuse_output_inside_original_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "failed"
            self.fixture(root)
            with self.assertRaisesRegex(ValueError, "outside"):
                diagnose.checked_failure(root, root / "new")


if __name__ == "__main__":
    unittest.main()
