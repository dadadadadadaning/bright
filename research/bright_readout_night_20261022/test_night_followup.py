import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import night_followup as night


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def completed(folder, variant="original", weight=0.0, smoke=False):
    plan = dict(variant=variant, effective_mode=variant, seed=3407, endpoint_step=2500,
                read_weight=weight, smoke_steps=3, test_accessed=False, extension_files=night.IDENTITY,
                source={"source_run": "/same/source"}, addon_dir="/same/addon", addon_sha256="abc",
                post500_lr_multiplier=.2, scheduler_horizon=5000, warmup_steps=500, maximum_lr=.0001,
                weight_decay=.005, grad_clip=1.0, optimizer="AdamW", fixed_train_count=450, support_arm="d")
    summary = dict(format=night.FORMAT, variant=variant, seed=3407, smoke=smoke, test_accessed=False,
                   complete=True, phase="complete_smoke" if smoke else "complete", training_complete=True,
                   full_validation_evidence=not smoke, last_complete_step=3 if smoke else 2500,
                   endpoint_step=3 if smoke else 2500)
    put(folder / "summary.json", summary)
    put(folder / "plan.json", plan)
    put(folder / "manifest.json", {"plan": plan})
    report = dict(format=night.FORMAT + "_EVALUATION", complete=True, full_validation_evidence=True,
                  final_integrity_pending=False, smoke=False, test_accessed=False, journal_verified=True,
                  step=2500, seed=3407, variant=variant, split="val", image_count=450,
                  requested_names=[str(i) for i in range(450)])
    path = folder / "val_step_002500" / "result.json"
    put(path, report)
    path.with_suffix(".json.sha256").write_text(night.sha(path) + "  result.json\n")
    return plan


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def queue(self):
        return night.Queue(argparse.Namespace(root=self.root, package=self.root, wait_hours=1,
                                             max_hours=1, include_read_aux=False))

    def test_complete_pair_and_weight(self):
        before = completed(self.root / "original")
        completed(self.root / "aux", "read_aux", .01)
        self.assertIsNotNone(night.inspect_run(self.root / "aux", "read_aux", .01, reference_plan=before))
        with self.assertRaisesRegex(ValueError, "plan differs"):
            night.inspect_run(self.root / "aux", "read_aux", .02)

    def test_pending_is_not_success_and_failed_is_not_retried(self):
        folder = self.root / "run"
        self.assertIsNone(night.inspect_run(folder, "original"))
        completed(folder)
        summary = night.read(folder / "summary.json")
        summary.update(complete=False, phase="evaluate_val_2500")
        put(folder / "summary.json", summary)
        self.assertIsNone(night.inspect_run(folder, "original"))
        summary.update(phase="failed", error="device assert")
        put(folder / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "Existing experiment failed"):
            night.inspect_run(folder, "original")

    def test_wrong_seed_scope_or_endpoint_rejected(self):
        folder = self.root / "run"
        for field, value in (("seed", 17), ("test_accessed", True), ("last_complete_step", 1500),
                             ("training_complete", False), ("full_validation_evidence", False)):
            completed(folder)
            summary = night.read(folder / "summary.json")
            summary[field] = value
            put(folder / "summary.json", summary)
            with self.subTest(field=field), self.assertRaises(ValueError):
                night.inspect_run(folder, "original")

    def test_complete_flag_requires_complete_unchanged_report(self):
        folder = self.root / "run"
        completed(folder)
        path = folder / "val_step_002500" / "result.json"
        report = night.read(path)
        report["image_count"] = 449
        put(path, report)
        with self.assertRaisesRegex(ValueError, "Val450"):
            night.inspect_run(folder, "original")
        completed(folder)
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            night.inspect_run(folder, "original")

    def test_changed_extension_or_reference_plan_rejected(self):
        folder = self.root / "run"
        reference = completed(folder)
        plan = night.read(folder / "plan.json")
        plan["extension_files"]["experiment.py"] = "bad"
        put(folder / "plan.json", plan)
        with self.assertRaisesRegex(ValueError, "plan differs"):
            night.inspect_run(folder, "original")
        completed(folder)
        reference["fixed_train_count"] = 300
        with self.assertRaisesRegex(ValueError, "fixed_train_count"):
            night.inspect_run(folder, "original", reference_plan=reference)

    def test_smoke_three_steps_not_full_validation(self):
        folder = self.root / "smoke"
        completed(folder, "no_label", smoke=True)
        self.assertIsNotNone(night.inspect_run(folder, "no_label", smoke=True))
        with self.assertRaises(ValueError):
            night.inspect_run(folder, "no_label", smoke=False)

    def test_active_lease_is_checked_without_retaining_lock(self):
        path = self.root / "active.lock"
        with path.open("a+") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertFalse(night.lock_available(path))
        self.assertTrue(night.lock_available(path))
        self.assertTrue(night.lock_available(path))

    def test_existing_incomplete_output_never_launches(self):
        queue = self.queue()
        (self.root / "no_label_smoke").mkdir()
        queue.run_child = lambda *a, **k: self.fail("Must not launch")
        with self.assertRaisesRegex(ValueError, "incomplete existing output"):
            queue.experiment("no_label", 0.0, "no_label", {})

    def test_complete_matching_outputs_are_reused(self):
        queue = self.queue()
        plan = completed(self.root / "reference")
        completed(self.root / "no_label_smoke", "no_label", smoke=True)
        completed(self.root / "no_label", "no_label")
        queue.run_child = lambda *a, **k: self.fail("Must not retrain complete output")
        queue.experiment("no_label", 0.0, "no_label", plan)
        self.assertEqual([s["state"] for s in queue.status["stages"]], ["reused_complete"]*2)

    def test_failed_baseline_blocks_queue(self):
        queue = self.queue()
        folder = self.root / "failed"
        completed(folder)
        summary = night.read(folder / "summary.json")
        summary.update(phase="failed", complete=False, error="route invalid")
        put(folder / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "Existing experiment failed"):
            queue.wait_baseline(folder, "original")

    def test_child_failure_propagates_and_preserves_log(self):
        queue = self.queue()
        with self.assertRaisesRegex(ValueError, "failed"):
            queue.run_child("failure", [sys.executable, "-c", "print('diagnostic'); raise SystemExit(7)"])
        self.assertEqual(queue.status["stages"][-1]["returncode"], 7)
        self.assertIn("diagnostic", (queue.output / "failure.log").read_text())

    def test_timeout_stops_only_owned_child(self):
        queue = self.queue()
        queue.deadline = time.monotonic() + .15
        with self.assertRaises(TimeoutError):
            queue.run_child("timeout", [sys.executable, "-c", "import time; time.sleep(20)"])
        entry = queue.status["stages"][-1]
        self.assertEqual(entry["state"], "interrupted")
        self.assertLess(entry["returncode"], 0)

    def test_sigterm_controller_cleans_owned_child(self):
        code = (
            "import argparse,signal,sys; from pathlib import Path; import night_followup as n; "
            "signal.signal(signal.SIGTERM,n.stop_signal); "
            "q=n.Queue(argparse.Namespace(root=Path(sys.argv[1]),package=Path(sys.argv[1]),"
            "wait_hours=1,max_hours=1,include_read_aux=False)); "
            "q.run_child('owned',[sys.executable,'-c','import time; time.sleep(30)'])"
        )
        controller = subprocess.Popen([sys.executable, "-B", "-c", code, str(self.root)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        child_pid = None
        try:
            end = time.monotonic()+5
            while time.monotonic() < end:
                statuses = list(self.root.glob("night_followup_*/status.json"))
                if statuses:
                    entries = night.read(statuses[0]).get("stages", [])
                    if entries and entries[-1].get("pid"):
                        child_pid = entries[-1]["pid"]
                        break
                time.sleep(.02)
            self.assertIsNotNone(child_pid)
            controller.terminate()
            controller.wait(timeout=5)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
        finally:
            if controller.poll() is None:
                controller.kill()
                controller.wait()
            if child_pid:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_command_preserves_reference_recipe_and_never_test(self):
        plan = completed(self.root / "ref")
        argv = night.command(self.root, self.root / "aux", "read_aux", .01, "train", plan)
        self.assertEqual(argv[4], "train")
        self.assertEqual(argv[argv.index("--source-run")+1], "/same/source")
        self.assertEqual(argv[argv.index("--steps")+1], "2500")
        self.assertEqual(argv[argv.index("--read-weight")+1], "0.01")
        self.assertNotIn("test", argv)


if __name__ == "__main__":
    unittest.main()
