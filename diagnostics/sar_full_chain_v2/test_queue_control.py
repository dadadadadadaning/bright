"""Independent safety tests; the integration fixture owns its only subprocess."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import queue_control as queue


class FakeProcesses:
    def __init__(self, identity):
        self.row = copy.deepcopy(identity)
        self.signals = []
        self.closed = []
        self.children = []
        self.stop_hook = None
        self.open_hook = None
        self.stop_works = True
        self.term_works = True
        self.terminating = False
        self.now = 0.0

    def identity(self, pid):
        return copy.deepcopy(self.row)

    def open_pidfd(self, pid):
        if self.open_hook:
            self.open_hook()
        return 83

    def send(self, fd, signum):
        if fd != 83:
            raise AssertionError("Unexpected process descriptor")
        self.signals.append(signum)
        if signum == signal.SIGSTOP:
            if self.stop_works:
                self.row["state"] = "T"
            if self.stop_hook:
                self.stop_hook()
        elif signum == signal.SIGTERM:
            self.terminating = True
        elif signum == signal.SIGCONT:
            if self.terminating and self.term_works:
                self.row = None
            elif self.row:
                self.row["state"] = "S"
        else:
            raise AssertionError("Unauthorized signal")

    def close(self, fd):
        self.closed.append(fd)

    def descendants(self, pid):
        return list(self.children)

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class QueueSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.parent = self.base / "runs"
        self.root = self.parent / "run_queue"
        self.root.mkdir(parents=True)
        self.script = self.base / "tool" / "chain.py"
        self.script.parent.mkdir()
        self.script.write_text("# unchanged queued launcher\n")
        self.launch = {
            "format": "SAR_FULL_CHAIN_V1", "root": str(self.root),
            "python": sys.executable, "q_root": str(self.root / "sar_q"),
            "soft_root": str(self.root / "sar_soft"),
            "bindings": {str(self.script): self.digest()},
        }
        self.launcher = {"root": str(self.root), "pid": 4321,
                         "q_root": self.launch["q_root"], "soft_root": self.launch["soft_root"]}
        self.state = {"format": "SAR_FULL_CHAIN_V1", "pid": 4321,
                      "pid_start_ticks": "999", "complete": False,
                      "phase": "waiting_declared_memory", "failure": None,
                      "q_root": self.launch["q_root"], "soft_root": self.launch["soft_root"]}
        self.write_metadata()
        (self.parent / "latest.json").write_text(json.dumps({"root": str(self.root)}))
        self.proc = FakeProcesses({"pid": 4321, "start_ticks": "999", "state": "S", "ppid": 1,
                                   "argv": [sys.executable, "-u", str(self.script), "--worker-root",
                                            str(self.root), "--inherited-lock-fd", "8"]})

    def tearDown(self):
        self.tmp.cleanup()

    def digest(self):
        return hashlib.sha256(self.script.read_bytes()).hexdigest()

    def write_metadata(self):
        for filename, value in (("launch.json", self.launch), ("launcher.json", self.launcher),
                                ("chain_state.json", self.state)):
            (self.root / filename).write_text(json.dumps(value))

    def write_q(self, **changes):
        value = {"complete": False, "current_stage": "waiting_gpu", "stages": [], "failure": None}
        value.update(changes)
        folder = Path(self.launch["q_root"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "workflow_state.json").write_text(json.dumps(value))

    def cancel(self):
        return queue.cancel_queued(self.parent, _processes=self.proc,
                                   _clock=self.proc.clock, _sleep=self.proc.sleep)

    def test_no_latest_does_not_inspect_or_signal(self):
        (self.parent / "latest.json").unlink()
        self.assertEqual(self.cancel()["status"], "no_active_queue")
        self.assertEqual(self.proc.signals, [])

    def test_dead_launcher_has_no_signals(self):
        self.proc.row = None
        self.assertEqual(self.cancel()["status"], "no_active_queue")
        self.assertEqual(self.proc.closed, [])

    def test_both_waiting_phases_and_versions_cancel_only_launcher(self):
        for phase in queue.WAITING_PHASES:
            for version in queue.FORMATS:
                with self.subTest(phase=phase, version=version):
                    self.state.update(phase=phase, format=version)
                    self.launch["format"] = version
                    self.write_metadata()
                    self.proc.row = {"pid": 4321, "start_ticks": "999", "state": "S", "ppid": 1,
                                     "argv": [sys.executable, "-u", str(self.script), "--worker-root", str(self.root)]}
                    self.proc.signals.clear()
                    self.proc.terminating = False
                    result = self.cancel()
                    self.assertEqual(result["status"], "cancelled")
                    self.assertEqual(self.proc.signals, [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT])
                    self.assertIs(result["memory_signalled"], False)
                    self.assertIs(result["probe_signalled"], False)
                    self.assertIs(result["training_children_signalled"], False)

    def test_q_waiting_gpu_with_no_stages_can_cancel(self):
        self.state["phase"] = "sar_q"
        self.write_metadata()
        self.write_q()
        self.assertEqual(self.cancel()["status"], "cancelled")

    def test_initializing_and_other_active_phases_never_paused(self):
        for phase in ("initializing", "starting", "sar_soft", "complete", "train"):
            with self.subTest(phase=phase):
                self.state["phase"] = phase
                self.write_metadata()
                with self.assertRaises(RuntimeError):
                    self.cancel()
                self.assertEqual(self.proc.signals, [])

    def test_q_started_or_completed_refuses_even_stale_waiting_phase(self):
        for changes in ({"stages": [{"name": "train"}]}, {"complete": True}):
            with self.subTest(changes=changes):
                self.write_q(**changes)
                with self.assertRaises(RuntimeError):
                    self.cancel()
                self.assertEqual(self.proc.signals, [])

    def test_q_nonempty_stages_cannot_cancel_gpu_wait(self):
        self.state["phase"] = "sar_q"
        self.write_metadata()
        self.write_q(stages=[{"status": "running"}])
        with self.assertRaises(RuntimeError):
            self.cancel()
        self.assertEqual(self.proc.signals, [])

    def test_mismatched_pid_start_ticks_cannot_signal(self):
        self.proc.row["start_ticks"] = "1000"
        with self.assertRaisesRegex(RuntimeError, "reused"):
            self.cancel()
        self.assertEqual(self.proc.signals, [])

    def test_wrong_root_or_script_or_interpreter_cannot_signal(self):
        for index, value in ((0, "/wrong/python"), (2, "/wrong/chain.py"), (4, str(self.base))):
            with self.subTest(index=index):
                previous = self.proc.row["argv"][index]
                self.proc.row["argv"][index] = value
                with self.assertRaises(RuntimeError):
                    self.cancel()
                self.assertEqual(self.proc.signals, [])
                self.proc.row["argv"][index] = previous

    def test_changed_bound_source_cannot_signal(self):
        self.script.write_text("# changed\n")
        with self.assertRaisesRegex(RuntimeError, "unchanged bound"):
            self.cancel()
        self.assertEqual(self.proc.signals, [])

    def test_pid_reuse_between_lookup_and_pidfd_prevents_pause(self):
        self.proc.open_hook = lambda: self.proc.row.update(start_ticks="1001")
        with self.assertRaisesRegex(RuntimeError, "reused"):
            self.cancel()
        self.assertEqual(self.proc.signals, [])
        self.assertEqual(self.proc.closed, [83])

    def test_race_to_active_training_after_pause_resumes_without_termination(self):
        def start_stage():
            self.state["phase"] = "sar_q"
            self.write_metadata()
            self.write_q(current_stage="train_baseline", stages=[{"status": "running"}])
        self.proc.stop_hook = start_stage
        with self.assertRaises(RuntimeError):
            self.cancel()
        self.assertEqual(self.proc.signals, [signal.SIGSTOP, signal.SIGCONT])
        self.assertEqual(self.proc.row["state"], "S")
        self.assertEqual(self.proc.closed, [83])

    def test_active_descendant_is_not_signalled_and_parent_resumes(self):
        self.proc.children = [{"pid": 9876, "ppid": 4321}]
        with self.assertRaisesRegex(RuntimeError, "descendants"):
            self.cancel()
        self.assertEqual(self.proc.signals, [signal.SIGSTOP, signal.SIGCONT])

    def test_failure_to_confirm_stop_resumes_with_bounded_polling(self):
        self.proc.stop_works = False
        with self.assertRaisesRegex(RuntimeError, "confirm"):
            self.cancel()
        self.assertEqual(self.proc.signals, [signal.SIGSTOP, signal.SIGCONT])
        self.assertLessEqual(self.proc.now, 2.1)

    def test_ignored_term_times_out_without_kill(self):
        self.proc.term_works = False
        with self.assertRaisesRegex(RuntimeError, "15 seconds"):
            self.cancel()
        self.assertEqual(self.proc.signals, [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT])
        self.assertLessEqual(self.proc.now, 15.1)
        self.assertEqual(self.proc.closed, [83])

    def test_pidfd_unavailable_fails_before_any_signal(self):
        def unavailable():
            raise RuntimeError("pidfd support unavailable")
        self.proc.open_hook = unavailable
        with self.assertRaisesRegex(RuntimeError, "pidfd"):
            self.cancel()
        self.assertEqual(self.proc.signals, [])

    def test_external_completed_q_source_is_never_cancelled(self):
        self.launch["q_root"] = str(self.base / "external_q")
        self.state["q_root"] = self.launch["q_root"]
        self.launcher["q_root"] = self.launch["q_root"]
        self.write_metadata()
        self.write_q(complete=True, stages=[{"status": "complete"}])
        self.assertEqual(self.cancel()["status"], "cancelled")

    def test_root_outside_parent_is_rejected_before_signals(self):
        with self.assertRaisesRegex(RuntimeError, "beneath"):
            queue.cancel_queued(self.parent, root=self.base, _processes=self.proc)
        self.assertEqual(self.proc.signals, [])

    @unittest.skipUnless(sys.platform == "linux" and hasattr(os, "pidfd_open")
                         and hasattr(signal, "pidfd_send_signal"), "Linux pidfds required")
    def test_real_owned_queue_process_exits_without_signalling_other_jobs(self):
        self.script.write_text('''import json, os, signal, sys, time
from pathlib import Path
root = Path(sys.argv[sys.argv.index("--worker-root") + 1])
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
state = json.loads((root / "chain_state.json").read_text())
state.update(pid=os.getpid(), pid_start_ticks=fields[19])
(root / "chain_state.tmp").write_text(json.dumps(state))
(root / "chain_state.tmp").replace(root / "chain_state.json")
(root / "ready").write_text("ready")
while True: time.sleep(.05)
''')
        self.launch["bindings"][str(self.script)] = self.digest()
        self.write_metadata()
        child = subprocess.Popen([sys.executable, "-u", str(self.script), "--worker-root", str(self.root)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            self.launcher["pid"] = child.pid
            (self.root / "launcher.json").write_text(json.dumps(self.launcher))
            deadline = time.monotonic() + 5
            while not (self.root / "ready").is_file():
                if child.poll() is not None or time.monotonic() > deadline:
                    self.fail("Owned queue fixture did not start: " + str(child.communicate(timeout=1)))
                time.sleep(.01)
            result = queue.cancel_queued(self.parent)
            self.assertEqual(result["status"], "cancelled")
            self.assertEqual(result["pid"], child.pid)
            self.assertEqual(child.wait(timeout=3), 0)
        finally:
            if child.poll() is None:
                # Cleanup can only address this test's directly owned child.
                os.kill(child.pid, signal.SIGCONT)
                child.terminate()
                child.wait(timeout=3)
            if child.stderr:
                child.stderr.close()


class ProcParsingTests(unittest.TestCase):
    def test_proc_comm_parentheses_start_ticks_and_descendants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for pid, ppid, state in ((100, 1, "S"), (101, 100, "S"), (102, 101, "S"), (103, 100, "Z")):
                folder = root / str(pid)
                folder.mkdir()
                fields = [state, str(ppid)] + ["0"] * 17 + ["76543"] + ["0"] * 5
                (folder / "stat").write_text(str(pid) + " (worker ) with spaces) " + " ".join(fields))
                (folder / "cmdline").write_bytes(b"python\0-u\0chain.py\0")
            processes = queue.LinuxProcesses(root)
            self.assertEqual(processes.identity(100)["start_ticks"], "76543")
            self.assertEqual(processes.identity(100)["argv"], ["python", "-u", "chain.py"])
            self.assertEqual({row["pid"] for row in processes.descendants(100)}, {101, 102})
            self.assertIsNone(processes.identity(103))


if __name__ == "__main__":
    unittest.main()
