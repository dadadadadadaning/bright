"""CPU-only orchestration contracts using synthetic manifests and child artifacts.

No checkpoint tensors are loaded and no Train/Val/Test image or CUDA is accessed.
The fake executor represents child outputs; run_command has separate real-process
exit-code tests so a completed file tree is never confused with real evidence.
"""
from contextlib import redirect_stdout
import copy
import fcntl
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

_SPEC = importlib.util.spec_from_file_location(
    "_bright_anchor_chain_under_test", Path(__file__).with_name("chain.py"))
chain = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(chain)


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")


class ChainContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.output = self.base / "results" / "new_chain"
        self.args = chain.parser().parse_args([
            "run", "--output", str(self.output),
            "--anchor-tool", str(self.base / "code" / "anchor"),
            "--original-tool", str(self.base / "code" / "original"),
            "--audit-tool", str(self.base / "code" / "audit"),
            "--helpers", str(self.base / "code" / "helpers")])
        self.plan = dict(
            ready_for_build_or_reuse=True, protected_paths=[],
            bank_actions=["parent", "build"],
            config=dict(repo=str(self.base / "repo"),
                        membership_root=str(self.base / "membership"),
                        protocol=str(self.base / "source.json"), device="cuda:0",
                        workers=4, seed=3407, key_name="t128", topk=64,
                        chunk_size=32, lr=0.0001, weight_decay=0.005,
                        bank_root=str(self.output / "parent"),
                        values=str(self.output / "values")))
        self.calls = []
        self.envs = []
        self.mutate = None
        self.fail_stage = None

    def tearDown(self):
        self.temp.cleanup()

    def training_artifacts(self, root):
        root = Path(root)
        flags = dict(format="BRIGHT_JOINT_ANCHOR_STUDY_V1", complete=True,
                     execution_complete=True, training_complete=True,
                     full_validation_evidence=True, test_accessed=False)
        dump(root / "workflow_state.json", flags)
        dump(root / "train/summary.json", dict(
            **flags, source_mode="fresh", arm="d", branch_step=500,
            endpoint_step=1500, final_integrity_pending=False,smoke=False,
            new_updates={"control":1000,"anchor":1000}))
        for step, folder in ((500,"initial_val"),(1500,"val_step_001500")):
            dump(root / "train" / folder / "result.json", dict(
                step=step, image_count=450, whole_val450_processed=True,
                complete=True, execution_complete=True,
                full_validation_evidence=True, final_integrity_pending=False))
            checkpoint = root / "train/checkpoints" / ("checkpoint_%06d.pt" % step)
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            checkpoint.write_bytes(("synthetic checkpoint %s" % step).encode())
            dump(checkpoint.with_suffix(".pt.json"),
                 dict(step=step,sha256=chain.digest(checkpoint)))

    def audit_artifacts(self, folder, step, smoke):
        folder = Path(folder)
        result = dict(format="BRIGHT_ANCHOR_SLOT_MEMORY_AUDIT_V1",complete=True,execution_complete=True,training=False,
                      test_accessed=False,final_integrity_pending=False,
                      image_count=2 if smoke else 450, checkpoint_step=step,
                      smoke=smoke,full_validation_evidence=not smoke,
                      replay={"passed":not smoke})
        dump(folder / "result.json",result)
        dump(folder / "summary.json",dict(
            **result,result_sha256=chain.digest(folder / "result.json")))

    def execute(self, command, log, row, save, lock_fd, env):
        name = Path(log).stem
        self.calls.append((name,list(command)))
        self.envs.append((name,dict(env)))
        self.assertGreaterEqual(lock_fd,0)
        Path(log).write_text("synthetic child: %s\n" % name)
        row.update(returncode=1 if name == self.fail_stage else 0,
                   log_sha256=chain.digest(log))
        save()
        if name == self.fail_stage:
            raise RuntimeError("Stage failed with returncode 1: " + name)
        if name in ("parent","build"):
            target = Path(command[command.index("--output")+1])
            dump(target / "summary.json",dict(complete=True,smoke=False,test_accessed=False))
        elif name == "fresh_training":
            self.training_artifacts(command[command.index("--output")+1])
        elif name.startswith("audit_") and name != "audit_contracts":
            self.audit_artifacts(command[command.index("--output")+1],
                                 int(command[command.index("--step")+1]),
                                 "--smoke" in command)
        if self.mutate:
            self.mutate(name,command)

    def run_pipeline(self, release=None):
        fake_source = SimpleNamespace(prepare_plan=lambda *a:copy.deepcopy(self.plan))
        with patch.dict(sys.modules,{"source_plan":fake_source}), \
             patch.object(chain,"verify_release",side_effect=release or (lambda:"release-sha")), \
             redirect_stdout(io.StringIO()):
            chain.pipeline(self.args,executor=self.execute)
        return chain.read(self.output / "workflow_state.json")

    def state(self):
        return chain.read(self.output / "workflow_state.json")

    def names(self):
        return [name for name,_ in self.calls]

    def test_build_train_then_smoke_and_both_diagnostics_in_exact_order(self):
        state = self.run_pipeline()
        expected = ["install_code","source_plan","chain_contracts","audit_contracts",
                    "parent","build","fresh_training","audit_smoke","audit_000500","audit_001500"]
        self.assertEqual(list(state["stages"]),expected)
        self.assertEqual(self.names(),[s for s in expected if s != "source_plan"])
        for key in ("complete","execution_complete","training_complete",
                    "diagnostic_complete","full_validation_evidence"):
            self.assertIs(state[key],True)
        self.assertIs(state["test_accessed"],False)
        self.assertEqual(state["current_stage"],"complete")
        self.assertTrue(all(r["status"] == "complete" for r in state["stages"].values()))
        training = dict(self.calls)["fresh_training"]
        for option,value in (("--start","fresh"),("--arm","d"),("--weight","0.1"),("--lr-multiplier","0.2")):
            self.assertEqual(training[training.index(option)+1],value)
        self.assertEqual(dict(self.calls)["audit_smoke"].count("--smoke"),1)
        self.assertNotIn("--smoke",dict(self.calls)["audit_000500"])
        for name,env in self.envs:
            if name in ("install_code","chain_contracts","audit_contracts"):
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"],"")
        # Tensor checkpoint bytes are never loaded by the orchestration layer.
        self.assertTrue((self.output / "study/train/checkpoints/checkpoint_000500.pt").is_file())

    def test_existing_banks_are_reused_without_build_stage(self):
        self.plan["bank_actions"] = []
        self.run_pipeline()
        self.assertNotIn("parent",self.names())
        self.assertNotIn("build",self.names())
        self.assertIn("fresh_training",self.names())

    def test_audit_steps_are_deduplicated_and_sorted(self):
        self.args.audit_steps = [1500,500,1500]
        self.run_pipeline()
        self.assertEqual(self.names()[-3:],["audit_smoke","audit_000500","audit_001500"])
        command = dict(self.calls)["audit_smoke"]
        self.assertEqual(command[command.index("--step")+1],"500")

    def test_child_failure_stops_all_later_stages(self):
        self.fail_stage = "build"
        with self.assertRaisesRegex(RuntimeError,"returncode 1"):
            self.run_pipeline()
        self.assertEqual(self.names()[-1],"build")
        self.assertNotIn("fresh_training",self.names())
        state = self.state()
        self.assertFalse(state["complete"] or state["training_complete"])
        self.assertEqual(state["stages"]["build"]["status"],"failed")
        self.assertTrue((self.output / "traceback.txt").is_file())

    def test_missing_frozen_source_records_failure_before_contracts_or_training(self):
        self.plan.update(ready_for_build_or_reuse=False,missing=["a1.pth","q.pth"])
        with self.assertRaisesRegex(ValueError,"dependencies unavailable"):
            self.run_pipeline()
        self.assertEqual(self.names(),["install_code"])
        self.assertEqual(self.state()["stages"]["source_plan"]["status"],"failed")
        self.assertEqual(chain.read(self.output / "source_plan.json")["missing"],["a1.pth","q.pth"])
        self.assertFalse((self.output / "study").exists())

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir(parents=True)
        sentinel = self.output / "keep.txt";sentinel.write_text("unchanged")
        with self.assertRaises(FileExistsError):self.run_pipeline()
        self.assertEqual(sentinel.read_text(),"unchanged")
        self.assertFalse(self.calls)
        self.assertFalse((self.output / "workflow_state.json").exists())

    def test_output_cannot_overlap_code(self):
        self.args.output = self.args.original_tool
        with self.assertRaisesRegex(ValueError,"overlaps input/code"):
            self.run_pipeline()
        self.assertFalse(self.calls)
        self.assertFalse(Path(self.args.output).exists())

    def test_duplicate_lock_prevents_any_launch(self):
        self.output.parent.mkdir(parents=True)
        with (self.output.parent / "anchor_chain_active.lock").open("a") as held:
            fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError,"Another anchor chain"):
                self.run_pipeline()
        self.assertFalse(self.calls)
        self.assertFalse(self.output.exists())

    def test_zero_exit_incomplete_training_summary_prevents_diagnostics(self):
        def mutate(name,command):
            if name == "fresh_training":
                path = self.output / "study/train/summary.json"
                data = chain.read(path);data["training_complete"] = False;dump(path,data)
        self.mutate = mutate
        with self.assertRaisesRegex(ValueError,"complete formal evidence"):
            self.run_pipeline()
        self.assertEqual(self.names()[-1],"fresh_training")
        state=self.state()
        self.assertFalse(state["training_complete"] or state["diagnostic_complete"])
        self.assertEqual(state["stages"]["fresh_training"]["returncode"],0)
        self.assertEqual(state["stages"]["fresh_training"]["status"],"failed")

    def test_partial_val_report_prevents_diagnostics(self):
        def mutate(name,command):
            if name == "fresh_training":
                path=self.output / "study/train/val_step_001500/result.json"
                data=chain.read(path);data["image_count"]=449;dump(path,data)
        self.mutate=mutate
        with self.assertRaisesRegex(ValueError,"complete Val450"):
            self.run_pipeline()
        self.assertEqual(self.names()[-1],"fresh_training")

    def test_checkpoint_hash_mismatch_prevents_diagnostics(self):
        def mutate(name,command):
            if name == "fresh_training":
                path=self.output / "study/train/checkpoints/checkpoint_000500.pt"
                path.write_bytes(b"changed checkpoint")
        self.mutate=mutate
        with self.assertRaisesRegex(ValueError,"Checkpoint missing/changed"):
            self.run_pipeline()
        self.assertNotIn("audit_smoke",self.names())

    def test_wrong_update_count_prevents_diagnostics(self):
        def mutate(name,command):
            if name == "fresh_training":
                path=self.output / "study/train/summary.json"
                data=chain.read(path);data["new_updates"]["anchor"]=999;dump(path,data)
        self.mutate=mutate
        with self.assertRaisesRegex(ValueError,"paired update counts"):
            self.run_pipeline()
        self.assertNotIn("audit_smoke",self.names())

    def test_incomplete_smoke_audit_prevents_full_audits(self):
        def mutate(name,command):
            if name == "audit_smoke":
                path=self.output / "audit_smoke/summary.json"
                data=chain.read(path);data["final_integrity_pending"]=True;dump(path,data)
        self.mutate=mutate
        with self.assertRaisesRegex(ValueError,"matching evidence"):
            self.run_pipeline()
        self.assertEqual(self.names()[-1],"audit_smoke")
        state=self.state()
        self.assertTrue(state["training_complete"])
        self.assertFalse(state["diagnostic_complete"] or state["complete"])

    def test_audit_checksum_mismatch_stops_later_checkpoint(self):
        def mutate(name,command):
            if name == "audit_000500":
                path=self.output / name / "result.json"
                data=chain.read(path);data["extra"]=1;dump(path,data)
        self.mutate=mutate
        with self.assertRaisesRegex(ValueError,"checksum mismatch"):
            self.run_pipeline()
        self.assertEqual(self.names()[-1],"audit_000500")
        self.assertNotIn("audit_001500",self.names())

    def test_failed_replay_is_not_promoted_even_with_valid_checksum(self):
        def mutate(name,command):
            if name == "audit_000500":
                path=self.output / name / "result.json"
                data=chain.read(path);data["replay"]["passed"]=False;dump(path,data)
                summary=self.output / name / "summary.json"
                data=chain.read(summary);data["result_sha256"]=chain.digest(path);dump(summary,data)
        self.mutate=mutate
        with self.assertRaisesRegex(ValueError,"did not replay"):
            self.run_pipeline()
        self.assertFalse(self.state()["full_validation_evidence"])

    def test_changed_chain_release_cannot_claim_complete(self):
        releases=iter(["before","after"])
        with self.assertRaisesRegex(ValueError,"code changed"):
            self.run_pipeline(release=lambda:next(releases))
        self.assertFalse(self.state()["complete"])
        self.assertTrue(self.state()["training_complete"])


class ChildProcessContracts(unittest.TestCase):
    def test_nonzero_exit_is_logged_and_raised(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);row={};saves=[]
            with (root/"lock").open("a") as lock,redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,"returncode 7"):
                    chain.run_command([sys.executable,"-c","print('synthetic failure');raise SystemExit(7)"],
                                      root/"child.log",row,lambda:saves.append(dict(row)),
                                      lock.fileno(),dict(__import__('os').environ))
            self.assertEqual(row["returncode"],7)
            self.assertEqual(row["log_sha256"],chain.digest(root/"child.log"))
            self.assertIn("synthetic failure",(root/"child.log").read_text())
            self.assertTrue(saves and "pid" in saves[0])

    def test_existing_log_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);log=root/"child.log";log.write_text("keep")
            with (root/"lock").open("a") as lock:
                with self.assertRaises(FileExistsError):
                    chain.run_command([sys.executable,"-c","raise SystemExit(99)"],log,{},lambda:None,
                                      lock.fileno(),{})
            self.assertEqual(log.read_text(),"keep")


if __name__ == "__main__":
    unittest.main()
