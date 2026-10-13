"""CPU-only tests of bounded allocation/probe behavior and checkpoint checks."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

import worker

SOURCE_CANDIDATES = [Path(os.environ["JOINT_ORIGINAL_TOOL"]) if os.environ.get("JOINT_ORIGINAL_TOOL") else None,
    Path(__file__).resolve().parent.parent/"joint_recovery_source"/"region_memory_joint_train_v1",
    Path("/root/shared-nvme/BRIGHT-master/New/joint_train_ZAtNI2Wi/region_memory_joint_train_v1")]
def _readable(path):
    try:
        return path is not None and (path/"test_train.py").is_file()
    except OSError:
        return False
SOURCE = next((p for p in SOURCE_CANDIDATES if _readable(p)), None)
if SOURCE is None:
    raise RuntimeError("Set JOINT_ORIGINAL_TOOL to the unchanged original joint package")
sys.path.insert(0,str(SOURCE))
import train
from test_train import FakeRuntime


class ProbeStopped(Exception):
    def __init__(self, report):
        self.report = report


class ProbeRuntime(FakeRuntime):
    instances=[]
    reference_error=False
    kernel_error=False
    no_stop=False
    seen_loader_names=[]
    backend="live_contiguous_probe"

    def __init__(self,args,output):
        super().__init__(args,output)
        self.packets=[]
        self.guard=SimpleNamespace(report=lambda:{"captured_packets":copy.deepcopy(self.packets)})
        self.batch_count=0

    def batches(self,names,train,seed):
        self.seen_loader_names=list(names)
        for name in names:
            self.batch_count+=1
            yield name

    def prepare(self,image):
        self.prepare_count+=1
        if self.kernel_error:
            raise RuntimeError("Synthetic CUDA kernel failure")
        path=Path(self.output)/"pool_debug"/"actual_512.pth"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"synthetic actual-output packet")
        row={"path":str(path),"sha256":train.digest(path),"image":image,"call":2,
             "content_channels":512,"backend":self.backend,
             "strict_cpu_gate_passed":not self.reference_error,
             "strict_original_cpu32_gate":{"passed":not self.reference_error}}
        self.packets.append(row)
        if self.reference_error:
            train.write_json(path.parent/"failure.json",{"phase":"cpu_reference_gate"})
            raise ValueError("Original-semantics CPU pooling reference gate failed")
        if not self.no_stop:
            raise ProbeStopped(row)
        return None


class CPUFullRuntime(ProbeRuntime):
    backend="cpu_fp32_reference"
    def prepare(self,image):
        example=FakeRuntime.prepare(self,image)
        path=Path(self.output)/"pool_debug"/"actual_512.pth"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"synthetic CPU actual-output packet")
        self.packets.append({"path":str(path),"sha256":train.digest(path),"image":image,"call":2,
             "content_channels":512,"backend":self.backend,"strict_cpu_gate_passed":True,
             "strict_original_cpu32_gate":{"passed":True}})
        return example


class WorkerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.base=Path(self.temp.name)
        ProbeRuntime.instances=[]
    def tearDown(self):
        self.temp.cleanup()
    def args(self,name="probe"):
        return SimpleNamespace(repo="synthetic",v3_root="synthetic",protocol=None,
            output=str(self.base/name),seed=3407,workers=0,device="cpu",topk=64,
            chunk_size=2,mode="smoke",steps=2,eval_every=20,lr=1e-4,weight_decay=5e-3)
    def saved(self):
        # Produce real three-reader optimizer states through the original CPU
        # training contracts; test artifacts are explicitly synthetic smoke.
        with redirect_stdout(io.StringIO()):
            summary=train.run_training(self.args("prefix"),runtime_factory=FakeRuntime)
        return torch.load(summary["last_checkpoint"],map_location="cpu",weights_only=False)
    def invoke(self,saved,mode="live-probe",runtime=ProbeRuntime,evidence=None):
        evidence=evidence or {"probe_image":"val_a","probe_call":2}
        with redirect_stdout(io.StringIO()):
            return worker.probe(train,self.args(),runtime,saved,{},evidence,ProbeStopped,mode=mode)
    def test_probe_restores_all_allocations_full_loader_one_batch_and_zero_updates(self):
        saved=self.saved()
        with mock.patch.object(torch.optim.AdamW,"step",side_effect=AssertionError("Probe called optimizer.step")), \
             mock.patch.object(torch.Tensor,"backward",side_effect=AssertionError("Probe called backward")), \
             mock.patch.object(train,"_evaluate",side_effect=AssertionError("Probe invoked unbounded validation")):
            report=self.invoke(saved)
        runtime=ProbeRuntime.instances[-1]
        self.assertEqual(runtime.seen_loader_names,runtime.val_names)
        self.assertEqual(runtime.batch_count,1)
        self.assertEqual(runtime.prepare_count,1)
        self.assertTrue(report["capture_ready"])
        self.assertTrue(report["gate_passed"])
        self.assertTrue(report["heads_and_optimizers_restored"])
        self.assertFalse(report["checkpoint_rng_restored"])
        self.assertEqual(report["new_updates_per_arm"],{a:0 for a in train.ARMS})
        self.assertEqual(report["optimizer_steps_called"],0)
        self.assertFalse(report["full_validation_evidence"])
        self.assertEqual(train.digest(report["capture"]),report["capture_sha256"])
        self.assertTrue(report["guard_report"]["restored"])
    def test_live_reference_mismatch_is_completed_diagnostic_not_training(self):
        saved=self.saved()
        class Mismatch(ProbeRuntime):
            reference_error=True
        report=self.invoke(saved,runtime=Mismatch)
        self.assertTrue(report["execution_complete"])
        self.assertTrue(report["capture_ready"])
        self.assertFalse(report["gate_passed"])
        self.assertEqual(report["failure"]["phase"],"cpu_reference_gate")
        self.assertEqual(report["new_updates_per_arm"],{a:0 for a in train.ARMS})
    def test_cpu_probe_declares_backend_and_strict_gate(self):
        saved=self.saved()
        forward_arms=[]
        new_heads=train._new_heads
        def instrument_heads(*args,**kwargs):
            heads,hashes=new_heads(*args,**kwargs)
            for arm,head in heads.items():
                head.register_forward_hook(lambda _head,_inputs,_output,arm=arm:forward_arms.append(arm))
            return heads,hashes
        with mock.patch.object(train,"_new_heads",side_effect=instrument_heads), \
             mock.patch.object(torch.optim.AdamW,"step",side_effect=AssertionError("CPU probe updated optimizer")):
            report=self.invoke(saved,mode="cpu-probe",runtime=CPUFullRuntime)
        self.assertEqual(report["backend"],"cpu_fp32_reference")
        self.assertTrue(report["strict_cpu_gate_passed"])
        self.assertTrue(report["independent_fp64_analysis_pending"])
        self.assertTrue(report["full_single_image_forward_complete"])
        self.assertEqual(forward_arms,list(train.ARMS))
        self.assertEqual(set(report["per_head_forward_checks"]),set(train.ARMS))
        self.assertEqual(ProbeRuntime.instances[-1].batch_count,1)
    def test_kernel_failure_is_not_complete_and_never_retries(self):
        saved=self.saved()
        class KernelFailure(ProbeRuntime):
            kernel_error=True
        with self.assertRaisesRegex(RuntimeError,"kernel failure"):
            self.invoke(saved,runtime=KernelFailure)
        self.assertEqual(ProbeRuntime.instances[-1].prepare_count,1)
        report=json.loads((self.base/"probe"/"summary.json").read_text())
        self.assertFalse(report["execution_complete"])
        self.assertFalse(report["capture_ready"])
        self.assertEqual(report["new_updates_per_arm"],{a:0 for a in train.ARMS})
    def test_wrong_context_and_missing_stop_rejected(self):
        saved=self.saved()
        with self.assertRaisesRegex(ValueError,"first in Val order"):
            self.invoke(saved,evidence={"probe_image":"val_b","probe_call":2})
        self.assertEqual(ProbeRuntime.instances[-1].prepare_count,0)
        with self.assertRaises(FileExistsError):
            self.invoke(saved)
    def test_mismatched_call_rejected_even_with_correct_packet(self):
        saved=self.saved()
        with self.assertRaisesRegex(ValueError,"call differs"):
            self.invoke(saved,evidence={"probe_image":"val_a","probe_call":3})
        report=json.loads((self.base/"probe"/"summary.json").read_text())
        self.assertFalse(report["capture_ready"])
    def test_cpu_probe_cannot_claim_complete_after_stats_only_stop(self):
        saved=self.saved()
        class PrematureCPU(ProbeRuntime):
            backend="cpu_fp32_reference"
        with self.assertRaisesRegex(RuntimeError,"before complete single-image"):
            self.invoke(saved,mode="cpu-probe",runtime=PrematureCPU)
        summary=json.loads((self.base/"probe"/"summary.json").read_text())
        self.assertFalse(summary["full_single_image_forward_complete"])
        self.assertFalse(summary["execution_complete"])
    def test_cpu_probe_nonfinite_reader_loss_rejected_without_updates(self):
        saved=self.saved()
        class BadLoss(CPUFullRuntime):
            def loss(self,probability,target):
                return probability.sum()*float("nan")
        with mock.patch.object(torch.optim.AdamW,"step",side_effect=AssertionError("Probe optimizer update")):
            with self.assertRaisesRegex(ValueError,"Nonfinite single-image CPU probe"):
                self.invoke(saved,mode="cpu-probe",runtime=BadLoss)
        summary=json.loads((self.base/"probe"/"summary.json").read_text())
        self.assertFalse(summary["full_single_image_forward_complete"])
        self.assertEqual(summary["new_updates_per_arm"],{a:0 for a in train.ARMS})
    def test_live_probe_must_stop_instead_of_silently_running_reader(self):
        saved=self.saved()
        class NoStop(ProbeRuntime):
            no_stop=True
        with self.assertRaisesRegex(RuntimeError,"did not stop"):
            self.invoke(saved,runtime=NoStop)
        self.assertEqual(ProbeRuntime.instances[-1].prepare_count,1)

    def test_direct_training_requires_parent_gates_before_checkpoint_load(self):
        fake_recovery=SimpleNamespace(check_preserved=lambda evidence:None,
            digest=lambda path:"unused",read=lambda path:{},source_hash=lambda path:"unused",
            verify_release=lambda path:None,inspect_failed=lambda root,tool:None,
            check_train_gates=mock.Mock(side_effect=ValueError("CPU-FP64 preflight missing")))
        args=SimpleNamespace(mode="train",output=str(self.base/"direct_train"),evidence="evidence.json")
        with mock.patch.dict(sys.modules,{"recovery":fake_recovery}), \
             mock.patch.object(torch,"load",side_effect=AssertionError("Loaded checkpoint before train gates")):
            with self.assertRaisesRegex(ValueError,"preflight missing"):
                worker.run(args)
        fake_recovery.check_train_gates.assert_called_once_with("evidence.json")
    def formal_saved_and_evidence(self):
        model=torch.nn.Linear(2,2)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        optimizer.zero_grad()
        model(torch.ones(1,2)).sum().backward()
        optimizer.step()
        source=self.base/"data"
        source.write_text("frozen")
        saved={"format":"SRAC_REGION_MEMORY_JOINT_TRAIN_V1","mode":"train","smoke":False,
               "training_complete":False,"step":1500,"config":{"steps":5000},
               "source_hash":"original","protocol_info":{"bank":"fixed"},
               "heads":{a:copy.deepcopy(model.state_dict()) for a in train.ARMS},
               "optimizers":{a:copy.deepcopy(optimizer.state_dict()) for a in train.ARMS},
               "updates_per_arm":{a:1500 for a in train.ARMS},
               "initial_head_hashes":{a:"initial" for a in train.ARMS},
               "rng":{"python":random.getstate(),"numpy":np.random.get_state(),
                      "torch_cpu":torch.get_rng_state(),"torch_cuda":[]},
               "input_file_sha256":{str(source):train.digest(source)}}
        evidence={"arguments":saved["config"],"original_source_hash":saved["source_hash"],
                  "protocol_info":saved["protocol_info"],"checkpoint_step":1500,
                  "initial_head_hashes":saved["initial_head_hashes"],
                  "sidecar":{"head_hashes":{a:worker.state_hash(saved["heads"][a]) for a in train.ARMS}}}
        return saved,evidence
    def test_loaded_checkpoint_checks_adamw_all_three_heads_rng_and_hashes(self):
        saved,evidence=self.formal_saved_and_evidence()
        worker.validate_loaded(saved,evidence)
        bad=copy.deepcopy(saved)
        bad["heads"]["rf"]["weight"][0,0]+=1
        with self.assertRaisesRegex(ValueError,"head checksum"):
            worker.validate_loaded(bad,evidence)
        bad=copy.deepcopy(saved)
        bad["optimizers"]["rt"]["state"][0]["exp_avg"].fill_(float("nan"))
        with self.assertRaisesRegex(ValueError,"Nonfinite optimizer"):
            worker.validate_loaded(bad,evidence)
        bad=copy.deepcopy(saved)
        bad["rng"]["torch_cpu"]=torch.zeros(1,dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError,"RNG"):
            worker.validate_loaded(bad,evidence)


if __name__=="__main__":
    unittest.main()
