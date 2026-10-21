"""CPU/synthetic contracts, not a reproduction of the remote CUDA failure."""
import contextlib
import copy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/"bright_route_failure_resume_v1.pyz"
sys.path.insert(0,str(ARCHIVE))
sys.path.extend([os.environ["BRIGHT_RESUME_TEST_SOURCE"],os.environ["BRIGHT_ABLATION_TEST_SOURCE"]])
import torch
import passive_routes
from passive_routes import PassiveRoutes
import failure_bindings
import training_entry
import resume_train

package=types.ModuleType("_passive_original")
package.__path__=[str(Path(os.environ["BRIGHT_ABLATION_REFERENCE"])/"deps/original")]
sys.modules[package.__name__]=package
model_tests=importlib.import_module(package.__name__+".test_model")
slots=importlib.import_module(package.__name__+".slots")


class PassiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.q=torch.tensor([[[.2,.3],[.8,.7]]])
        self.ids=torch.tensor([[0,1],[2,3]])
        self.position=dict(mode="evaluation",step=1500,index=167,image="target",active_arm="r",stage="forward_begin")
        self.trace=PassiveRoutes(torch,model_tests.JointRegionMemoryClassifier,self.root,self.position)

    def test_accepted_call_keeps_identity_and_invokes_original_allclose_once(self):
        original=slots.validate_routes
        wrapped=self.trace.wrap(original,"unit")
        qbefore=self.q.clone()
        with mock.patch.object(torch,"allclose",wraps=torch.allclose) as allclose, \
             mock.patch.object(self.trace,"capture",side_effect=AssertionError("copy on success")):
            result=wrapped(self.q,self.ids,4,1,2)
        self.assertEqual(allclose.call_count,1)
        self.assertTrue(torch.equal(self.q,qbefore))
        self.assertEqual(result.shape,(1,2,2))
        self.assertEqual(result.data_ptr(),self.ids.data_ptr())

    def test_negative_short_circuits_original_sum_check(self):
        bad=self.q.clone()
        bad[0,0,0]=-.1
        wrapped=self.trace.wrap(slots.validate_routes,"unit")
        with mock.patch.object(torch,"allclose",wraps=torch.allclose) as allclose, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,"nonnegative"):
                wrapped(bad,self.ids,4,1,2)
        report=json.loads((self.root/"route_failure.json").read_text())
        self.assertEqual(allclose.call_count,0)
        self.assertTrue(report["original_bool"][-1]["result"])
        self.assertEqual(report["original_allclose"],[])
        self.assertEqual(report["weights"]["negative_count"],1)

    def test_failed_sum_saves_actual_original_reduction_operands(self):
        wrapped=self.trace.wrap(slots.validate_routes,"unit")
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,"sum to one"):
                wrapped(self.q*2,self.ids,4,1,2)
        saved=torch.load(self.root/"route_failure_cpu.pt",weights_only=False)
        self.assertFalse(saved["report"]["original_allclose"][0]["result"])
        self.assertTrue(torch.equal(saved["tensors"]["allclose_0_0"],torch.tensor([[2.,2.]])))
        self.assertFalse(saved["report"]["original_allclose"][0]["cpu_check_of_original_operands"])
        self.assertEqual(saved["report"]["position"]["active_arm"],"r")

    def test_false_original_predicate_is_not_overridden_by_valid_cpu_recheck(self):
        original_allclose=torch.allclose
        calls=0
        def flaky(*a,**kw):
            nonlocal calls
            calls+=1
            return False if calls==1 else original_allclose(*a,**kw)
        wrapped=self.trace.wrap(slots.validate_routes,"unit")
        with mock.patch.object(torch,"allclose",side_effect=flaky),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,"sum to one"):
                wrapped(self.q,self.ids,4,1,2)
        record=json.loads((self.root/"route_failure.json").read_text())["original_allclose"][0]
        self.assertFalse(record["result"])
        self.assertTrue(record["cpu_check_of_original_operands"])

    def test_runtime_error_is_reraised_without_copying_tensors(self):
        wrapped=self.trace.wrap(slots.validate_routes,"unit")
        failure=RuntimeError("synthetic CUDA exception")
        with mock.patch.object(torch,"allclose",side_effect=failure), \
             mock.patch.object(torch.Tensor,"to",side_effect=AssertionError("post-error tensor read")), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError) as caught:
                wrapped(self.q,self.ids,4,1,2)
        self.assertIs(caught.exception,failure)
        saved=torch.load(self.root/"route_failure_cpu.pt",weights_only=False)
        self.assertEqual(saved["tensors"],{})
        self.assertTrue(saved["report"]["tensor_reads_skipped"])

    def test_evidence_io_error_does_not_replace_original_exception(self):
        wrapped=self.trace.wrap(slots.validate_routes,"unit")
        with mock.patch.object(self.trace,"capture",side_effect=OSError("disk full")), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,"nonnegative"):
                wrapped(-self.q,self.ids,4,1,2)
        self.assertIn("disk full",self.trace.evidence_error)

    def test_original_classifier_logits_gradients_and_adamw_update_match(self):
        inputs=model_tests.fixture()
        original=model_tests.JointRegionMemoryClassifier(hidden=16,heads=4,chunk_size=2)
        with torch.no_grad(): original.joint_head[-1].weight.normal_(0,.1)
        observed=copy.deepcopy(original)
        opt1=torch.optim.AdamW(original.parameters(),lr=1e-4)
        opt2=torch.optim.AdamW(observed.parameters(),lr=1e-4)
        first=original(**inputs)["logits"]
        first.square().mean().backward()
        grads={n:p.grad.clone() for n,p in original.named_parameters() if p.grad is not None}
        opt1.step()
        prior=slots.validate_routes
        with self.trace:
            second=observed(**inputs)["logits"]
            second.square().mean().backward()
            for n,p in observed.named_parameters():
                if n in grads: self.assertTrue(torch.equal(grads[n],p.grad),n)
            opt2.step()
        self.assertTrue(torch.equal(first,second))
        self.assertTrue(all(torch.equal(v,observed.state_dict()[n]) for n,v in original.state_dict().items()))
        self.assertIs(slots.validate_routes,prior)
        self.assertGreaterEqual(self.trace.calls,5)
        self.assertFalse((self.root/"route_failure.json").exists())

    def test_original_helpers_preserved_byte_for_byte(self):
        with zipfile.ZipFile(ARCHIVE) as z:
            ledger=json.loads(z.read("INHERITED_TRAINER.json"))
            for name,digest in ledger["files"].items():
                self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),digest,name)


class BindingTests(unittest.TestCase):
    def test_actual_route_rejection_has_active_r_and_restores_every_binding(self):
        inputs=model_tests.fixture()
        model=model_tests.JointRegionMemoryClassifier(hidden=16,heads=4,chunk_size=2)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            class Runtime:
                def prepare(self,name): return types.SimpleNamespace(image=name)
            class Factory:
                def prepare(self,example,arms): return {a:None for a in arms}
                def forward(self,head,example,arm,support):
                    data=dict(inputs)
                    if arm=="r":
                        data["route_q"]=-data["route_q"]
                    return head(**data)
            factory=Factory()
            def evaluate(runtime,heads,names,folder,**kwargs):
                example=runtime.prepare(names[0])
                supports=factory.prepare(example,heads)
                for arm,head in heads.items():
                    result=factory.forward(head,example,arm,supports[arm])
                    evaluator._validate_probability(result["probabilities"],None,arm)
            evaluator=types.SimpleNamespace(evaluate=evaluate,_validate_probability=lambda *a:None)
            bindings=types.SimpleNamespace(torch=torch,Runtime=Runtime,
                original_train=types.SimpleNamespace(JointRegionMemoryClassifier=type(model)))
            modules={"evaluation.py":evaluator,"supports.py":types.SimpleNamespace(SupportFactory=Factory)}
            bridge=types.SimpleNamespace(load_runtime=lambda *a:bindings,named=lambda p,n:modules[Path(p).name])
            context=dict(bridge=bridge,addon=root)
            before_forward=Factory.forward
            before_runtime=bindings.Runtime
            before_load=bridge.load_runtime
            before_validator=slots.validate_routes
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,"nonnegative"):
                    with failure_bindings.observe_bindings(context,root):
                        loaded=bridge.load_runtime({})
                        bridge.named(root/"supports.py","s")
                        ev=bridge.named(root/"evaluation.py","e")
                        runtime=loaded.Runtime()
                        ev.evaluate(runtime,{"d":model,"r":model},["target"],root/"val_step_001500",
                                    step=1500,split="val")
            record=json.loads((root/"route_failure.json").read_text())
            self.assertEqual(record["position"]["active_arm"],"r")
            self.assertEqual(record["position"]["image"],"target")
            self.assertEqual(record["position"]["step"],1500)
            self.assertEqual(record["position"]["index"],0)
            self.assertEqual(record["position"]["stage"],"forward_begin")
            self.assertIs(Factory.forward,before_forward)
            self.assertIs(bindings.Runtime,before_runtime)
            self.assertIs(bridge.load_runtime,before_load)
            self.assertIs(slots.validate_routes,before_validator)
            self.assertNotIn("prepare",vars(runtime))


from test_resume_integration import ResumeIntegrationTests

class InstrumentedResumeIntegration(ResumeIntegrationTests):
    """Original update/restore/checkpoint/evaluation state machine with new binding hooks.

    The tiny integration classifier has no routes; route instrumentation itself
    is tested above using the real original classifier.
    """
    def context(self,**kwargs):
        context=super().context(**kwargs)
        bridge=context["bridge"]
        named,load=bridge.named,bridge.load_runtime
        def compatible(path,label):
            value=named(path,label)
            if Path(path).name=="evaluation.py":
                value._validate_probability=lambda *a:None
            return value
        def runtime(source):
            value=load(source)
            value.torch=torch
            return value
        bridge.named,bridge.load_runtime=compatible,runtime
        context["standalone_evaluation"]="synthetic certificate"
        context["extension_sha256"]=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
        return context

    def run_branch(self,*a,**kw):
        with mock.patch.object(training_entry,"observe_bindings",failure_bindings.observe_bindings), \
             mock.patch.object(sys,"argv",[str(ARCHIVE)]), \
             mock.patch.object(resume_train,"run_training",training_entry.observed_run), \
             mock.patch.object(resume_train,"imported_file",training_entry.described_binding), \
             mock.patch.object(failure_bindings,"PassiveRoutes",side_effect=lambda *a,**k:contextlib.nullcontext()):
            return super().run_branch(*a,**kw)


if __name__=="__main__":
    unittest.main()
