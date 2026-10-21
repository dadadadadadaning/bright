"""CPU/synthetic tests only; never evidence of real CUDA execution or accuracy."""
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "bright_training_next_v1.pyz"
SOURCE = Path(os.environ["BRIGHT_RESUME_TEST_SOURCE"])
ADDON = Path(os.environ["BRIGHT_ABLATION_TEST_SOURCE"])
REFERENCE = Path(os.environ["BRIGHT_ABLATION_REFERENCE"])
sys.path.insert(0, str(ARCHIVE))
sys.path.extend([str(SOURCE), str(ADDON)])

import torch
import evidence
import resume_train
import training_entry
import training_common
import no_read_baseline as baseline
from test_resume_integration import ResumeIntegrationTests, CPUState, pixel_nll, auxiliary_loss
from test_evidence import make_report, write_report
import ablation_runner

package = types.ModuleType("_next_original")
package.__path__ = [str(REFERENCE / "deps/original")]
sys.modules[package.__name__] = package
model_tests = importlib.import_module("_next_original.test_model")


class BundleTests(unittest.TestCase):
    def test_fixed_helpers_are_byte_identical(self):
        with zipfile.ZipFile(ARCHIVE) as z:
            ledger = json.loads(z.read("BASE_HELPERS.json"))
            self.assertEqual(ledger["source_sha256"],
                "0f7fd0a85ca2db5727509875c84c1beba1011d2bd4300a8b6b5ea86b59e75f07")
            for name, digest in ledger["files"].items():
                self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(), digest, name)
        self.assertIn(str(ARCHIVE), evidence.__file__)

    def test_wrong_checkpoint_is_rejected_before_training(self):
        context = dict(parent_root=Path(training_common.DEFAULT_ROOT), seed=3407,
                       parent={"step":1500,"sha256":"0"*64}, plan={"endpoint_step":2500})
        with self.assertRaisesRegex(ValueError,"recorded seed3407"):
            training_common.certificate(context)


class PairEvaluationTests(unittest.TestCase):
    def test_real_evaluator_accepts_pair_and_keeps_cpu_results_nonformal(self):
        import test_evaluation as fixtures
        fixtures.EvaluationContracts.setUpClass()
        fixture = fixtures.EvaluationContracts()
        fixture.setUp()
        try:
            fixture.heads = {"d":fixture.heads["d"],"h":fixture.heads["e"]}
            runtime = fixture.runtime()
            runtime.val_names = ["val_%s" % i for i in range(450)]
            rng = fixture.legacy.state.state_fingerprint(fixture.legacy.state.capture_rng())
            with baseline.pair_schema(fixtures.evaluation,evidence):
                report = fixture.run_evaluation(runtime)
                self.assertIn("d_minus_h",report["comparisons_pp"])
                self.assertEqual(report["arms"],["d","h"])
                self.assertEqual(report["image_count"],450)
                self.assertTrue(report["smoke"])
                self.assertFalse(report["full_validation_evidence"])
                with self.assertRaises(ValueError):
                    evidence.validate_report(report,step=500,split="val",names=runtime.val_names,
                                             arms=["d","h"],primary=False)
            self.assertEqual(rng,fixture.legacy.state.state_fingerprint(fixture.legacy.state.capture_rng()))
            self.assertNotIn("h",fixtures.evaluation.ARMS)
            self.assertNotIn("h",evidence.VALID_ARMS)
        finally:
            fixture.tearDown()


class CertificateTests(unittest.TestCase):
    def test_completed_report_binds_exact_checkpoint_and_rejects_changed_journal(self):
        from evaluate_checkpoint import finalize_fresh_report, REQUIRED_CHECKS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "parent"
            output = Path(directory) / "evaluation"
            addon = Path(directory) / "addon"
            root.mkdir(); output.mkdir(); addon.mkdir()
            (root/"trainer.pyz").write_bytes(b"synthetic archive identity")
            (addon/"evaluation.py").write_text("# synthetic source identity")
            names = ["val_%s" % n for n in range(450)]
            arms = list("efbdrl")
            report = make_report(1500,"val",names,arms)
            path, _ = write_report(output/"val_step_001500",report,names)
            checks = {k:True for k in REQUIRED_CHECKS}
            lineage = {"synthetic":True}
            trainer_sha = training_common.sha256(root/"trainer.pyz")
            provenance = dict(source_root=str(root),checkpoint_step=1500,checkpoint_sha256="a"*64,
                trainer_sha256=trainer_sha,lineage=lineage,
                evaluation_source_sha256=training_common.sha256(addon/"evaluation.py"))
            finalize_fresh_report(path,report,output=output,names=names,evidence=evidence,
                                  checks=checks,provenance=provenance)
            (output/"summary.json").write_text(json.dumps(dict(phase="standalone_evaluation_complete",
                complete=True,image_count=450,integrity_checks=checks)))
            context = dict(parent_root=root,parent={"step":1500,"sha256":"a"*64},
                seed=3407,plan={"endpoint_step":2500,"arms":arms},lineage=lineage,
                addon=addon,manifest={"val_names":names},evidence_files={})
            with mock.patch.object(training_common,"DEFAULT_ROOT",str(root)), \
                 mock.patch.object(training_common,"CHECKPOINT_SHA256","a"*64), \
                 mock.patch.object(training_common,"TRAINER_SHA256",trainer_sha):
                training_common.certificate(context,output)
                self.assertIn(str(path),context["evidence_files"])
                with path.with_name("images.jsonl").open("a") as stream:
                    stream.write('{}\n')
                with self.assertRaises(ValueError):
                    training_common.certificate(context,output)


class RealModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.inputs = model_tests.fixture()
        self.d = model_tests.JointRegionMemoryClassifier(hidden=16, heads=4, chunk_size=2)
        # Crucial: zero-initialized classifier would hide ANY support dependence.
        with torch.no_grad():
            self.d.joint_head[-1].weight.normal_(0, .1)
        self.h = copy.deepcopy(self.d)
        baseline.disable_support_read(self.h)

    def test_h_is_independent_of_support_but_retains_current_image_features(self):
        before = self.h(**self.inputs)
        altered = {k: v.clone() for k,v in self.inputs.items()}
        altered["support_f"] *= -3
        altered["support_t"] *= 4
        altered["support_labels"] = altered["support_labels"].roll(1, -1)
        altered["support_ids"] = altered["support_ids"].flip(1)
        after = self.h(**altered)
        self.assertTrue(torch.equal(before["logits"], after["logits"]))
        self.assertEqual(int(before["memory_context"].count_nonzero()), 0)
        self.assertFalse(torch.equal(self.d(**self.inputs)["logits"], self.d(**altered)["logits"]))
        altered["pixel_features"] += .5
        self.assertFalse(torch.equal(before["logits"], self.h(**altered)["logits"]))

    def test_gradients_and_real_adamw_steps(self):
        heads = {"d": self.d, "h": self.h}
        common = {k:v for k,v in self.inputs.items() if not k.startswith("support_")}
        support = {k:v for k,v in self.inputs.items() if k.startswith("support_")}
        original_factory = types.SimpleNamespace(prepare=lambda ex, arms: {"d":support},
            forward=lambda model, ex, arm, sup: model(**ex.common, **sup), provenance=lambda: {})
        factory = baseline.PairFactory(original_factory)
        example = types.SimpleNamespace(common=common, target=torch.randint(4, (1,4,8)))
        runtime = types.SimpleNamespace(loss=lambda p,t:pixel_nll(p,t)+auxiliary_loss(p,t),
                                        deps=types.SimpleNamespace(region_nll=pixel_nll))
        opts = {a:torch.optim.AdamW(h.parameters(), lr=1e-4, weight_decay=.005) for a,h in heads.items()}
        initial_h = self.h.joint_head[-1].weight.detach().clone()
        for step in range(1, 4):
            rows = ablation_runner.update_arms(heads, opts, example, runtime, factory, CPUState(),
                                               step=step, multiplier=.2)
            self.assertEqual(rows["h"]["reader_grad_abs_sum"], 0)
            self.assertGreater(rows["d"]["reader_grad_abs_sum"], 0)
        self.assertFalse(torch.equal(initial_h, self.h.joint_head[-1].weight))
        for name,p in self.h.named_parameters():
            if name.startswith(baseline.READER_PREFIXES):
                self.assertIsNone(p.grad)
            elif name.startswith(("region_encoder", "slot_feature_encoder", "pixel_projection")):
                self.assertIsNotNone(p.grad)


class ObservedResumeIntegration(ResumeIntegrationTests):
    """Run the original success, interrupted-update and endpoint-recovery tests with wrapper."""
    def context(self, **kwargs):
        context = super().context(**kwargs)
        named = context["bridge"].named
        def compatible(path, label):
            result = named(path, label)
            if Path(path).name == "evaluation.py":
                result._validate_probability = lambda *args: None
            return result
        context["bridge"].named = compatible
        context["standalone_evaluation"] = "synthetic-certificate"
        return context

    def run_branch(self, *args, **kwargs):
        with mock.patch.object(resume_train, "run_training", training_entry.observed_run), \
             mock.patch.object(resume_train, "imported_file", training_entry.described_binding):
            return super().run_branch(*args, **kwargs)


class PairWorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = CPUState()
        self.train = ["train_%s" % n for n in range(4)]
        self.val = ["val_%s" % n for n in range(450)]
        self.fixed = self.train[:2]
        self.closed = False
        self.inputs = model_tests.fixture()
        self.protocol = {"execution_settings":{"cuda_launch_blocking":None}, "fixture":"synthetic CPU"}
        self.input_file = self.root / "input.txt"
        self.input_file.write_text("Synthetic fixture only")
        self.policy = self.root / "policy.json"
        self.policy.write_text(json.dumps({"numeric_policy":{"num_threads":1,
            "num_interop_threads":torch.get_num_interop_threads()}}))
        self.config = dict(chunk_size=2, lr=1e-4, weight_decay=.005, protocol=str(self.policy))

    def context(self, fail_eval=False):
        owner = self
        class Runtime:
            def __init__(self, args, output):
                self.device = torch.device("cpu")
                self.protocol_info = owner.protocol
                self.input_ledger = {owner.input_file:training_common.sha256(owner.input_file)}
                self.deps = types.SimpleNamespace(region_nll=pixel_nll)
                self.guard_report = dict(restored=True, failures=[], restore_conflicts=[])
            def __enter__(self): return self
            def __exit__(self, *args): owner.closed=True
            def batches(self, names, train, seed):
                yield from names
            def prepare(self, name):
                return types.SimpleNamespace(image=name, event="synthetic",
                    common={k:v for k,v in owner.inputs.items() if not k.startswith("support_")},
                    target=torch.arange(32).reshape(1,4,8) % 4)
            def loss(self, p,t): return pixel_nll(p,t)+auxiliary_loss(p,t)
        class Factory:
            def __init__(self, runtime, seed): pass
            def prepare(self, example, arms):
                return {"d":{k:v for k,v in owner.inputs.items() if k.startswith("support_")}}
            def forward(self, head, ex, arm, support): return head(**ex.common, **support)
            def provenance(self): return {"synthetic":True}
        original = types.SimpleNamespace(
            JointRegionMemoryClassifier=lambda chunk_size:model_tests.JointRegionMemoryClassifier(
                chunk_size=chunk_size, hidden=16), _formal_runtime=lambda *a:None,
            _names=lambda *a:(self.train, self.val), _checked_verify=lambda *a:{"fixture":True},
            _assert_example=lambda *a:None, _seed=lambda seed:torch.manual_seed(seed))
        bindings = types.SimpleNamespace(Runtime=Runtime, original_train=original,
            legacy=types.SimpleNamespace(state=self.state),
            legacy_bridge=types.SimpleNamespace(verify_dependencies=lambda *a:None))
        def evaluate(runtime, heads, names, folder, **kw):
            if fail_eval and kw["step"] == 4 and kw["split"] == "val":
                raise ValueError("synthetic final validation failure")
            report = make_report(kw["step"],kw["split"],names,baseline.ARMS,kw["primary_endpoint"])
            write_report(folder,report,names)
            return report
        evaluator = types.SimpleNamespace(ARMS=tuple("efbdacrln"), CONTRASTS=evidence.CONTRASTS,
            evaluate=evaluate, print_table=lambda *a:None, _validate_probability=lambda *a:None)
        def named(path,label):
            return {"ablation_runner.py":ablation_runner,"supports.py":types.SimpleNamespace(SupportFactory=Factory),
                    "evaluation.py":evaluator}[Path(path).name]
        bridge = types.SimpleNamespace(named=named,load_runtime=lambda *a:bindings,
            runtime_arguments=lambda source,seed,output:types.SimpleNamespace(**owner.config),
            verify_ledgers=lambda *a:None,check_package=lambda *a:"synthetic-addon")
        return dict(plan=dict(endpoint_step=4,post500_lr_multiplier=.2,
            validation_steps=[0,4],addon_release_sha256="synthetic-addon",source={"config":self.config}),
            seed=3407,addon=ADDON,bridge=bridge,evidence_files={},manifest=dict(protocol_info=self.protocol,
                train_names=self.train,val_names=self.val,fixed_train_names=self.fixed))

    def run_case(self, fail=False):
        output = self.root / "output"
        output.mkdir()
        original_write = baseline.write_json
        def guarded(path, value):
            if Path(path).name == "result.json" and value.get("complete"):
                self.assertTrue(self.closed)
            return original_write(path,value)
        old_format = ablation_runner.FORMAT
        try:
            with mock.patch.object(torch.cuda,"is_available",return_value=True), \
                 mock.patch.object(torch.cuda,"is_current_stream_capturing",return_value=False), \
                 mock.patch.object(baseline,"write_json",side_effect=guarded), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = baseline.run_pair(self.context(fail),output)
        finally:
            ablation_runner.FORMAT = old_format
        return code, output, json.loads((output/"summary.json").read_text())

    def test_real_updates_checkpoint_roundtrip_and_promotion_after_runtime_close(self):
        code, output, summary = self.run_case()
        self.assertEqual(code,0,summary.get("error"))
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["updates_per_arm"],{"d":4,"h":4})
        saved = torch.load(output/"checkpoints/checkpoint_000004.pt",weights_only=False)
        self.assertEqual(saved["format"],baseline.FORMAT)
        self.assertFalse(torch.equal(saved["heads"]["d"]["joint_head.2.weight"],
                                     saved["heads"]["h"]["joint_head.2.weight"]))
        for opt in saved["optimizers"].values():
            self.assertTrue(all(int(v["step"])==4 for v in opt["state"].values()))
        self.assertTrue(all(e["complete"] for e in summary["evaluations"].values()))
        self.assertIn("d_minus_h", json.loads((output/"val_step_000004/result.json").read_text())["comparisons_pp"])

    def test_validation_failure_keeps_checkpoint_and_all_reports_pending(self):
        code, output, summary = self.run_case(True)
        self.assertEqual(code,1)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["last_complete_step"],4)
        self.assertTrue((output/"checkpoints/checkpoint_000004.pt").is_file())
        self.assertTrue((output/"evaluation_failure.json").is_file())
        self.assertFalse(json.loads((output/"val_step_000000/result.json").read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
