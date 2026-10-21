"""CPU/synthetic contracts; no claim of reproducing the server CUDA failure."""
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

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "bright_route_eval_probe_v1.pyz"
sys.path.insert(0, str(ARCHIVE))
sys.path.extend([os.environ["BRIGHT_ROUTE_TEST_SOURCE"], os.environ["BRIGHT_ABLATION_TEST_SOURCE"]])
import torch
from test_route_trace import bindings
from eval_route_observer import SelectiveRouteTrace, track_evaluation
import route_eval_probe as driver


class ObserverTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.q = torch.tensor([[[.2, .3], [.8, .7]]])
        self.ids = torch.tensor([[[0, 1], [2, 3]]])
        self.cls, self.model, self.slots, self.calls, self.error = bindings()
        self.position = dict(index=167, image="target", active_arm="r", stage="forward_begin")
        self.trace = SelectiveRouteTrace(torch, self.cls, lambda *a, **k: None, self.position)

    def test_prefix_accepted_routes_are_not_copied_or_recomputed(self):
        with mock.patch.object(self.trace, "_capture", side_effect=AssertionError("prefix observation")):
            with self.trace:
                self.cls().forward(self.q, self.ids)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.trace.diagnostic_error_count, 0)

    def test_dense_keeps_input_bytes_rng_and_exact_validator_return(self):
        self.trace.dense = True
        before_q, before_ids, rng = self.q.clone(), self.ids.clone(), torch.get_rng_state()
        original = self.model["validate_routes"]
        with self.trace:
            result = self.cls().forward(self.q, self.ids)
        self.assertIs(result[0], self.ids)
        self.assertTrue(torch.equal(before_q, self.q))
        self.assertTrue(torch.equal(before_ids, self.ids))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIs(self.model["validate_routes"], original)
        self.assertIs(self.slots["validate_routes"], original)
        record = self.trace.summary()["validators"]["sparse_unpool"]["latest"]
        self.assertEqual(record["callsites"][0]["path"], "synthetic_slots.py")
        self.assertEqual(record["callsites"][1]["path"], "synthetic_model.py")

    def test_negative_and_wrong_mass_evidence_preserves_exception(self):
        bad = torch.tensor([[[-.1, .5], [1.1, .8]]])
        with self.assertRaises(ValueError) as caught:
            with self.trace:
                self.cls().forward(bad, self.ids)
        self.assertIs(caught.exception, self.error)
        failure = self.trace.summary()["first_validation_failure"]
        self.assertEqual(failure["arm"], "r")
        self.assertEqual(failure["weights"]["negative_count"], 1)
        self.assertEqual(failure["cpu_float64_sums"]["bad_count"], 1)
        self.assertEqual(failure["bad_pixel_count"], 2)

    def test_same_bytes_previously_accepted_are_explicitly_identified(self):
        count = 0
        def flaky(q, ids, *args):
            nonlocal count
            count += 1
            if count == 2:
                raise self.error
            return ids
        self.model["validate_routes"] = self.slots["validate_routes"] = flaky
        self.trace.dense = True
        with self.assertRaises(ValueError):
            with self.trace:
                self.cls().forward(self.q, self.ids)
        failure = self.trace.summary()["first_validation_failure"]
        self.assertEqual(failure["same_bytes_accepted_earlier"]["sequence"], 1)
        self.assertTrue(failure["pre_post_bytes_equal"])

    def test_runtime_exception_causes_no_new_tensor_reads(self):
        error = RuntimeError("synthetic device failure")
        def fail(*args): raise error
        self.model["validate_routes"] = fail
        with mock.patch.object(self.trace, "_cpu_copy", side_effect=AssertionError("unexpected tensor read")):
            with self.assertRaises(RuntimeError) as caught:
                with self.trace:
                    self.cls().forward(self.q, self.ids)
        self.assertIs(caught.exception, error)
        self.assertTrue(self.trace.tensor_reads_blocked)
        self.assertNotIn("weights", self.trace.summary()["first_validation_failure"])

    def test_cpu_snapshot_is_owned_and_savable_after_failure(self):
        with self.assertRaises(ValueError):
            with self.trace:
                self.cls().forward(-self.q, self.ids)
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(self.trace, "_cpu_copy", side_effect=AssertionError("late GPU read")):
                self.trace.save_cpu_case(Path(temp) / "case.pt")
            saved = torch.load(Path(temp) / "case.pt", weights_only=False)
            self.assertEqual(saved["validators"]["classifier"]["first_failure_case"]["q"].device.type, "cpu")


class ActualEvaluatorTests(unittest.TestCase):
    def test_original_evaluator_failure_records_active_r_not_last_success_d(self):
        import test_evaluation as fixtures
        fixtures.EvaluationContracts.setUpClass()
        fixture = fixtures.EvaluationContracts()
        fixture.setUp()
        try:
            fixture.heads.update(r=copy.deepcopy(fixture.heads["d"]), l=copy.deepcopy(fixture.heads["d"]))
            runtime = fixture.runtime()
            cls, _, _, _, _ = bindings()
            position = dict(index=-1)
            events = []
            def emit(event, **fields): events.append((event, fields))
            trace = SelectiveRouteTrace(torch, cls, emit, position)
            forward = fixture.factory.forward
            failure = ValueError("synthetic r forward failure")
            def call(head, example, arm, support):
                if arm == "r": raise failure
                return forward(head, example, arm, support)
            fixture.factory.forward = call
            original_validator = fixtures.evaluation._validate_probability
            with self.assertRaises(ValueError) as caught:
                with track_evaluation(runtime, fixture.factory, fixtures.evaluation,
                                      runtime.val_names, 1, trace, emit):
                    fixture.run_evaluation(runtime)
            self.assertIs(caught.exception, failure)
            self.assertEqual(position["active_arm"], "r")
            self.assertEqual(position["stage"], "forward_begin")
            self.assertEqual([x[1]["active_arm"] for x in events if x[0] == "forward_end"], list("efbd"))
            self.assertNotIn("prepare", vars(runtime))
            self.assertIs(fixture.factory.forward, call)
            self.assertIs(fixtures.evaluation._validate_probability, original_validator)
        finally:
            fixture.tearDown()

    def test_real_classifier_output_and_optimizer_are_unchanged(self):
        package = types.ModuleType("_route_probe_reference")
        package.__path__ = [str(Path(os.environ["BRIGHT_ABLATION_REFERENCE"]) / "deps/original")]
        sys.modules[package.__name__] = package
        module = importlib.import_module(package.__name__ + ".test_model")
        inputs = module.fixture()
        model = module.JointRegionMemoryClassifier(hidden=16, heads=4, chunk_size=2)
        with torch.no_grad(): model.joint_head[-1].weight.normal_(0, .1)
        optimizer = torch.optim.AdamW(model.parameters())
        before = copy.deepcopy(model.state_dict())
        original_opt = copy.deepcopy(optimizer.state_dict())
        position = dict(index=167, image="target", active_arm="r", stage="forward_begin")
        trace = SelectiveRouteTrace(torch, type(model), lambda *a, **k: None, position)
        trace.dense = True
        with torch.no_grad():
            reference = model(**inputs)["probabilities"]
            with trace:
                observed = model(**inputs)["probabilities"]
        self.assertTrue(torch.equal(reference, observed))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertEqual(original_opt, optimizer.state_dict())
        self.assertEqual(trace.validator_failure_count, 0)
        self.assertGreaterEqual(trace.validation_call_count, 5)


class ProvenanceTests(unittest.TestCase):
    def test_bundle_preserves_original_resume_helpers(self):
        with zipfile.ZipFile(ARCHIVE) as archive:
            ledger = json.loads(archive.read("BASE_HELPERS.json"))
            for name, digest in ledger["files"].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest, name)

    def test_failure_preflight_requires_zero_updates_and_correct_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = ["image_%s" % i for i in range(450)]
            names[167] = driver.TARGET_IMAGE
            context = dict(manifest=dict(val_names=names), parent=dict(sha256="cp"), evidence_files={})
            (root/"trainer.pyz").write_bytes(b"synthetic")
            (root/"traceback.txt").write_text("synthetic traceback")
            driver.write_json(root/"evaluation_failure.json", dict(index=167,image=driver.TARGET_IMAGE,
                error="ValueError: route weights must be nonnegative and sum to one"))
            status = dict(phase="failed",last_complete_step=1500,new_updates_per_arm={a:0 for a in driver.ARMS})
            driver.write_json(root/"summary.json", status)
            driver.write_json(root/"manifest.json", dict(val_names=names,parent=dict(sha256="cp")))
            (root/"val_step_001500").mkdir()
            (root/"val_step_001500/images.jsonl").write_text("".join(json.dumps(dict(image=n))+"\n" for n in names[:167]))
            with mock.patch.object(driver,"FAILURE_TRAINER_SHA",driver.sha256(root/"trainer.pyz")):
                driver.check_failure(context,root)
                status["new_updates_per_arm"]["e"] = 1
                driver.write_json(root/"summary.json",status)
                with self.assertRaisesRegex(ValueError,"zero-update"):
                    driver.check_failure(context,root)


class WorkerIntegrationTests(unittest.TestCase):
    """Real saved-state restoration and model forwards; synthetic data/runtime."""
    def run_case(self, fail):
        import checkpoint
        import trajectory
        import ablation_runner
        sys.path.append(os.environ["BRIGHT_RESUME_TEST_SOURCE"])
        from test_resume_integration import CPUState
        package = types.ModuleType("_route_worker_reference")
        package.__path__ = [str(Path(os.environ["BRIGHT_ABLATION_REFERENCE"])/"deps/original")]
        sys.modules[package.__name__] = package
        model_tests = importlib.import_module(package.__name__ + ".test_model")
        state = CPUState()
        inputs = model_tests.fixture()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        policy = root/"policy.json"
        policy.write_text(json.dumps(dict(numeric_policy=dict(num_threads=1,
            num_interop_threads=torch.get_num_interop_threads()))))
        data = root/"input.txt"
        data.write_text("synthetic input")
        ledger = {str(data):driver.sha256(data)}
        config = dict(chunk_size=2,lr=1e-4,weight_decay=.005,protocol=str(policy))
        protocol = dict(execution_settings=dict(cuda_launch_blocking=None))
        train, val = ["train_%d" % n for n in range(2500)], ["val0","val1","val2"]
        runtime_instances = []
        class Runtime:
            def __init__(self,args,output):
                self.protocol_info = protocol
                self.device = torch.device("cpu")
                self.input_ledger = {data:ledger[str(data)]}
                self.prepared = []
                runtime_instances.append(self)
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def evaluation_batches(self,names,seed): return list(names)
            def prepare(self,name):
                self.prepared.append(name)
                return types.SimpleNamespace(image=name, common={k:v for k,v in inputs.items() if not k.startswith("support_")})
        class Factory:
            def __init__(self,*args,**kw): pass
            def prepare(self,example,arms): return {a:{k:v for k,v in inputs.items() if k.startswith("support_")} for a in arms}
            def forward(self,model,example,arm,support):
                if fail and example.image == "val1" and arm == "r":
                    raise ValueError("synthetic R failure")
                return model(**example.common,**support)
        original = types.SimpleNamespace(JointRegionMemoryClassifier=lambda chunk_size:
            model_tests.JointRegionMemoryClassifier(hidden=16,heads=4,chunk_size=chunk_size),
            _formal_runtime=lambda *a:None,_names=lambda *a:(train,val),
            _checked_verify=lambda *a:None,_restored_guard=lambda *a:dict(failures=[]))
        heads, opts, _ = ablation_runner.model_copies(torch,original,state,driver.ARMS,config,torch.device("cpu"))
        for opt in opts.values():
            for p in opt.param_groups[0]["params"]:
                opt.state[p] = dict(step=torch.tensor(1500.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.ones_like(p))
        saved = dict(format="BRIGHT_SLOT_MEMORY_RESUME_CHECKPOINT_V1",step=1500,
            heads={a:copy.deepcopy(h.state_dict()) for a,h in heads.items()},
            optimizers={a:copy.deepcopy(o.state_dict()) for a,o in opts.items()},rng=state.capture_rng(),input_file_sha256=ledger)
        def evaluate(runtime,heads,names,folder,**kw):
            self.assertTrue(kw["smoke"])
            self.assertFalse(kw["primary_endpoint"])
            folder.mkdir()
            with (folder/"images.jsonl").open("w") as journal, torch.no_grad():
                for name in names:
                    example = runtime.prepare(name)
                    support = kw["factory"].prepare(example,heads)
                    for arm,head in heads.items():
                        p = kw["factory"].forward(head,example,arm,support[arm])["probabilities"]
                        evaluator._validate_probability(p,None,arm)
                    journal.write(json.dumps(dict(image=name))+"\n")
            return {}
        evaluator = types.SimpleNamespace(evaluate=evaluate,_validate_probability=lambda *a:None)
        binding = types.SimpleNamespace(Runtime=Runtime,original_train=original,
            legacy=types.SimpleNamespace(state=state),legacy_bridge=types.SimpleNamespace(verify_dependencies=lambda *a:None))
        modules = {"ablation_runner.py":ablation_runner,"supports.py":types.SimpleNamespace(SupportFactory=Factory),"evaluation.py":evaluator}
        bridge = types.SimpleNamespace(load_runtime=lambda *a:binding,named=lambda p,n:modules[Path(p).name],
            runtime_arguments=lambda *a,**kw:types.SimpleNamespace(**config),verify_ledgers=lambda *a:None,
            check_package=lambda *a:"synthetic")
        context = dict(manifest=dict(protocol_info=protocol,train_names=train,val_names=val),
            plan=dict(source=dict(config=config),addon_release_sha256="synthetic"),bridge=bridge,addon=root,
            train=root,parent=dict(path=root/"cp.pt",sha256="synthetic"),parent_journal=root/"train.jsonl",
            lineage=dict(synthetic=True),seed=3407,evidence_files=ledger)
        output = root/"output"
        output.mkdir()
        with mock.patch.object(torch.cuda,"is_available",return_value=True), \
             mock.patch.object(checkpoint,"load",return_value=(saved,dict(sha256="synthetic"))), \
             mock.patch.object(trajectory,"reconstruct_order",return_value=(None,train,None)), \
             mock.patch.object(driver,"TARGET_INDEX",1), \
             mock.patch.object(torch.optim.AdamW,"step",side_effect=AssertionError("optimizer update")), \
             contextlib.redirect_stdout(io.StringIO()):
            code = driver.run_probe(context,output)
        summary = driver.read_json(output/"summary.json")
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["full_validation_evidence"])
        self.assertEqual(summary["optimizer_updates"],0)
        self.assertEqual(runtime_instances[0].prepared,["val0","val1"])
        return code,summary

    def test_worker_prefix_success_does_not_promote_training(self):
        code, summary = self.run_case(False)
        self.assertEqual(code,0,summary.get("error"))
        self.assertTrue(summary["diagnostic_complete"])
        self.assertTrue(all(summary["integrity_checks"].values()))
        self.assertEqual(summary["completed_images"],2)

    def test_worker_failure_keeps_precise_arm_and_partial_count(self):
        code, summary = self.run_case(True)
        self.assertEqual(code,2)
        self.assertEqual(summary["error"],"ValueError: synthetic R failure")
        self.assertEqual(summary["last_position"]["active_arm"],"r")
        self.assertEqual(summary["completed_images"],1)


if __name__ == "__main__":
    unittest.main()
