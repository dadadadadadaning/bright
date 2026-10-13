"""CPU contracts for continuation; never claimed as real BRIGHT/CUDA evidence."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

SOURCE_CANDIDATES = [
    Path(os.environ["JOINT_ORIGINAL_TOOL"]) if os.environ.get("JOINT_ORIGINAL_TOOL") else None,
    Path("/root/shared-nvme/BRIGHT-master/New/joint_train_ZAtNI2Wi/region_memory_joint_train_v1"),
    Path(__file__).resolve().parent.parent / "joint_recovery_source" / "region_memory_joint_train_v1",
    Path(__file__).resolve().parent / "original_tool" / "region_memory_joint_train_v1",
]
def _available_source(path):
    try:
        return path is not None and (path/"test_train.py").is_file()
    except OSError:
        return False


SOURCE = next((p for p in SOURCE_CANDIDATES if _available_source(p)), None)
if SOURCE is None:
    raise RuntimeError("Set JOINT_ORIGINAL_TOOL to the unchanged original joint training tool")
sys.path.insert(0, str(SOURCE))
from test_train import FakeRuntime
import train
from recover_training import continue_training, _batch_name, _reconstruct_iterator


class OrderedRuntime(FakeRuntime):
    """Local shuffle plus stochastic training inputs test order and RNG restoration."""
    instances = []

    def __init__(self, args, output):
        super().__init__(args, output)
        self.train_prepared = []

    def batches(self, names, train, seed):
        values = list(names)
        if train:
            random.Random(seed).shuffle(values)
        yield from values

    def prepare(self, image):
        example = super().prepare(image)
        if image in self.train_names:
            self.train_prepared.append(image)
            offset = random.random() + float(np.random.random()) + float(torch.rand(()))
            example.common["pixel_features"] += offset
        return example


class FailAfterPrefix(OrderedRuntime):
    def prepare(self, image):
        if image in self.val_names and len(self.train_prepared) == 2:
            raise RuntimeError("Synthetic post-checkpoint validation failure")
        return super().prepare(image)


class ContinuationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        OrderedRuntime.instances = []

    def tearDown(self):
        self.temp.cleanup()

    def args(self, name, **extra):
        fields = dict(repo="synthetic", v3_root="synthetic", protocol=None,
                      output=str(self.base/name), seed=3407, workers=0, device="cpu",
                      topk=64, chunk_size=2, mode="smoke", steps=6, eval_every=2,
                      lr=1e-4, weight_decay=5e-3)
        fields.update(extra)
        return SimpleNamespace(**fields)

    def prefix(self):
        args = self.args("failed")
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "post-checkpoint"):
            train.run_training(args, runtime_factory=FailAfterPrefix)
        checkpoint = self.base/"failed"/"checkpoints"/"checkpoint_000002.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        rows = [json.loads(x) for x in (self.base/"failed"/"train.jsonl").read_text().splitlines()]
        return saved, rows, checkpoint

    def recovery(self, saved, rows, name="continued", factory=OrderedRuntime, **extra):
        args = self.args(name, **extra)
        with redirect_stdout(io.StringIO()):
            result = continue_training(train, args, factory, saved, rows,
                         {"pool_fix": "synthetic-equivalent", "artifact_sha256": "synthetic"})
        return args, result

    def assert_nested_equal(self, left, right):
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_split_matches_full_weights_optimizers_rng_and_data_order_cpu(self):
        with redirect_stdout(io.StringIO()):
            full = train.run_training(self.args("full"), runtime_factory=OrderedRuntime)
        saved, rows, checkpoint = self.prefix()
        prefix_digest = train.digest(checkpoint)
        args, result = self.recovery(saved, rows)
        completed = torch.load(result["last_checkpoint"], map_location="cpu", weights_only=False)
        full_saved = torch.load(full["last_checkpoint"], map_location="cpu", weights_only=False)
        for key in ("heads", "optimizers"):
            self.assert_nested_equal(completed[key], full_saved[key])
        self.assert_nested_equal(completed["rng"]["torch_cpu"], full_saved["rng"]["torch_cpu"])
        self.assertEqual(completed["rng"]["python"], full_saved["rng"]["python"])
        np.testing.assert_equal(completed["rng"]["numpy"], full_saved["rng"]["numpy"])
        actual_rows = [json.loads(x) for x in (Path(args.output)/"train.jsonl").read_text().splitlines()]
        full_rows = [json.loads(x) for x in (self.base/"full"/"train.jsonl").read_text().splitlines()]
        self.assertEqual(actual_rows[:2], rows)
        self.assertEqual([(x["image"],x["epoch"]) for x in actual_rows],
                         [(x["image"],x["epoch"]) for x in full_rows])
        self.assertEqual(train.digest(checkpoint), prefix_digest)
        self.assertTrue(result["complete"])
        self.assertFalse(result["full_validation_evidence"])
        self.assertFalse(result["exact_historical_resume_claim"])
        self.assertEqual(result["updates_per_arm"], {a:6 for a in train.ARMS})
        self.assertEqual(result["new_updates_per_arm"], {a:4 for a in train.ARMS})
        self.assertEqual(result["gradient_evidence"], full["gradient_evidence"])
        self.assertEqual(result["training_coverage"], full["training_coverage"])
        self.assertEqual(completed["resume_state"]["epoch"], 1)
        self.assertEqual(completed["resume_state"]["cursor"], 2)
        sidecar = json.loads(Path(result["last_checkpoint"]+".json").read_text())
        self.assertEqual(sidecar["sha256"], train.digest(result["last_checkpoint"]))
        self.assertEqual(sidecar["protocol_info"], completed["protocol_info"])
        self.assertEqual(sidecar["recovery_binding"], completed["recovery_binding"])
        reevaluation = json.loads((Path(args.output)/"checkpoint_reval"/"result.json").read_text())
        self.assertEqual(reevaluation["checkpoint_step"], 2)
        self.assertFalse(reevaluation["primary_endpoint"])
        self.assertTrue(reevaluation["saved_predictions_reopened"])

    def test_journal_image_mismatch_fails_before_new_update(self):
        saved, rows, _ = self.prefix()
        rows = copy.deepcopy(rows)
        rows[0]["image"] = "train_a" if rows[0]["image"] != "train_a" else "train_b"
        with self.assertRaisesRegex(ValueError, "data order differs"):
            self.recovery(saved, rows)
        runtime = OrderedRuntime.instances[-1]
        self.assertEqual(runtime.train_prepared, [])
        summary = json.loads((self.base/"continued"/"summary.json").read_text())
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["new_updates_per_arm"], {a:0 for a in train.ARMS})

    def test_journal_nan_and_missing_gradient_rejected_before_runtime(self):
        saved, rows, _ = self.prefix()
        for arm in train.ARMS:
            bad = copy.deepcopy(rows)
            bad[1]["heads"][arm]["grad_norm"] = float("nan")
            count = len(OrderedRuntime.instances)
            with self.assertRaisesRegex(ValueError, "gradient evidence"):
                self.recovery(saved, bad, name="nan_"+arm)
            self.assertEqual(len(OrderedRuntime.instances), count)
            bad = copy.deepcopy(rows)
            for row in bad:
                row["heads"][arm]["reader_grad_abs_sum"] = 0
            with self.assertRaisesRegex(ValueError, "no Reader gradient"):
                self.recovery(saved, bad, name="zero_"+arm)

    def test_all_three_optimizer_and_equal_update_contracts(self):
        saved, rows, _ = self.prefix()
        bad = copy.deepcopy(saved)
        bad["updates_per_arm"]["rf"] = 1
        with self.assertRaisesRegex(ValueError, "equal updates"):
            self.recovery(bad, rows)
        bad = copy.deepcopy(saved)
        del bad["optimizers"]["rt"]
        with self.assertRaisesRegex(ValueError, "three heads and optimizers"):
            self.recovery(bad, rows)

    def test_candidate_prevalidation_failure_stops_without_updates(self):
        saved, rows, _ = self.prefix()
        class BadCandidate(OrderedRuntime):
            def prepare(self, image):
                if image in self.val_names:
                    raise RuntimeError("Candidate cannot process validation")
                return super().prepare(image)
        with self.assertRaisesRegex(RuntimeError, "cannot process validation"):
            self.recovery(saved, rows, factory=BadCandidate)
        self.assertEqual(OrderedRuntime.instances[-1].train_prepared, [])
        self.assertFalse((self.base/"continued"/"checkpoints").exists())

    def test_original_and_runtime_ledger_fingerprints_preserved(self):
        saved, rows, _ = self.prefix()
        inherited = self.base/"inherited-data"
        inherited.write_text("frozen old input")
        added = self.base/"newly-seen-data"
        added.write_text("frozen new input")
        saved["input_file_sha256"] = {str(inherited):train.digest(inherited)}
        class LedgerRuntime(OrderedRuntime):
            def __init__(self, args, output):
                super().__init__(args, output)
                self.input_ledger = {added:train.digest(added)}
                self.protocol_info["recovery_binding"] = {"runtime_candidate":True}
        _, result = self.recovery(saved, rows, factory=LedgerRuntime)
        completed = torch.load(result["last_checkpoint"], map_location="cpu", weights_only=False)
        self.assertEqual(completed["input_file_sha256"],
                         {str(inherited):train.digest(inherited),str(added):train.digest(added)})
        inherited.write_text("changed")
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            self.recovery(saved, rows, name="tampered")

    def test_changed_budget_schedule_and_old_output_rejected(self):
        saved, rows, _ = self.prefix()
        with self.assertRaisesRegex(ValueError, "configuration: steps"):
            self.recovery(saved, rows, steps=7)
        path = self.base/"existing"
        path.mkdir()
        (path/"marker").write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.recovery(saved, rows, name="existing")
        self.assertEqual((path/"marker").read_text(), "preserve")

    def test_input_changed_during_continuation_fails_final_integrity(self):
        saved, rows, _ = self.prefix()
        inherited = self.base/"inherited-input"
        inherited.write_text("preserve")
        saved["input_file_sha256"] = {str(inherited):train.digest(inherited)}
        class MutatingRuntime(OrderedRuntime):
            def prepare(self, image):
                result = super().prepare(image)
                if image in self.train_names:
                    inherited.write_text("changed after prefix validation")
                return result
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            self.recovery(saved, rows, factory=MutatingRuntime)
        report = json.loads((self.base/"continued"/"summary.json").read_text())
        self.assertFalse(report["complete"])
        self.assertTrue(report["training_complete"])
        self.assertFalse(report["full_validation_evidence"])
        self.assertTrue(Path(report["last_checkpoint"]).is_file())

    def test_real_single_image_batch_and_epoch_rollover_reconstruction(self):
        class RealBatches(OrderedRuntime):
            def batches(self, names, train, seed):
                for name in super().batches(names, train, seed):
                    yield (None,None,None,None,[name])
        runtime = RealBatches(self.args("unused"), self.base)
        rows = []
        for epoch in range(2):
            for batch in runtime.batches(runtime.train_names, True, 3407+epoch):
                rows.append(dict(step=len(rows)+1,epoch=epoch,image=_batch_name(batch)))
        epoch, cursor, iterator = _reconstruct_iterator(runtime, runtime.train_names, rows[:5], 3407)
        self.assertEqual((epoch,cursor),(1,1))
        self.assertEqual(_batch_name(next(iterator)), rows[5]["image"])


if __name__ == "__main__":
    unittest.main()
