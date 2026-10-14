"""Synthetic contracts only; no BRIGHT Test image is read here."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import tempfile
import unittest

import numpy as np
import torch

import locked_test as candidate


class LockedTestContracts(unittest.TestCase):
    def test_val_selection_precedes_test_and_has_stable_tie_order(self):
        values = {arm: {"foreground_miou": 0.5} for arm in candidate.ARMS}
        self.assertEqual(candidate._val_selection(values)["selected"], "grid_probq")
        values["d"]["foreground_miou"] = 0.51
        self.assertEqual(candidate._val_selection(values)["selected"], "d")
        with self.assertRaisesRegex(ValueError, "Invalid final Val450"):
            candidate._val_selection({**values, "b": {"foreground_miou": float("nan")}})

    def test_state_hash_matches_released_training_algorithm(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.BatchNorm1d(3))
        expected = hashlib.sha256()
        for name, value in model.state_dict().items():
            t = value.detach().cpu().contiguous()
            expected.update(name.encode())
            expected.update(str(t.dtype).encode())
            expected.update(str(tuple(t.shape)).encode())
            expected.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        self.assertEqual(candidate.state_hash(model.state_dict()), expected.hexdigest())
        changed = dict(model.state_dict())
        changed["1.num_batches_tracked"] = changed["1.num_batches_tracked"] + 1
        self.assertNotEqual(candidate.state_hash(changed), expected.hexdigest())

    def test_four_class_and_foreground_are_distinct(self):
        cm = np.diag([90, 60, 30, 10]).astype(np.int64)
        def base_metric(value):
            iou = [1.0, 0.8, 0.4, 0.2]
            return {"iou": iou, "foreground_miou": sum(iou[1:]) / 3}
        result = candidate.metrics(cm, SimpleNamespace(_metrics=base_metric))
        self.assertAlmostEqual(result["foreground_miou"], 1.4 / 3)
        self.assertAlmostEqual(result["four_class_miou"], 2.4 / 4)

    def test_fixed_test_split_rejects_overlap_and_missing_gt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "splitname" / "standard_ML"
            split.mkdir(parents=True)
            train = ["train_%04d" % i for i in range(2918)]
            val = ["val_%04d" % i for i in range(450)]
            test = ["test_%04d" % i for i in range(878)]
            for name, rows in (("train_set.txt", train), ("val_set.txt", val), ("test_set.txt", test)):
                (split / name).write_text("\n".join(rows) + "\n")
            e0 = root / "e0.json"
            e0.write_text(json.dumps({"input_paths": {"train_list": str(split / "train_set.txt"),
                                                    "val_list": str(split / "val_set.txt")},
                                      "arguments": {"dataset": str(root)}}))
            protocol = root / "source.json"
            protocol.write_text(json.dumps({"format": "SOFT_MEMORY_Q_SOURCE_V1",
                                            "e0_metadata_protocol": str(e0)}))
            formal = SimpleNamespace(manifest={"train_names": train, "val_names": val})
            with self.assertRaisesRegex(ValueError, "ground truth"):
                candidate._canonical_test_split(protocol, formal)
            (split / "test_set.txt").write_text("\n".join(test[:-1] + [val[0]]) + "\n")
            with self.assertRaisesRegex(ValueError, "overlaps"):
                candidate._canonical_test_split(protocol, formal)

    def test_saved_prediction_reopen_detects_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "predictions").mkdir()
            names = ["first", "second"]
            rows = []
            cm = {a: np.zeros((4, 4), dtype=np.int64) for a in candidate.ARMS}
            event_cm = {"event": {a: np.zeros((4, 4), dtype=np.int64) for a in candidate.ARMS}}
            def confusion(target, pred):
                return np.bincount(4*target.ravel().astype(np.int64)+pred.ravel().astype(np.int64),
                                   minlength=16).reshape(4, 4)
            for index, name in enumerate(names):
                target = np.array([[[0, 1], [2, 3]]], dtype=np.uint8)
                preds = {a: target.copy() for a in candidate.ARMS}
                path = folder / ("predictions/%04d.npz" % index)
                np.savez_compressed(path, target=target,
                                    **{"prediction_" + a: pred for a,pred in preds.items()})
                current = {a: confusion(target, pred) for a,pred in preds.items()}
                for arm in candidate.ARMS:
                    cm[arm] += current[arm]
                    event_cm["event"][arm] += current[arm]
                rows.append(dict(index=index, image=name, event="event",
                                 file="predictions/%04d.npz" % index,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                 confusion={a: current[a].tolist() for a in candidate.ARMS}))
            tool = SimpleNamespace(common=SimpleNamespace(digest=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()),
                                   train=SimpleNamespace(_confusion=confusion))
            candidate._reopen(folder, rows, cm, event_cm, names, tool)
            rows[0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "file changed"):
                candidate._reopen(folder, rows, cm, event_cm, names, tool)

    def test_incomplete_formal_run_stops_before_test_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train" / "final_val").mkdir(parents=True)
            (root / "workflow_state.json").write_text(json.dumps({"complete": False}))
            for name in ("summary.json", "manifest.json"):
                (root / "train" / name).write_text("{}")
            (root / "train" / "final_val" / "result.json").write_text("{}")
            tool = SimpleNamespace(common=SimpleNamespace(source_hash=lambda: "test-source",
                                                          FORMAT="SRAC_SOFT_MEMORY_Q_REBASED_V1"))
            with self.assertRaisesRegex(ValueError, "full workflow"):
                candidate.validate_formal(root, tool)

    def test_output_is_never_created_inside_training_tool_or_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = root / "training"
            training.mkdir()
            tool_root = root / "tool"
            tool_root.mkdir()
            dataset = root / "dataset"
            dataset.mkdir()
            e0 = root / "e0.json"
            e0.write_text(json.dumps({"arguments": {"dataset": str(dataset)}}))
            source = root / "source.json"
            source.write_text(json.dumps({"e0_metadata_protocol": str(e0)}))
            formal = SimpleNamespace(root=training, train=training / "train",
                saved={"config": {"protocol": str(source)}})
            tool = SimpleNamespace(root=tool_root)
            for forbidden in (training / "test", tool_root / "test", dataset / "test"):
                with self.assertRaisesRegex(ValueError, "outside"):
                    candidate._protect_output(forbidden, formal, tool)
                self.assertFalse(forbidden.exists())
            self.assertEqual(candidate._protect_output(root / "allowed", formal, tool),
                             root / "allowed")

    def test_training_code_tamper_is_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tool = root / "soft_memory_q_rebased_v1_fix2"
            tool.mkdir()
            code = tool / "common.py"
            code.write_text("VALUE = 1\n")
            (tool / "SHA256SUMS.json").write_text(json.dumps({
                "common.py": hashlib.sha256(code.read_bytes()).hexdigest()}))
            run = root / "run"
            run.mkdir()
            source = hashlib.sha256(b"common.py" + code.read_bytes()).hexdigest()
            (run / "workflow_state.json").write_text(json.dumps({"code_sha256": source}))
            self.assertEqual(candidate._verify_tool_before_import(tool, run), source)
            code.write_text("VALUE = 2\n")
            with self.assertRaisesRegex(ValueError, "before import"):
                candidate._verify_tool_before_import(tool, run)


if __name__ == "__main__":
    unittest.main()
