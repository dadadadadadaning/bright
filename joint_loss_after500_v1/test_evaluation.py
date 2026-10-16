"""CPU-only evaluation evidence contracts; no BRIGHT/CUDA/Test validation."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import continuation
import evaluation
from objectives import objective_terms


def simple_metrics(matrix):
    cm = np.asarray(matrix)
    tp, gt, pred = np.diag(cm), cm.sum(1), cm.sum(0)
    def ratio(numerator, denominator):
        return [float(n) / float(d) if d else None for n, d in zip(numerator, denominator)]
    iou = ratio(tp, gt + pred - tp)
    return {"confusion": cm.tolist(), "iou": iou,
            "precision": ratio(tp, pred), "recall": ratio(tp, gt),
            "foreground_miou": sum(x or 0 for x in iou[1:]) / 3}


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv2d(3, 4, 1)
        self.dropout = torch.nn.Dropout(.2)

    def forward(self, *, features, offset, **unused):
        assert not self.training, "Evaluation must disable head training"
        return {"probabilities": (self.projection(self.dropout(features)) + offset).softmax(1)}


class EvaluationRuntime:
    def __init__(self, unsafe=False, query_gt=False):
        self.train_names = ["train0", "train1"]
        self.val_names = ["val0", "val1"]
        self.device = torch.device("cpu")
        self.protocol_info = {"synthetic": True}
        self.bank = {"events": torch.tensor([7]), "images": torch.tensor([7]),
                     "eligible": torch.tensor([True])}
        self.unsafe = unsafe
        self.query_gt = query_gt
        self.prepared = []
        self.reference_calls = 0

    def loss(self, probability, target):
        self.reference_calls += 1
        return objective_terms(probability, target)["loss"]

    def batches(self, names, train, seed):
        random.random()
        np.random.uniform()
        torch.rand(1)
        yield from reversed(names) if train else names

    def prepare(self, name):
        self.prepared.append(name)
        target = torch.tensor([[[0, 1], [2, 255]]]) if name.endswith("0") else torch.tensor([[[0, 0], [3, 3]]])
        return SimpleNamespace(
            image=name, event="event_" + name[-1],
            common={"features": torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2) / 11,
                    "base_probability_full": torch.full((1, 4, 2, 2), .25)},
            supports={"d": {"offset": .1, "support_ids": torch.tensor([[0]])}},
            target=target,
            diagnostics={"query_GT_used_for_supports": self.query_gt,
                         "event_id": 7 if self.unsafe else 8, "image_id": 8})


class EvaluationContracts(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name) / "result"
        head = TinyHead()
        self.heads = {name: copy.deepcopy(head) for name in continuation.BINDINGS}
        self.heads["d_no_bg_lovasz"].eval()
        # A mixed parent/child training mode must also survive evaluation.
        self.heads["d_control"].dropout.eval()
        self.original_train = SimpleNamespace(_assert_example=lambda x: None, _metrics=simple_metrics)

    def tearDown(self):
        self.temp.cleanup()
        torch.set_num_threads(self.threads)

    def modes(self):
        return {name: [m.training for m in head.modules()] for name, head in self.heads.items()}

    def evaluate(self, runtime, split="val", save=True):
        names = runtime.val_names if split == "val" else runtime.train_names
        with redirect_stdout(io.StringIO()):
            return evaluation.evaluate(runtime, self.heads, names, self.folder,
                step=1500, split=split, seed=3407, original_train=self.original_train,
                save_predictions=save)

    def test_synthetic_small_val_is_not_formal_and_archives_reopened(self):
        runtime = EvaluationRuntime()
        modes, rng = self.modes(), continuation.capture_rng()
        result = self.evaluate(runtime)
        for key in ("execution_complete", "smoke", "final_integrity_pending", "saved_predictions_reopened"):
            self.assertTrue(result[key], key)
        for key in ("complete", "full_validation_evidence", "whole_val450_processed", "test_accessed", "training"):
            self.assertFalse(result[key], key)
        self.assertEqual(result["image_count"], 2)
        self.assertEqual(result["valid_pixel_count"], 7)
        self.assertEqual(len(list((self.folder / "predictions").glob("*.npz"))), 2)
        self.assertEqual(runtime.prepared, runtime.val_names)
        self.assertEqual(modes, self.modes())
        self.assertEqual(continuation.state_fingerprint(rng), continuation.state_fingerprint(continuation.capture_rng()))
        # Reference loss is checked on the first image of each path only.
        self.assertEqual(runtime.reference_calls, len(evaluation.PATHS))
        self.assertEqual(result["metrics"]["d_control"], result["metrics"]["d_no_bg_lovasz"])

    def test_both_objectives_class_contributions_and_event_means_are_consistent(self):
        result = self.evaluate(EvaluationRuntime())
        rows = [json.loads(s) for s in (self.folder / "images.jsonl").read_text().splitlines()]
        for name, metrics in result["metrics"].items():
            self.assertEqual(metrics["class_valid_counts"], [3, 1, 1, 2])
            self.assertEqual(metrics["class_present_image_counts"], [2, 1, 1, 1])
            self.assertAlmostEqual(metrics["mean_image_loss"], metrics["mean_image_nll"] + metrics["mean_image_lovasz"], places=6)
            self.assertAlmostEqual(metrics["mean_image_candidate_loss"], metrics["mean_image_nll"] + metrics["mean_image_candidate_lovasz"], places=6)
            self.assertAlmostEqual(metrics["mean_image_loss"] - metrics["mean_image_candidate_loss"], metrics["mean_image_background_lovasz_contribution"], places=6)
            self.assertAlmostEqual(sum(metrics["mean_image_class_nll_contribution"]), metrics["mean_image_nll"], places=6)
            self.assertAlmostEqual(sum(metrics["mean_image_class_lovasz_contribution"]), metrics["mean_image_lovasz"], places=6)
            self.assertAlmostEqual(sum(metrics["mean_image_class_lovasz_contribution"][1:]), metrics["mean_image_candidate_lovasz"], places=6)
            for key in evaluation.SCALAR_TERMS:
                self.assertAlmostEqual(metrics["mean_image_" + key], np.mean([r[key][name] for r in rows]))
                event_mean = sum(result["per_event"][e][name]["mean_image_" + key] * count for e, count in result["per_event_image_count"].items()) / result["image_count"]
                self.assertAlmostEqual(metrics["mean_image_" + key], event_mean)
        self.assertEqual(result["trained_objective"]["d_control"], "loss")
        self.assertEqual(result["trained_objective"]["d_no_bg_lovasz"], "candidate_loss")
        out = io.StringIO()
        with redirect_stdout(out):
            evaluation.print_table(result)
        self.assertIn("common original loss | candidate loss", out.getvalue())
        self.assertIn("candidate objective alone is not a benefit", out.getvalue())

    def test_fixed_train_keeps_set_and_records_observed_order(self):
        runtime = EvaluationRuntime()
        result = self.evaluate(runtime, split="fixed_train", save=False)
        self.assertEqual(runtime.prepared, list(reversed(runtime.train_names)))
        self.assertEqual(result["names_sha256"], evaluation.names_hash(sorted(runtime.train_names)))
        self.assertNotEqual(result["order_sha256"], result["requested_order_sha256"])
        self.assertFalse(result["full_validation_evidence"])
        self.assertFalse(result["prediction_evidence_complete"])
        self.assertIn("not full Train2918", result["fixed_train_scope"])

    def test_same_event_support_fails_and_restores_all_modes_and_rng(self):
        initial, modes = continuation.capture_rng(), self.modes()
        with self.assertRaisesRegex(ValueError, "query events"):
            self.evaluate(EvaluationRuntime(unsafe=True))
        self.assertEqual(continuation.state_fingerprint(initial), continuation.state_fingerprint(continuation.capture_rng()))
        self.assertEqual(modes, self.modes())
        self.assertFalse((self.folder / "result.json").exists())

    def test_query_gt_support_leak_refused(self):
        with self.assertRaisesRegex(ValueError, "Query GT separation"):
            self.evaluate(EvaluationRuntime(query_gt=True))

    def test_test_cross_split_and_train_prediction_export_refused(self):
        runtime = EvaluationRuntime()
        for split, names, save in (("test", runtime.val_names, False), ("val", runtime.train_names, False), ("fixed_train", runtime.train_names, True)):
            with self.subTest(split=split, save=save), self.assertRaises(ValueError):
                evaluation.evaluate(runtime, self.heads, names, self.folder,
                    step=1500, split=split, seed=3407, original_train=self.original_train,
                    save_predictions=save)

    def test_val_order_checked(self):
        class BadOrder(EvaluationRuntime):
            def batches(self, names, train, seed):
                yield from reversed(names)
        with self.assertRaisesRegex(ValueError, "Val loader"):
            self.evaluate(BadOrder())
        self.assertFalse((self.folder / "result.json").exists())

    def test_saved_archive_corruption_rejected(self):
        reopen = evaluation._reopen_predictions
        def corrupt(folder, journal, *args):
            with (Path(folder) / journal[0]["file"]).open("ab") as stream:
                stream.write(b"corrupt")
            return reopen(folder, journal, *args)
        with patch.object(evaluation, "_reopen_predictions", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "prediction hash mismatch"):
                self.evaluate(EvaluationRuntime())
        self.assertFalse((self.folder / "result.json").exists())

    def test_transition_distinguishes_rescue_harm_with_equal_net_tp(self):
        target = np.array([2, 2, 2, 1, 255], dtype=np.uint8)
        before = np.array([0, 2, 1, 1, 0], dtype=np.uint8)
        after = np.array([2, 0, 1, 2, 3], dtype=np.uint8)
        counts = evaluation.transition(target, before, after)
        report = evaluation.transition_report(counts, "old", "new")
        self.assertEqual(report["rescued_by_gt_class"][2], 1)
        self.assertEqual(report["harmed_by_gt_class"][2], 1)
        self.assertEqual(report["net_correct_change_by_gt_class"][2], 0)
        self.assertEqual(report["damaged_correct_to_background"], 1)
        self.assertEqual(report["damaged_background_to_correct"], 1)
        self.assertEqual(report["damaged_to_background_net_change"], 0)
        self.assertEqual(report["correct_intact_changed_to_damaged"], 1)
        self.assertEqual(int(counts.sum()), 4)

    def test_initial_both_forks_compare_old_d_and_grid_with_pp_tolerance(self):
        matrix = np.diag([10, 20, 30, 40])
        old = {"image_count": 450, "execution_complete": True,
               "metrics": {name: simple_metrics(matrix) for name in ("grid_probq", "d")}}
        current = {"image_count": 450, "execution_complete": True,
                   "metrics": {name: simple_metrics(matrix) for name in evaluation.PATHS}}
        self.assertTrue(evaluation.compare_initial_to_old(current, old)["passed"])
        current["metrics"]["d_no_bg_lovasz"]["iou"][2] -= .0003
        self.assertFalse(evaluation.compare_initial_to_old(current, old)["passed"])
        current["metrics"]["d_no_bg_lovasz"]["iou"][2] += .0003
        current["image_count"] = 2
        self.assertFalse(evaluation.compare_initial_to_old(current, old)["passed"])

    def test_contrasts_have_only_loss_and_frozen_reference_comparisons(self):
        result = {"metrics": {"grid_probq": {"foreground_miou": .60},
                              "d_control": {"foreground_miou": .61},
                              "d_no_bg_lovasz": {"foreground_miou": .62}}}
        contrast = evaluation.metriccontrast(result)
        self.assertEqual(set(contrast), {"d_no_bg_lovasz_minus_d_control", "d_control_minus_grid_probq", "d_no_bg_lovasz_minus_grid_probq"})
        self.assertAlmostEqual(contrast["d_no_bg_lovasz_minus_d_control"], 1.)


if __name__ == "__main__":
    unittest.main()
