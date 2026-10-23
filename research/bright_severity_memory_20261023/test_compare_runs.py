"""Stdlib checks for attribution and provenance in new fixed2500 comparisons."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analyze_results import CLASSES, metrics
from compare_runs import FORMAT, PLAN_FIELDS, compare


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


class CompareRunContracts(unittest.TestCase):
    def write_run(self, folder, variant, *, better=False, varied=False):
        folder.mkdir()
        names = [f"image_{i}" for i in range(450)]
        plan = {key: "same" for key in PLAN_FIELDS}
        plan.update(format=FORMAT, variant=variant, endpoint_step=2500,
                    read_weight=0, added_loss=None, method={"memory_gain_bound": 2.0})
        manifest = dict(format=FORMAT, plan=plan, seed=3407,
                        initial_core_head_sha256="same-core", initial_head_sha256="intentionally-different-"+variant,
                        train_names=["train1"], val_names=names, actual_training_names=["train1"],
                        fixed_train_names=["train1"], protocol_info={"execution_settings": {"cuda_launch_blocking": "1"}},
                        parameter_count=100 if variant == "original" else 104, support_arm="d",
                        runtime_config={"output": str(folder), "lr": 1e-4}, fresh_initialization=True,
                        test_accessed=False)
        write_json(folder/"manifest.json", manifest)
        write_json(folder/"summary.json", dict(format=FORMAT, complete=True, training_complete=True,
                   phase="complete", full_validation_evidence=True, smoke=False, test_accessed=False,
                   last_complete_step=2500, endpoint_step=2500, variant=variant, seed=3407))
        train = [dict(step=i, image="train1", event="event1", epoch=0, epoch_seed=3407,
                      lr=0.0001, optimizer_step=i, variant=variant, read_loss_evaluated=False) for i in range(1, 2501)]
        (folder/"train.jsonl").write_text("".join(json.dumps(r)+"\n" for r in train))
        dest = folder/"val_step_002500"
        dest.mkdir()
        rows = []
        for i, name in enumerate(names):
            cm = [[10, 1, 0, 0], [2, 5, 1, 0], [1, 2, 3, 0], [0, 1, 1, 4]]
            if varied and i == 0:
                cm = [[90000, 10000, 0, 0], [8000, 10000, 1000, 1000],
                      [0, 1000, 1000, 0], [0, 1000, 0, 1000]]
            pred = copy.deepcopy(cm)
            if better:
                pred[1][0] -= 1
                pred[1][1] += 1
                pred[2][1] -= 1
                pred[2][2] += 1
            rows.append(dict(index=i, image=name, event="event1" if i < 300 else "event2",
                             confusion={"grid_probq": cm, variant: pred}, test_accessed=False))
        journal = dest/"images.jsonl"
        journal.write_text("".join(json.dumps(r)+"\n" for r in rows))
        totals = {arm: [[sum(row["confusion"][arm][r][c] for row in rows) for c in range(4)] for r in range(4)]
                  for arm in ("grid_probq", variant)}
        report = dict(format=FORMAT+"_EVALUATION", complete=True,
                      final_integrity_pending=False, phase="complete", smoke=False, split="val", seed=3407,
                      step=2500, class_names=CLASSES, requested_names=names, image_count=450, variant=variant,
                      full_validation_evidence=True, journal_verified=True, test_accessed=False,
                      images_sha256=hashlib.sha256(journal.read_bytes()).hexdigest(),
                      metrics={arm: {"confusion": cm, "miou4_pct": -999} for arm, cm in totals.items()})
        path = dest/"result.json"
        write_json(path, report)
        return path

    def pair(self, temp, **kwargs):
        return self.write_run(Path(temp)/"a", "calibration", **kwargs), self.write_run(Path(temp)/"b", "severity_memory", better=True, **kwargs)

    def edit(self, path, callback):
        d = json.loads(Path(path).read_text())
        callback(d)
        write_json(path, d)

    def edit_rows(self, report, *, file="images.jsonl", root=False, callback):
        path = report.parent.parent/file if root else report.parent/file
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        callback(rows)
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        if not root:
            self.edit(report, lambda d: d.update(images_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def test_recomputes_metrics_and_net_error_changes_from_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = self.pair(temp)
            result, rows = compare(a, b)
            self.assertGreater(result["differences"]["miou4_pp"], 0)
            self.assertTrue(result["training_trajectory_verified"])
            self.assertEqual(result["net_error_direction_changes"]["damaged_to_intact"], -450)
            self.assertEqual(result["net_error_direction_changes"]["building_to_background"], -450)
            self.assertEqual(len(rows), 450)
            self.assertTrue(result["grid_confusion_identical"])
            self.assertEqual(result["grid_execution_difference"]["max_absolute_confusion_entry"], 0)

    def test_parameter_counts_and_full_hash_may_differ_but_core_must_match(self):
        with tempfile.TemporaryDirectory() as temp:
            a = self.write_run(Path(temp)/"a", "original")
            b = self.write_run(Path(temp)/"b", "severity_memory", better=True)
            result, _ = compare(a, b)
            self.assertEqual(result["parameter_counts"], {"before": 100, "after": 104})
            self.edit(b.parent.parent/"manifest.json", lambda d: d.update(initial_core_head_sha256="changed"))
            with self.assertRaisesRegex(ValueError, "initial_core_head"):
                compare(a, b)

    def test_dataset_metric_is_not_mean_image_metric(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = self.pair(temp, varied=True)
            result, rows = compare(a, b)
            aggregate = metrics(json.loads(b.read_text())["metrics"]["severity_memory"]["confusion"])["miou4_pct"]
            self.assertEqual(result["metrics"]["after"]["miou4_pct"], aggregate)
            self.assertGreater(abs(aggregate - sum(r["after_miou4_pct"] for r in rows)/450), 1)
            self.assertEqual(set(result["per_event"]), {"event1", "event2"})

    def test_rejects_incomplete_and_test_results(self):
        for changes in ({"complete": False}, {"final_integrity_pending": True}, {"image_count": 449},
                        {"split": "test"}, {"test_accessed": True}, {"step": 1500}):
            with tempfile.TemporaryDirectory() as temp:
                a, b = self.pair(temp)
                self.edit(b, lambda d: d.update(changes))
                with self.assertRaises(ValueError):
                    compare(a, b)

    def test_rejects_journal_hash_and_aggregate_mismatches(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = self.pair(temp)
            self.edit(b, lambda d: d.update(images_sha256="bad"))
            with self.assertRaisesRegex(ValueError, "hash"):
                compare(a, b)
            self.edit(b, lambda d: d.update(images_sha256=hashlib.sha256((b.parent/"images.jsonl").read_bytes()).hexdigest()))
            self.edit(b, lambda d: d["metrics"]["severity_memory"]["confusion"][0].__setitem__(0, 999))
            with self.assertRaisesRegex(ValueError, "aggregate"):
                compare(a, b)

    def test_rejects_training_order_lr_and_partial_journals(self):
        for field, value in (("epoch", 1), ("lr", 0.0002), ("epoch_seed", 42)):
            with tempfile.TemporaryDirectory() as temp:
                a, b = self.pair(temp)
                self.edit_rows(b, file="train.jsonl", root=True, callback=lambda rows: rows[0].update({field: value}))
                with self.assertRaisesRegex(ValueError, "training trajectory"):
                    compare(a, b)
        with tempfile.TemporaryDirectory() as temp:
            a, b = self.pair(temp)
            self.edit_rows(b, file="train.jsonl", root=True, callback=lambda rows: rows.pop())
            with self.assertRaisesRegex(ValueError, "fresh steps"):
                compare(a, b)

    def test_execution_settings_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = self.pair(temp)
            self.edit(b.parent.parent/"manifest.json", lambda d: d["protocol_info"]["execution_settings"].update(cuda_launch_blocking=None))
            with self.assertRaisesRegex(ValueError, "protocol_info"):
                compare(a, b)

    def test_grid_difference_magnitude_is_reported_not_hidden(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = self.pair(temp)
            def change(rows):
                rows[0]["confusion"]["grid_probq"][0][0] -= 1
                rows[0]["confusion"]["grid_probq"][0][1] += 1
            self.edit_rows(b, callback=change)
            def aggregate(d):
                d["metrics"]["grid_probq"]["confusion"][0][0] -= 1
                d["metrics"]["grid_probq"]["confusion"][0][1] += 1
            self.edit(b, aggregate)
            result, _ = compare(a, b)
            self.assertFalse(result["grid_confusion_identical"])
            self.assertEqual(result["grid_execution_difference"]["max_absolute_confusion_entry"], 1)
            self.assertEqual(result["grid_execution_difference"]["sum_absolute_confusion_entries"], 2)
            self.assertLess(result["grid_execution_difference"]["miou4_difference_pp"], 0)
            self.assertIn("grid_comparison_note", result)

    def test_per_image_gt_count_disagreement_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = self.pair(temp)
            def change(rows):
                for arm in rows[0]["confusion"]:
                    rows[0]["confusion"][arm][0][0] += 1
            self.edit_rows(b, callback=change)
            def aggregate(d):
                for arm in d["metrics"]:
                    d["metrics"][arm]["confusion"][0][0] += 1
            self.edit(b, aggregate)
            with self.assertRaisesRegex(ValueError, "ground-truth|GT"):
                compare(a, b)


if __name__ == "__main__":
    unittest.main()
