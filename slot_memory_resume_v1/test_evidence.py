"""Stdlib evidence contracts using explicitly synthetic full-count fixtures."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import evidence


ARMS = ["e", "f", "b", "d", "r", "l"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _image_objectives():
    return {
        "original_loss": .3, "nll": .2, "lovasz": .1,
        "class_nll_contribution": [.05] * 4,
        "class_lovasz_contribution": [.025] * 4,
        "class_valid_counts": [1] * 4, "class_present_image_counts": [1] * 4,
        "valid_count": 4,
    }


def _repeated_mean(value, count):
    total = 0.0
    for _ in range(count):
        total += value
    return total / count


def _metric(count):
    objectives = _image_objectives()
    return {
        "class_names": list(evidence.CLASSES),
        "confusion": [[count if row == column else 0 for column in range(4)] for row in range(4)],
        "target_pixels": [count] * 4, "class_valid_counts": [count] * 4,
        "class_present_image_counts": [count] * 4,
        "iou": [1.] * 4, "precision": [1.] * 4, "recall": [1.] * 4,
        "foreground_miou": 1., "valid_miou": 1.,
        **{"mean_image_" + term: _repeated_mean(objectives[term], count)
           for term in evidence.SCALAR_TERMS},
        **{"mean_image_" + term: [_repeated_mean(value, count) for value in objectives[term]]
           for term in evidence.VECTOR_TERMS},
    }


def _paired(count, pairs):
    counts = [[[count if g == b == a else 0 for a in range(4)]
               for b in range(4)] for g in range(4)]
    return {key: {"axes": ["GT", before, after], "counts": copy.deepcopy(counts)}
            for key, (before, after) in pairs.items()}


def fixture_event(index, count):
    return "event0" if index < (count + 1) // 2 else "event1"


def make_report(step, split, names, arms=ARMS, primary=False):
    """Synthetic metrics fixture for integration tests; never runtime evidence."""
    names, arms = list(names), list(arms)
    paths = [evidence.BASELINE] + arms
    pairs = evidence._pairs(arms)
    event_counts = {}
    for index in range(len(names)):
        event = fixture_event(index, len(names))
        event_counts[event] = event_counts.get(event, 0) + 1
    interactions = {}
    if set("dbfe") <= set(arms):
        interactions["(d-b)-(f-e)"] = 0.
    if set("dcba") <= set(arms):
        interactions["(d-c)-(b-a)"] = 0.
    return {
        "format": evidence.FORMAT, "step": step, "split": split, "arms": arms,
        "class_names": list(evidence.CLASSES),
        "phase": "evaluation_complete_integrity_pending",
        "execution_complete": True, "complete": False, "final_integrity_pending": True,
        "full_validation_evidence": False, "whole_val450_processed": split == "val",
        "primary_endpoint": False, "primary_endpoint_requested": primary,
        "smoke": False, "training": False, "test_accessed": False,
        "evaluation_loader_train": False, "prediction_arrays_saved": False,
        "reference_formula_verified_on_first_image": True, "query_GT_not_support_input": True,
        "inherited_support_exclusion_passed": True, "paired_counts_reopened": True,
        "objective_totals_reopened": True, "per_event_totals_reopened": True,
        "head_state_unchanged": True, "module_modes_restored": True, "rng_restored": True,
        "image_count": len(names), "valid_pixel_count": 4 * len(names),
        "names_sha256": evidence.names_hash(sorted(names)),
        "order_sha256": evidence.names_hash(names),
        "requested_order_sha256": evidence.names_hash(names),
        "trained_objective": {
            path: "frozen" if path == evidence.BASELINE else "original_nll_plus_lovasz"
            for path in paths},
        "metrics": {path: _metric(len(names)) for path in paths},
        "per_event": {
            event: {path: _metric(count) for path in paths}
            for event, count in event_counts.items()},
        "per_event_image_count": event_counts,
        "transitions": _paired(len(names), pairs),
        "per_event_transitions": {
            event: _paired(count, pairs) for event, count in event_counts.items()},
        "comparisons_pp": {key: 0. for key in pairs},
        "per_event_comparisons_pp": {
            event: {key: 0. for key in pairs} for event in event_counts},
        "interactions_pp": interactions,
        "per_event_interactions_pp": {
            event: dict(interactions) for event in event_counts},
        "head_hashes": {arm: hashlib.sha256(arm.encode()).hexdigest() for arm in arms},
        "journal_sha256": "0" * 64,
        "mechanism_audit_image_count": 0, "mechanism_audits_reopened": False,
        "mechanism_audit_evidence": None,
    }


def write_report(folder, report, names):
    """Write synthetic results with the real per-image objective journal schema."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    journal = folder / "images.jsonl"
    with journal.open("w", encoding="utf-8") as stream:
        for index, name in enumerate(names):
            row = dict(
                index=index, image=name, event=fixture_event(index, len(names)), valid_pixels=4,
                confusion={path: [[int(r == c) for c in range(4)] for r in range(4)]
                           for path in report["metrics"]},
                objectives={path: _image_objectives() for path in report["metrics"]},
                paired_counts={key: pair["counts"]
                               for key, pair in _paired(1, evidence._pairs(report["arms"])).items()})
            stream.write(json.dumps(row) + "\n")
    report["journal_sha256"] = sha256(journal)
    path = folder / "result.json"
    write_json(path, report)
    path.with_name("result.json.sha256").write_text(sha256(path) + "  result.json\n")
    return path, report


def rehash_report(path, report):
    """Re-sign deliberately edited test evidence; checks must exceed checksums."""
    report["journal_sha256"] = sha256(path.with_name("images.jsonl"))
    write_json(path, report)
    path.with_name("result.json.sha256").write_text(sha256(path) + "  result.json\n")


class EvidenceContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "new_resume"
        self.root.mkdir()
        self.val_names = [f"val_{index:04d}" for index in range(450)]
        self.fixed_names = ["train_fixed_0", "train_fixed_1"]
        self.arms = list(ARMS)
        self.expected = evidence.expected_evaluations(
            1000, 2500, self.val_names, self.fixed_names)

    def validate(self, report, **kwargs):
        args = dict(step=1000, split="val", names=self.val_names, arms=self.arms, primary=False)
        args.update(kwargs)
        return evidence.validate_report(report, **args)

    def make_results(self, expected=None):
        result = []
        for step, split, names, primary in self.expected if expected is None else expected:
            report = make_report(step, split, names, self.arms, primary)
            result.append(write_report(self.root / f"{split}_{step}", report, names))
        return result

    def promote(self, results, **kwargs):
        args = dict(arms=self.arms, val_names=self.val_names, fixed_names=self.fixed_names,
                    endpoint=2500, boundary_step=1000, write_json=write_json,
                    sha256=sha256, output_root=self.root)
        args.update(kwargs)
        return evidence.promote_reports(results, **args)

    def test_expected_schedule_includes_full_boundary_and_endpoint_diagnostics(self):
        self.assertEqual([(step, split, primary) for step, split, _, primary in self.expected], [
            (1000, "val", False), (1000, "fixed_train", False), (1500, "val", False),
            (2000, "val", False), (2500, "val", True), (2500, "fixed_train", False)])
        odd = evidence.expected_evaluations(1000, 2100, self.val_names, self.fixed_names)
        self.assertEqual([step for step, split, _, _ in odd if split == "val"],
                         [1000, 1500, 2000, 2100])

    def test_validate_is_read_only_and_accepts_actual_schema(self):
        report = make_report(1000, "val", self.val_names)
        original = copy.deepcopy(report)
        self.assertIsNone(self.validate(report))
        self.assertEqual(report, original)

    def test_nll_cross_reduction_gap_is_diagnostic_and_values_are_unchanged(self):
        report = make_report(1000, "val", self.val_names)
        # A legal FP32 CUDA32-lane NLL reduction differs from masked Tensor.sum.
        scalar, class_sum = .10539745539426804, .10536051541566849
        for metrics in [report["metrics"], *report["per_event"].values()]:
            for metric in metrics.values():
                metric["mean_image_nll"] = scalar
                metric["mean_image_class_nll_contribution"] = [class_sum / 4] * 4
                metric["mean_image_original_loss"] = scalar + metric["mean_image_lovasz"]
        original = copy.deepcopy(report)
        self.validate(report)
        diagnostic = evidence.nll_reduction_diagnostics(report)
        self.assertEqual(report, original)
        self.assertEqual(len(diagnostic["records"]), len(report["metrics"]) * 3)
        self.assertGreater(diagnostic["max_absolute_relative_gap"], 3e-4)
        self.assertEqual(diagnostic["max_absolute_gap"], abs(class_sum - scalar))
        self.assertEqual({row["scope"] for row in diagnostic["records"]}, {"global", "event"})
        self.assertTrue(all(row["signed_gap"] == class_sum - scalar for row in diagnostic["records"]))
        report["metrics"]["d"]["mean_image_class_lovasz_contribution"][0] += .01
        with self.assertRaisesRegex(ValueError, "class lovasz sum.*actual=.*expected="):
            self.validate(report)

    def test_saved_validation_replays_every_objective_without_mutation(self):
        report = make_report(1000, "val", self.val_names)
        path, report = write_report(self.root / "val_1000", report, self.val_names)
        original = copy.deepcopy(report)
        files = {p: p.read_bytes() for p in path.parent.iterdir()}
        self.validate(report)
        self.assertEqual(evidence.validate_saved_report(
            path, report, self.val_names, sha256, self.root), path)
        self.assertEqual(report, original)
        self.assertEqual({p: p.read_bytes() for p in path.parent.iterdir()}, files)

    def test_rehashed_global_objective_mutations_cannot_bypass_journal_replay(self):
        report = make_report(1000, "val", self.val_names)
        path, report = write_report(self.root / "val_1000", report, self.val_names)
        original = copy.deepcopy(report)
        for field in ("mean_image_nll", "mean_image_class_nll_contribution",
                      "mean_image_class_lovasz_contribution", "class_present_image_counts"):
            changed = copy.deepcopy(original)
            # Alter every scope consistently enough that cross-event identities
            # cannot detect the NLL mutation; persisted per-image data still can.
            for metrics in [changed["metrics"], *changed["per_event"].values()]:
                if field == "mean_image_nll":
                    metrics["d"][field] += .01
                    metrics["d"]["mean_image_original_loss"] += .01
                elif field == "class_present_image_counts":
                    metrics["d"][field][0] -= 1
                else:
                    metrics["d"][field][0] += .01
                    metrics["d"][field][1] -= .01
            rehash_report(path, changed)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "objective replay"):
                evidence.validate_saved_report(path, changed, self.val_names, sha256, self.root)

    def test_rehashed_per_event_objective_mutation_is_detected(self):
        report = make_report(1000, "val", self.val_names)
        path, report = write_report(self.root / "val_1000", report, self.val_names)
        for event, delta in (("event0", .01), ("event1", -.01)):
            report["per_event"][event]["d"]["mean_image_class_nll_contribution"][0] += delta
        rehash_report(path, report)
        self.validate(report)
        with self.assertRaisesRegex(ValueError, "event/event0/d/mean_image_class_nll"):
            evidence.validate_saved_report(path, report, self.val_names, sha256, self.root)

    def test_rehashed_missing_or_corrupt_raw_objectives_are_rejected(self):
        report = make_report(1000, "fixed_train", self.fixed_names)
        path, report = write_report(self.root / "fixed_1000", report, self.fixed_names)
        journal = path.with_name("images.jsonl")
        original_rows = [json.loads(line) for line in journal.read_text().splitlines()]
        edits = (
            lambda row: row.pop("objectives"),
            lambda row: row["objectives"].pop("d"),
            lambda row: row["objectives"]["d"].pop("nll"),
            lambda row: row["objectives"]["d"]["class_valid_counts"].__setitem__(0, 0),
            lambda row: row["objectives"]["d"]["class_present_image_counts"].__setitem__(0, 0),
            lambda row: row["objectives"]["d"]["class_nll_contribution"].__setitem__(0, float("nan")),
            lambda row: row["objectives"]["d"]["class_nll_contribution"].__setitem__(0, -.01),
            lambda row: row["objectives"]["d"]["class_nll_contribution"].__setitem__(0, .06),
        )
        for index, edit in enumerate(edits):
            rows = copy.deepcopy(original_rows)
            edit(rows[0])
            journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
            rehash_report(path, report)
            with self.subTest(edit=index), self.assertRaises(ValueError):
                evidence.validate_saved_report(path, report, self.fixed_names, sha256, self.root)

    def test_raw_all_ignore_and_absent_class_objectives_preserve_zero_contract(self):
        values = _image_objectives()
        values.update({term: 0. for term in evidence.SCALAR_TERMS})
        values.update({term: [0.] * 4 for term in evidence.VECTOR_TERMS})
        values.update({term: [0] * 4 for term in evidence.VECTOR_COUNTS})
        values["valid_count"] = 0
        row = {"valid_pixels": 0, "objectives": {"d": values},
               "confusion": {"d": [[0] * 4 for _ in range(4)]}}
        self.assertEqual(evidence._checked_objectives(row, ["d"]), {"d": values})
        empty_metric = {"mean_image_nll": 0., "mean_image_class_nll_contribution": [0.] * 4}
        diagnostic = evidence.nll_reduction_diagnostics(
            {"metrics": {"d": empty_metric}, "per_event": {}})
        self.assertEqual(diagnostic["records"][0]["relative_gap"], 0.)
        self.assertEqual(diagnostic["max_absolute_relative_gap"], 0.)
        values["nll"] = values["original_loss"] = .1
        with self.assertRaisesRegex(ValueError, "All-ignore"):
            evidence._checked_objectives(row, ["d"])
        values["nll"] = values["original_loss"] = 0.
        values["class_nll_contribution"][0] = .1
        with self.assertRaisesRegex(ValueError, "Absent-class"):
            evidence._checked_objectives(row, ["d"])

    def test_smoke_incomplete_promoted_or_false_integrity_flags_are_rejected(self):
        changes = [
            ("smoke", True), ("complete", True), ("execution_complete", False),
            ("full_validation_evidence", True), ("final_integrity_pending", False),
            ("phase", "complete"), ("primary_endpoint", True),
            ("whole_val450_processed", False), ("test_accessed", True),
            ("evaluation_loader_train", True), ("rng_restored", False),
            ("paired_counts_reopened", False), ("per_event_totals_reopened", False),
        ]
        for key, value in changes:
            report = make_report(1000, "val", self.val_names)
            report[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.validate(report)

    def test_exact_names_order_counts_and_arm_identities_are_required(self):
        changes = [
            ("image_count", 449), ("valid_pixel_count", 1799),
            ("names_sha256", "f" * 64), ("order_sha256", "f" * 64),
            ("requested_order_sha256", "f" * 64), ("step", 500),
            ("arms", list(reversed(self.arms))),
        ]
        for key, value in changes:
            report = make_report(1000, "val", self.val_names)
            report[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.validate(report)
        with self.assertRaisesRegex(ValueError, "Val450"):
            self.validate(make_report(1000, "val", self.val_names[:2]), names=self.val_names[:2])

    def test_metric_and_paired_marginal_or_event_changes_are_rejected(self):
        edits = (
            lambda report: report["metrics"].pop("grid_probq"),
            lambda report: report["metrics"]["d"]["confusion"][0].__setitem__(0, 449),
            lambda report: report["metrics"]["d"].__setitem__("foreground_miou", .99),
            lambda report: report["metrics"]["d"].__setitem__("mean_image_original_loss", .4),
            lambda report: report["metrics"]["d"]["class_present_image_counts"].__setitem__(0, 451),
            lambda report: report["transitions"]["d_minus_b"]["counts"][0][0].__setitem__(0, 449),
            lambda report: report["per_event_image_count"].__setitem__("event0", 224),
            lambda report: report["comparisons_pp"].__setitem__("d_minus_b", 1.),
            lambda report: report["interactions_pp"].__setitem__("(d-b)-(f-e)", 1.),
            lambda report: report["per_event"]["event0"]["d"].__setitem__("mean_image_nll", float("nan")),
        )
        for index, edit in enumerate(edits):
            report = make_report(1000, "val", self.val_names)
            edit(report)
            with self.subTest(edit=index), self.assertRaises(ValueError):
                self.validate(report)

    def test_only_endpoint_val_primary_and_all_checks_precede_any_publication(self):
        results = self.make_results()
        original_old = self.root.parent / "old_failed_run_initial_val_result.json"
        original_old.write_text('{"step":0,"complete":false}\n')
        old_bytes = original_old.read_bytes()
        self.promote(results)
        for path, report in results:
            saved = json.loads(path.read_text())
            self.assertEqual(saved, report)
            self.assertTrue(report["complete"])
            self.assertFalse(report["final_integrity_pending"])
            self.assertEqual(report["phase"], "complete")
            self.assertEqual(report["full_validation_evidence"], report["split"] == "val")
            self.assertEqual(report["primary_endpoint"], report["split"] == "val" and report["step"] == 2500)
            self.assertEqual(report["resume_boundary_step"], 1000)
            self.assertEqual(path.with_name("result.json.sha256").read_text().split(),
                             [sha256(path), "result.json"])
        self.assertEqual(original_old.read_bytes(), old_bytes)
        with self.assertRaises(ValueError):
            self.promote(results)

    def test_missing_schedule_report_does_not_promote_any_remaining_report(self):
        results = self.make_results()
        originals = {path: path.read_bytes() for path, _ in results}
        missing = [item for item in results if item[1]["step"] != 1500]
        with self.assertRaisesRegex(ValueError, "schedule is incomplete"):
            self.promote(missing)
        for path, report in results:
            self.assertEqual(path.read_bytes(), originals[path])
            self.assertFalse(report["complete"])

    def test_last_invalid_report_prevents_first_promotion(self):
        results = self.make_results()
        originals = {path: path.read_bytes() for path, _ in results}
        results[-1][1]["rng_restored"] = False
        with self.assertRaises(ValueError):
            self.promote(results)
        for path, _ in results:
            self.assertEqual(path.read_bytes(), originals[path])
        self.assertFalse(results[0][1]["complete"])

    def test_historical_zero_or_500_reports_are_never_promoted(self):
        results = self.make_results()
        for step in (0, 500):
            report = make_report(step, "val", self.val_names)
            path = self.root.parent / f"historical_{step}" / "result.json"
            path.parent.mkdir()
            path.write_text(json.dumps(report))
            before = path.read_bytes()
            with self.subTest(step=step), self.assertRaisesRegex(ValueError, "historical"):
                self.promote(results + [(path, report)])
            self.assertEqual(before, path.read_bytes())

    def test_outside_output_and_symlinked_paths_are_rejected_without_rewrite(self):
        results = self.make_results()
        path, report = results[0]
        old_folder = self.root.parent / "old_failed_run"
        old_folder.mkdir()
        for file in path.parent.iterdir():
            (old_folder / file.name).write_bytes(file.read_bytes())
        outside = old_folder / "result.json"
        before = outside.read_bytes()
        with self.assertRaisesRegex(ValueError, "outside"):
            self.promote([(outside, report)] + results[1:])
        self.assertEqual(before, outside.read_bytes())
        alias = self.root / "alias"
        alias.symlink_to(path.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.promote([(alias / "result.json", report)] + results[1:])

    def test_persisted_report_checksum_and_journal_mutations_are_rejected(self):
        results = self.make_results()
        path, report = results[-1]
        journal = path.with_name("images.jsonl")
        sidecar = path.with_name("result.json.sha256")
        originals = {p: p.read_bytes() for p in (path, journal, sidecar)}
        for changed in (path, journal, sidecar):
            if changed == sidecar:
                changed.write_text("0" * 64 + "  result.json\n")
            else:
                with changed.open("ab") as stream:
                    stream.write(b" ")
            with self.subTest(file=changed.name), self.assertRaises((ValueError, json.JSONDecodeError)):
                self.promote(results)
            self.assertFalse(results[0][1]["complete"])
            changed.write_bytes(originals[changed])

    def test_journal_identity_recheck_detects_rehashed_mutation(self):
        results = self.make_results()
        path, report = results[-1]
        journal = path.with_name("images.jsonl")
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        rows[0]["image"] = self.val_names[0]
        journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
        report["journal_sha256"] = sha256(journal)
        write_json(path, report)
        path.with_name("result.json.sha256").write_text(sha256(path) + "  result.json\n")
        with self.assertRaisesRegex(ValueError, "identity/order"):
            self.promote(results)
        self.assertFalse(results[0][1]["complete"])

    def test_explicit_schedule_can_use_driver_planned_intermediate_steps(self):
        expected = [(1000, "val", self.val_names, False),
                    (1000, "fixed_train", self.fixed_names, False),
                    (1002, "val", self.val_names, False),
                    (1004, "val", self.val_names, True),
                    (1004, "fixed_train", self.fixed_names, False)]
        results = self.make_results(expected)
        self.promote(results, endpoint=1004, expected=expected)
        self.assertEqual(sum(report["primary_endpoint"] for _, report in results), 1)

    def test_endpoint_checkpoint_can_recover_only_endpoint_evaluations(self):
        expected = evidence.expected_evaluations(2500, 2500, self.val_names, self.fixed_names)
        self.assertEqual([(step, split, primary) for step, split, _, primary in expected],
                         [(2500, "val", True), (2500, "fixed_train", False)])
        for explicit in (False, True):
            with self.subTest(explicit_schedule=explicit):
                self.root = Path(self.temp.name) / f"endpoint_recovery_{explicit}"
                self.root.mkdir()
                results = self.make_results(expected)
                kwargs = {"expected": expected} if explicit else {}
                self.promote(results, boundary_step=2500, endpoint=2500, **kwargs)
                self.assertTrue(all(report["complete"] for _, report in results))
                self.assertEqual([report["primary_endpoint"] for _, report in results], [True, False])
                self.assertTrue(all(report["resume_boundary_step"] == 2500 for _, report in results))
        with self.assertRaisesRegex(ValueError, "precede"):
            evidence.expected_evaluations(2500, 2000, self.val_names, self.fixed_names)


if __name__ == "__main__":
    unittest.main()
