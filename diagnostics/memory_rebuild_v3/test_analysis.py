"""Standard-library, synthetic saved-record tests for standalone detailed report."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest import mock

import ANALYZE_MEMORY_V3_RESULTS as analysis
from test_summary import SummaryTests


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.fixture_case = SummaryTests()
        self.fixture_case.setUp()
        self.addCleanup(self.fixture_case.doCleanups)
        self.root = self.fixture_case.root

    def fixture(self):
        state, bank, manifest, protocol, diagnose, result = self.fixture_case.fixture()
        for arm in ("a1_pixel", "direct_region", "grid_probq", *("labelvote_" + arm for arm in analysis.ARMS)):
            row = result["metrics"].setdefault(arm, dict(valid_miou=.1))
            row["iou"] = [.8] + [row["valid_miou"]] * 3
            row["support"] = [100, 40, 30, 20]
        for event, arms in result["per_event"].items():
            for row in arms.values():
                row.update(iou=[.8] + [row["valid_miou"]] * 3, support=[10, 4, 3, 2])
        for arm in analysis.ARMS:
            result["posthoc_weighted_label_evidence"][arm] = dict(
                all=dict(queries=10, supported_queries=8, soft_compatibility_sum=4,
                         valid_mass=15, supported_valid_mass=15, mass_weighted_compatibility_sum=6.2),
                foreground_pure=dict(queries=2, supported_queries=1, soft_compatibility_sum=.7,
                                     valid_mass=6, supported_valid_mass=6, mass_weighted_compatibility_sum=3),
                foreground_mixed=dict(queries=3, supported_queries=2, soft_compatibility_sum=1.1,
                                      valid_mass=4, supported_valid_mass=4, mass_weighted_compatibility_sum=1),
                background_diluted=dict(queries=4, supported_queries=4, soft_compatibility_sum=2,
                                        valid_mass=4, supported_valid_mass=4, mass_weighted_compatibility_sum=2),
                mostly_background=dict(queries=1, supported_queries=1, soft_compatibility_sum=.2,
                                       valid_mass=1, supported_valid_mass=1, mass_weighted_compatibility_sum=.2))
            for part, counts in {"all": (2, 2, 5, 2), "foreground_pure": (1, 1, 0, 0),
                                 "foreground_mixed": (0, 0, 2, 1), "background_diluted": (1, 1, 3, 1),
                                 "mostly_background": (0, 0, 0, 0)}.items():
                result["posthoc_weighted_label_evidence"][arm][part].update(dict(zip(
                    ("intact_queries", "intact_vote_damaged_or_destroyed", "damaged_or_destroyed_queries",
                     "damaged_or_destroyed_vote_intact"), counts)))
        inventory = result["bank_label_inventory"]
        inventory["all"]["source_image_count"] = inventory["eligible"]["source_image_count"] = 2
        inventory["eligible"].update(soft_mass=[4, 2, 1, 1], valid_label_mass_by_class=[40, 20, 15, 25])
        inventory["by_event"] = dict(
            alpha=dict(all=dict(support_count=3, dominant_counts=[1, 1, 1, 0]),
                       eligible=dict(support_count=2, dominant_counts=[0, 1, 1, 0])),
            beta=dict(all=dict(support_count=7, dominant_counts=[1, 3, 2, 1]),
                      eligible=dict(support_count=6, dominant_counts=[1, 2, 2, 1])))
        result["paired_neighbor_overlap"] = {k: dict(query_count=10, both_supported_query_count=8,
            neighbor_jaccard_sum=2, neighbor_overlap_fraction_sum=4, top1_agreement_count=4) for k in ("top8", "top64")}
        result["query_eligibility"] = dict(total=12, active=10, inactive_both_keys=2, unlabeled_posthoc_excluded=0)
        self.rebind(bank, manifest, protocol, diagnose, result)
        return state, bank, manifest, protocol, diagnose, result

    def rebind(self, bank, manifest, protocol, diagnose, result):
        sha = self.fixture_case.write("keys/protocol.json", protocol)
        for record in (bank, manifest, diagnose, result):
            record["rebuild_protocol_sha256"] = sha
        self.fixture_case.write("keys/summary.json", bank)
        result["paired_manifest_sha256"] = self.fixture_case.write("keys/manifest.json", manifest)
        self.fixture_case.write("diagnose/summary.json", diagnose)
        self.fixture_case.write("diagnose/result.json", result)

    def text(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            analysis.report(analysis.load_results(self.root))
        return output.getvalue()

    def test_full_golden_retrieval_classes_partitions_overlap_and_event(self):
        self.fixture()
        text = self.text()
        self.assertIn("| t128_top8 | 10 | 8 | 80.0000 | 0.5000 | 0.7500 |", text)
        self.assertIn("| t128_top8 | Damaged | 3.0000 | 0.1000 | 50.0000 |", text)
        self.assertIn("| t128_top8 | foreground_pure | 2 | 20.0000 | 1 | 50.0000 | 0.7000 | 0.5000 |", text)
        self.assertIn("| top64 | 80.0000 | 0.2500 | 0.5000 | 50.0000 |", text)
        self.assertIn("| alpha | mIoU | N/A | 20.0000 | +3.0000 | -2.0000 |", text)
        self.assertIn("| alpha | 2 | 6 | 0 | 1 | 1 | 0 |", text)
        self.assertIn('"inactive_both_keys": 2', text)
        self.assertIn("不是像素 Recall", text)
        self.assertIn("未重新校验大 bank", text)

    def test_seven_arm_metrics_preserve_stored_fixed_mean(self):
        *_, result = self.fixture()
        result["metrics"]["a1_pixel"]["iou"] = [.8, None, .7, .1]
        result["metrics"]["a1_pixel"]["valid_miou"] = .2
        self.fixture_case.write("diagnose/result.json", result)
        text = self.text()
        self.assertIn("| a1_pixel | 20.0000 | 80.0000 | N/A | 70.0000 | 10.0000 | -5.0000 |", text)
        for arm in analysis.ARMS:
            self.assertIn("| labelvote_" + arm + " |", text)
        self.assertIn("| direct_region |", text)

    def test_transition_saved_counts_verified_and_signed_values(self):
        self.fixture()
        text = self.text()
        self.assertIn("| t128_top64 | Intact | 1 | 2 | 50.0000 | 25.0000 | -1 | +1 |", text)
        self.assertIn("| t128_top64 | Damaged | 3 | 1 | 60.0000 | 20.0000 | +2 | +0 |", text)
        self.assertIn("joint 校核： 2 /4", text)
        self.assertIn("| t128_top8 | Damaged | N/A", text)

    def test_missing_statistics_and_null_objects_are_na(self):
        *_, result = self.fixture()
        for key in ("metrics", "posthoc_retrieval", "posthoc_weighted_label_evidence", "transitions", "per_event", "paired_neighbor_overlap"):
            result[key] = None
        self.fixture_case.write("diagnose/result.json", result)
        text = self.text()
        self.assertIn("| a1_pixel | N/A", text)
        self.assertIn("| t128_top64 | Damaged | N/A", text)
        self.assertIn("| top64 | N/A", text)

    def test_missing_by_event_is_na_not_zero_same_event(self):
        _, bank, manifest, protocol, diagnose, result = self.fixture()
        del result["bank_label_inventory"]["by_event"]
        self.rebind(bank, manifest, protocol, diagnose, result)
        self.assertIn("| alpha | N/A | N/A |", self.text())

    def test_event_gt_absence_not_inferred_from_nonnull_iou(self):
        *_, result = self.fixture()
        result["per_event"]["beta"]["grid_probq"]["support"][2] = 0
        result["per_event"]["beta"]["grid_probq"]["iou"][2] = 0
        self.fixture_case.write("diagnose/result.json", result)
        self.assertIn("| beta | Damaged | 0 | 0.0000 |", self.text())

    def test_boolean_pseudo_numeric_and_zero_denominator_do_not_create_statistics(self):
        self.assertIsNone(analysis.number(True))
        self.assertIsNone(analysis.ratio(1, False))
        self.assertEqual(analysis.fmt(True), "N/A")
        self.assertEqual(analysis.count(True), "N/A")
        self.assertIsNone(analysis.value({"negative": -1}, "negative"))
        self.assertIsNone(analysis.ratio(-1, 2))
        self.assertIsNone(analysis.vector_total({"bad": [-1, 2, 3, 4]}, "bad"))
        *_, result = self.fixture()
        row = result["posthoc_retrieval"]["t128_top8"]
        row["supported_query_count"] = 0
        row["query_class_mass"] = [0, 0, 0, 0]
        self.fixture_case.write("diagnose/result.json", result)
        self.assertIn("| t128_top8 | Damaged | 0.0000 | N/A | N/A |", self.text())

    def test_false_formal_checks_scope_or_error_rejected(self):
        for field in ("labels", "training", "error"):
            with self.subTest(field=field):
                state, _, _, _, diagnose, result = self.fixture()
                if field == "labels":
                    diagnose["validation"][field] = result["validation"][field] = False
                    self.fixture_case.write("diagnose/summary.json", diagnose)
                    self.fixture_case.write("diagnose/result.json", result)
                elif field == "training":
                    result[field] = True
                    self.fixture_case.write("diagnose/result.json", result)
                else:
                    state[field] = "synthetic error"
                    self.fixture_case.write("pipeline_state.json", state)
                with self.assertRaises(ValueError):
                    analysis.load_results(self.root)

    def test_all_nine_formal_checks_required(self):
        for field in analysis.CHECKS:
            with self.subTest(field=field):
                *_, diagnose, result = self.fixture()
                diagnose["validation"][field] = result["validation"][field] = False
                self.fixture_case.write("diagnose/summary.json", diagnose)
                self.fixture_case.write("diagnose/result.json", result)
                with self.assertRaisesRegex(ValueError, "九项检查"):
                    analysis.load_results(self.root)

    def test_changed_protocol_bytes_or_source_binding_refused(self):
        _, _, _, protocol, _, _ = self.fixture()
        protocol["unbound_extra"] = 1
        self.fixture_case.write("keys/protocol.json", protocol)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            analysis.load_results(self.root)
        *_, result = self.fixture()
        result["source_sha256"] = "e" * 64
        self.fixture_case.write("diagnose/result.json", result)
        with self.assertRaisesRegex(ValueError, "来源绑定"):
            analysis.load_results(self.root)

    def test_joint_tamper_false_rate_or_boolean_count_refused(self):
        for field, changed in (("rescue", True), ("rescue_rate", .9), ("net_tp", 999)):
            *_, result = self.fixture()
            result["transitions"]["grid_probq__labelvote_t128_top64"]["per_class"]["Damaged"][field] = changed
            self.fixture_case.write("diagnose/result.json", result)
            with self.assertRaisesRegex(ValueError, "joint 不一致"):
                self.text()

    def test_read_only_only_six_jsons_no_optional_heavy_payload_reads(self):
        self.fixture()
        (self.root / "keys/paired_bank.pt").write_bytes(b"must not load")
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        real_open, opened = Path.open, []
        def read_only_json(path, mode="r", *args, **kwargs):
            self.assertEqual(mode, "rb")
            self.assertEqual(path.suffix, ".json")
            opened.append(path)
            return real_open(path, mode, *args, **kwargs)
        with mock.patch.object(Path, "open", read_only_json):
            self.text()
        self.assertEqual(len(opened), 6)
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_cli_success_and_missing_record_failure(self):
        self.fixture()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(analysis.main(["--root", str(self.root)]), 0)
        (self.root / "keys/manifest.json").unlink()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(analysis.main(["--root", str(self.root)]), 1)

    def test_inventory_quality_uses_separate_four_class_denominators(self):
        self.fixture()
        text = self.text()
        self.assertIn("| Background | 12.5000 | 50.0000 | 40.0000 |", text)
        self.assertIn("| Damaged | 37.5000 | 12.5000 | 15.0000 |", text)
        self.assertIn("保留背景", text)

    def test_region_posterior_misvote_ratios_not_pixel_rescue(self):
        self.fixture()
        text = self.text()
        self.assertIn("| t128_top8 | all | 10 | 100.0000 | 8 | 80.0000 | 0.5000 | 0.4133 | 100.0000 | 40.0000 |", text)
        self.assertIn("不是像素 rescue", text)


if __name__ == "__main__":
    unittest.main()
