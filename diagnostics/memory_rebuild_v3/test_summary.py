"""Standard-library synthetic saved-record tests; no BRIGHT models/data imports."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import SUMMARIZE_MEMORY_REBUILD_V3 as summary


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.e0, self.bank_sha, self.member_sha = "a" * 64, "b" * 64, "c" * 64, "d" * 64

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def read(self, name):
        return json.loads((self.root / name).read_text())

    def state(self, ready=False):
        return dict(format=summary.PIPE_FORMAT, source_sha256=self.source, protocol_sha256=self.e0,
                    training=False, test_accessed=False, historical_replay_required=False,
                    historical_reproduction_claim=False, complete=ready, full_validation_evidence=ready,
                    smoke_complete=True, current="complete" if ready else "keys", error=None,
                    stages={"keys": {"status": "complete" if ready else "running"},
                            "diagnose": {"status": "complete" if ready else "pending"},
                            "report": {"status": "complete" if ready else "pending"}})

    def fixture(self):
        state = self.state(ready=True)
        self.write("pipeline_state.json", state)
        scope = dict(training=False, test_accessed=False, train_only=True, smoke=False,
                     historical_replay_required=False, historical_fp32_reproduction_claim=False)
        inventory = dict(class_names=["background", "intact", "damaged", "destroyed"],
                         all=dict(support_count=10, dominant_counts=[2, 4, 3, 1]),
                         eligible=dict(support_count=8, dominant_counts=[1, 3, 3, 1]))
        protocol = dict(format=summary.PROTOCOL_FORMAT, **scope, source_sha256=self.source,
                        protocol_sha256=self.e0, source_e0_sha256=self.e0,
                        paired_bank_sha256=self.bank_sha, membership_sha256=self.member_sha,
                        support_count=10, storage_dtype="torch.float32", query_dtype="torch.float32",
                        bank_label_inventory=inventory)
        protocol_sha = self.write("keys/protocol.json", protocol)
        bank = dict(format=summary.BANK_FORMAT, **scope, complete=True, full_validation_evidence=False,
                    immutable={"source": True, "data": True}, image_count=7, extraction_forward_count=7,
                    support_count=10, paired_bank_sha256=self.bank_sha, membership_sha256=self.member_sha,
                    rebuild_protocol_sha256=protocol_sha)
        self.write("keys/summary.json", bank)
        manifest = dict(bank, source_sha256=self.source, protocol_sha256=self.e0, source_e0_sha256=self.e0,
                        fresh_label_checks_passed=True, same_pass_q_checks_passed=True,
                        saved_fp32_roundtrip_exact=True, pool_ok=True, t128_storage="torch.float32",
                        f256_storage="torch.float32", label_storage="torch.float32")
        manifest_sha = self.write("keys/manifest.json", manifest)
        diagnose = dict(format=summary.DIAG_FORMAT, training=False, test_accessed=False,
                        historical_reproduction_claim=False, labelvote_is_untrained_diagnostic=True,
                        complete=True, full_validation_evidence=True, execution_complete=True,
                        image_count=450, smoke=False, validation={key: True for key in summary.CHECKS},
                        immutable={"source": True}, rebuild_protocol_sha256=protocol_sha)
        self.write("diagnose/summary.json", diagnose)
        retrieval = dict(query_count=10, supported_query_count=8, soft_compatibility_sum=4,
                         query_class_mass=[1, 4, 3, 2], covered_query_class_mass=[1, 4, 1.5, .5],
                         class_soft_compatibility_sum=[.9, 2, .3, .8],
                         quality_sum=[0, 0, 0, 0, 6, 0])
        partitions = dict(foreground_pure=dict(supported_valid_mass=6, mass_weighted_compatibility_sum=3),
                          foreground_mixed=dict(supported_valid_mass=4, mass_weighted_compatibility_sum=1))
        joint = [[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)]
        for gt, old, new, count in ((1, 1, 1, 6), (1, 1, 2, 2), (1, 2, 1, 1), (1, 2, 2, 1),
                                     (2, 2, 2, 4), (2, 2, 1, 1), (2, 1, 2, 3), (2, 1, 1, 2),
                                     (3, 3, 3, 2), (3, 3, 1, 2), (3, 1, 3, 1), (3, 1, 1, 5),
                                     (0, 0, 1, 2), (0, 2, 0, 1), (0, 3, 3, 1)):
            joint[gt][old][new] = count
        classes = dict(Intact=dict(rescue=1, harm=2, rescue_rate=.5, harm_rate=.25, net_tp=-1, delta_fp=1),
                       Damaged=dict(rescue=3, harm=1, rescue_rate=.6, harm_rate=.2, net_tp=2, delta_fp=0),
                       Destroyed=dict(rescue=1, harm=2, rescue_rate=1/6, harm_rate=.5, net_tp=-1, delta_fp=0))
        transitions = {"grid_probq__labelvote_" + arm:
                       dict(joint_gt_reference_candidate=joint, per_class=classes)
                       for arm in summary.FOCUS_ARMS}
        result = dict(diagnose, source_sha256=self.source, source_anchor_protocol_sha256=self.e0,
                      paired_bank_sha256=self.bank_sha, paired_manifest_sha256=manifest_sha,
                      paired_bank_smoke=False, bank_label_inventory=inventory,
                      metrics={"a1_pixel": dict(valid_miou=.20), "grid_probq": dict(valid_miou=.25),
                               **{"labelvote_" + arm: dict(valid_miou=.26) for arm in summary.ARMS}},
                      posthoc_retrieval={arm: retrieval.copy() for arm in summary.ARMS},
                      posthoc_weighted_label_evidence={arm: partitions.copy() for arm in summary.ARMS},
                      transitions=transitions,
                      per_event={"alpha": dict(grid_probq=dict(valid_miou=.2),
                                               labelvote_t128_top64=dict(valid_miou=.23),
                                               labelvote_f256_top64=dict(valid_miou=.18)),
                                 "beta": dict(grid_probq=dict(valid_miou=.3),
                                              labelvote_t128_top64=dict(valid_miou=.25),
                                              labelvote_f256_top64=dict(valid_miou=.34))})
        self.write("diagnose/result.json", result)
        return state, bank, manifest, protocol, diagnose, result

    def test_gold_ratios_and_percentage_points(self):
        self.fixture()
        report = summary.read_summary(self.root)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["missing_fields"], [])
        row = report["retrieval"][0]
        for field, expected in (("support_coverage", .8), ("soft_compatibility", .5),
                                ("damaged_soft_coverage", .5), ("destroyed_soft_coverage", .25),
                                ("weighted_support_purity", .75), ("delta_grid_pp", 1)):
            self.assertAlmostEqual(row[field], expected)
        self.assertEqual(report["foreground"][0]["mass_weighted_compatibility"], .5)
        self.assertEqual(report["foreground"][1]["mass_weighted_compatibility"], .25)
        text = summary.render(report)
        self.assertIn("80.000 | 0.500 | 50.000 | 25.000 | 0.750 | 26.000 | +1.000", text)
        self.assertIn("不能解释为 Reader 收益", text)
        self.assertIn("不是 Recall", text)

    def test_pending_reports_progress_and_never_reads_smoke_scores(self):
        self.write("pipeline_state.json", self.state())
        self.write("keys/summary.json", dict(format=summary.BANK_FORMAT, training=False, test_accessed=False,
                                             complete=False, image_count=251, phase="Train_paired_rebuild"))
        self.write("diagnose_smoke/result.json", {"metrics": {"fake": 999}})
        report = summary.read_summary(self.root)
        text = summary.render(report)
        self.assertEqual(report["status"], "pending")
        self.assertEqual(report["image_count"], 251)
        self.assertIn("keys.log", text)
        self.assertNotIn("999", text)
        self.assertNotIn("| key / TopK", text)

    def test_missing_pending_keys_not_error(self):
        self.write("pipeline_state.json", self.state())
        self.assertEqual(summary.read_summary(self.root)["status"], "pending")

    def test_waits_for_pipeline_report_even_when_val450_ready(self):
        state, *_ = self.fixture()
        state.update(complete=False, full_validation_evidence=False, current="report")
        state["stages"]["report"]["status"] = "running"
        self.write("pipeline_state.json", state)
        report = summary.read_summary(self.root)
        self.assertEqual(report["status"], "pending")
        self.assertEqual(report["image_count"], 450)
        self.assertIn("等待流水线报告", report["phase"])
        self.assertNotIn("| key / TopK", summary.render(report))

    def test_wrong_count_rejects_false_formal_claim(self):
        *_, diagnose, result = self.fixture()
        diagnose["image_count"] = result["image_count"] = 2
        self.write("diagnose/summary.json", diagnose)
        self.write("diagnose/result.json", result)
        with self.assertRaisesRegex(ValueError, "不是完整 Val450"):
            summary.read_summary(self.root)

    def test_nine_gates_all_required(self):
        for gate in summary.CHECKS:
            with self.subTest(gate=gate):
                *_, diagnose, result = self.fixture()
                diagnose["validation"][gate] = result["validation"][gate] = False
                self.write("diagnose/summary.json", diagnose)
                self.write("diagnose/result.json", result)
                with self.assertRaisesRegex(ValueError, "九项验证"):
                    summary.read_summary(self.root)

    def test_smoke_bank_rejected(self):
        *_, result = self.fixture()
        result["paired_bank_smoke"] = True
        self.write("diagnose/result.json", result)
        with self.assertRaisesRegex(ValueError, "smoke 库"):
            summary.read_summary(self.root)

    def test_changed_protocol_bytes_rejected(self):
        *_, protocol, _diagnose, _result = self.fixture()
        protocol["extra"] = 1
        self.write("keys/protocol.json", protocol)
        with self.assertRaisesRegex(ValueError, "新协议 JSON SHA256"):
            summary.read_summary(self.root)

    def test_changed_manifest_bytes_rejected(self):
        _, _, manifest, *_ = self.fixture()
        manifest["extra"] = 1
        self.write("keys/manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "paired_manifest_sha256"):
            summary.read_summary(self.root)

    def test_cross_metadata_source_and_e0_mismatch_rejected(self):
        for field in ("source_sha256", "source_anchor_protocol_sha256", "paired_bank_sha256"):
            with self.subTest(field=field):
                *_, result = self.fixture()
                result[field] = "e" * 64
                self.write("diagnose/result.json", result)
                with self.assertRaisesRegex(ValueError, "来源绑定"):
                    summary.read_summary(self.root)

    def test_metadata_scope_no_training_no_test_historical_claim(self):
        for field in ("training", "test_accessed", "historical_reproduction_claim"):
            with self.subTest(field=field):
                *_, result = self.fixture()
                result[field] = True
                self.write("diagnose/result.json", result)
                with self.assertRaises(ValueError):
                    summary.read_summary(self.root)

    def test_immutable_false_rejected(self):
        *_, result = self.fixture()
        result["immutable"]["source"] = False
        self.write("diagnose/result.json", result)
        with self.assertRaisesRegex(ValueError, "immutable"):
            summary.read_summary(self.root)

    def test_missing_counts_and_posthoc_are_na_never_zero(self):
        *_, result = self.fixture()
        del result["posthoc_retrieval"]["t128_top8"]["query_count"]
        del result["posthoc_retrieval"]["f256_top64"]
        del result["posthoc_weighted_label_evidence"]
        self.write("diagnose/result.json", result)
        report = summary.read_summary(self.root)
        self.assertIsNone(report["retrieval"][0]["support_coverage"])
        self.assertIsNone(report["retrieval"][3]["destroyed_soft_coverage"])
        self.assertIn("t128_top8.query_count", report["missing_fields"])
        self.assertIn("posthoc_weighted_label_evidence", report["missing_fields"])
        self.assertIn("N/A", summary.render(report))

    def test_zero_denominators_are_na_not_misleading_zero(self):
        *_, result = self.fixture()
        row = result["posthoc_retrieval"]["t128_top8"]
        row.update(query_count=0, supported_query_count=0, query_class_mass=[0] * 4,
                   covered_query_class_mass=[0] * 4, soft_compatibility_sum=0, quality_sum=[0] * 6)
        self.write("diagnose/result.json", result)
        row = summary.read_summary(self.root)["retrieval"][0]
        for field in ("support_coverage", "soft_compatibility", "damaged_soft_coverage", "weighted_support_purity"):
            self.assertIsNone(row[field])

    def test_missing_foreground_groups_does_not_hide_overall_bias(self):
        *_, result = self.fixture()
        del result["posthoc_weighted_label_evidence"]["t128_top8"]["foreground_mixed"]
        self.write("diagnose/result.json", result)
        report = summary.read_summary(self.root)
        self.assertIsNone(report["foreground"][1]["mass_weighted_compatibility"])
        self.assertIn("t128_top8.foreground_mixed", report["missing_fields"])

    def test_missing_completed_result_refused(self):
        self.fixture()
        (self.root / "diagnose/result.json").unlink()
        with self.assertRaisesRegex(ValueError, "缺少记录"):
            summary.read_summary(self.root)

    def test_reject_bad_array_and_nonfinite_stat(self):
        for value in ([1, 2], [1, 2, float("nan"), 4]):
            *_, result = self.fixture()
            result["posthoc_retrieval"]["t128_top8"]["query_class_mass"] = value
            self.write("diagnose/result.json", result)
            with self.assertRaises(ValueError):
                summary.read_summary(self.root)

    def test_malformed_json_and_scope_cli_exit_one(self):
        self.write("pipeline_state.json", self.state())
        (self.root / "pipeline_state.json").write_text("{", encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            self.assertEqual(summary.main(["--root", str(self.root)]), 1)
        self.assertIn("格式错误", output.getvalue())

    def test_pending_cli_exit_zero_and_no_writes(self):
        self.write("pipeline_state.json", self.state())
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(summary.main(["--root", str(self.root)]), 0)
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_ready_no_bank_source_model_dataset_reads_and_no_writes(self):
        self.fixture()
        (self.root / "keys/paired_bank.pt").write_bytes(b"must not load")
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        real_open = Path.open
        def read_only_json(path, mode="r", *args, **kwargs):
            self.assertEqual(path.suffix, ".json", "Only saved JSON records may be opened")
            self.assertIn(mode, ("r", "rb"))
            return real_open(path, mode, *args, **kwargs)
        with mock.patch.object(Path, "open", read_only_json):
            report = summary.read_summary(self.root)
        self.assertEqual(report["status"], "ready")
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_pipeline_partial_completion_claim_refused(self):
        state, *_ = self.fixture()
        state["full_validation_evidence"] = False
        self.write("pipeline_state.json", state)
        with self.assertRaisesRegex(ValueError, "完整声明不一致"):
            summary.read_summary(self.root)

    def test_running_partial_stage_json_reports_wait_not_failure(self):
        self.write("pipeline_state.json", self.state())
        (self.root / "keys").mkdir()
        (self.root / "keys/summary.json").write_text('{"format":', encoding="utf-8")
        report = summary.read_summary(self.root)
        self.assertEqual(report["status"], "pending")
        self.assertIsNone(report["error"])
        self.assertIn("正在写入", summary.render(report))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(summary.main(["--root", str(self.root)]), 0)

    def test_completed_partial_stage_json_strictly_refused(self):
        self.fixture()
        (self.root / "keys/summary.json").write_text('{"format":', encoding="utf-8")
        with self.assertRaises(summary.RecordReadError):
            summary.read_summary(self.root)

    def test_missing_inventory_scope_reported_as_na(self):
        _, bank, manifest, protocol, diagnose, result = self.fixture()
        for record in (protocol, result):
            record["bank_label_inventory"].pop("eligible", None)
        protocol_sha = self.write("keys/protocol.json", protocol)
        for record in (bank, manifest, diagnose, result):
            record["rebuild_protocol_sha256"] = protocol_sha
        self.write("keys/summary.json", bank)
        result["paired_manifest_sha256"] = self.write("keys/manifest.json", manifest)
        self.write("diagnose/summary.json", diagnose)
        self.write("diagnose/result.json", result)
        report = summary.read_summary(self.root)
        self.assertIn("bank_label_inventory.eligible", report["missing_fields"])
        self.assertIn("| eligible | N/A", summary.render(report))

    def test_top64_saved_transitions_verified_against_hand_counted_joint(self):
        self.fixture()
        report = summary.read_summary(self.root)
        self.assertEqual(report["transition_checks"], {arm: True for arm in summary.FOCUS_ARMS})
        intact, damaged, destroyed = report["transitions"][:3]
        self.assertEqual((intact["rescue"], intact["harm"], intact["net_tp"], intact["delta_fp"]), (1, 2, -1, 1))
        self.assertEqual((damaged["rescue"], damaged["harm"], damaged["net_tp"], damaged["delta_fp"]), (3, 1, 2, 0))
        self.assertEqual((destroyed["rescue"], destroyed["harm"], destroyed["net_tp"], destroyed["delta_fp"]), (1, 2, -1, 0))
        self.assertAlmostEqual(destroyed["rescue_rate"], 1 / 6)
        self.assertEqual(intact["harm_rate"], .25)
        text = summary.render(report)
        self.assertIn("1 | 2 | 50.000 | 25.000 | -1 | +1", text)
        self.assertIn("不是训练后 Reader 收益", text)

    def test_transition_count_or_rate_inconsistent_with_joint_refused(self):
        for field, value in (("rescue", 99), ("harm_rate", .9), ("net_tp", -99), ("delta_fp", 99)):
            with self.subTest(field=field):
                *_, result = self.fixture()
                result["transitions"]["grid_probq__labelvote_t128_top64"]["per_class"]["Damaged"][field] = value
                self.write("diagnose/result.json", result)
                with self.assertRaisesRegex(ValueError, "与 joint 不一致"):
                    summary.read_summary(self.root)

    def test_missing_transition_saved_class_does_not_get_derived_replacement(self):
        *_, result = self.fixture()
        record = result["transitions"]["grid_probq__labelvote_t128_top64"]
        del record["per_class"]["Damaged"]
        self.write("diagnose/result.json", result)
        report = summary.read_summary(self.root)
        row = report["transitions"][1]
        self.assertIsNone(row["rescue"])
        self.assertIsNone(row["net_tp"])
        self.assertIn("transitions.grid_probq__labelvote_t128_top64.per_class.Damaged", report["missing_fields"])
        self.assertIn("| t128_top64 | Damaged | N/A", summary.render(report))

    def test_missing_transition_joint_preserves_saved_counts_unverified(self):
        *_, result = self.fixture()
        del result["transitions"]["grid_probq__labelvote_t128_top64"]["joint_gt_reference_candidate"]
        self.write("diagnose/result.json", result)
        report = summary.read_summary(self.root)
        self.assertEqual(report["transitions"][1]["rescue"], 3)
        self.assertFalse(report["transition_checks"]["t128_top64"])
        self.assertTrue(any("joint_gt_reference_candidate" in name for name in report["missing_fields"]))

    def test_transition_joint_bad_shape_negative_or_float_rejected(self):
        for change in ("shape", "negative", "float"):
            *_, result = self.fixture()
            joint = result["transitions"]["grid_probq__labelvote_t128_top64"]["joint_gt_reference_candidate"]
            if change == "shape":
                joint.pop()
            else:
                joint[0][0][0] = -1 if change == "negative" else 1.0
            self.write("diagnose/result.json", result)
            with self.assertRaisesRegex(ValueError, "非负整数"):
                summary.read_summary(self.root)

    def test_transition_zero_denominator_rates_stay_na(self):
        *_, result = self.fixture()
        pair = result["transitions"]["grid_probq__labelvote_t128_top64"]
        pair["joint_gt_reference_candidate"] = [[[0] * 4 for _ in range(4)] for _ in range(4)]
        pair["per_class"] = {name: dict(rescue=0, harm=0, net_tp=0, delta_fp=0,
                                        rescue_rate=None, harm_rate=None) for name in summary.CLASS_NAMES[1:]}
        self.write("diagnose/result.json", result)
        report = summary.read_summary(self.root)
        self.assertIsNone(report["transitions"][1]["harm_rate"])
        self.assertIsNone(report["transitions"][1]["rescue_rate"])
        self.assertFalse(any("rescue_rate" in name for name in report["missing_fields"]))

    def test_per_event_differences_are_percentage_points_not_relative_percent(self):
        self.fixture()
        events = summary.read_summary(self.root)["per_event"]
        self.assertEqual([row["event"] for row in events], ["alpha", "beta"])
        self.assertAlmostEqual(events[0]["t128_top64_delta_pp"], 3)
        self.assertAlmostEqual(events[0]["f256_top64_delta_pp"], -2)
        self.assertAlmostEqual(events[1]["t128_top64_delta_pp"], -5)
        self.assertAlmostEqual(events[1]["f256_top64_delta_pp"], 4)
        text = summary.render(summary.read_summary(self.root))
        self.assertIn("| alpha | 20.000 | +3.000 | -2.000 |", text)
        self.assertIn("不包含显著性", text)

    def test_missing_event_and_transition_fields_show_na(self):
        *_, result = self.fixture()
        del result["per_event"]["alpha"]["labelvote_f256_top64"]
        del result["transitions"]
        self.write("diagnose/result.json", result)
        report = summary.read_summary(self.root)
        self.assertIsNone(report["per_event"][0]["f256_top64_delta_pp"])
        self.assertIsNone(report["transitions"][0]["rescue"])
        self.assertIn("transitions", report["missing_fields"])
        self.assertIn("per_event.alpha.labelvote_f256_top64", report["missing_fields"])

    def test_class_compatibility_uses_each_query_class_mass(self):
        self.fixture()
        row = summary.read_summary(self.root)["retrieval"][0]
        for observed, expected in zip(row["class_soft_compatibility"], [.9, .5, .1, .4]):
            self.assertAlmostEqual(observed, expected)
        self.assertNotEqual(row["class_soft_compatibility"][2], .3 / 8)

    def test_missing_class_compatibility_is_na_without_fallback(self):
        *_, result = self.fixture()
        del result["posthoc_retrieval"]["t128_top8"]["class_soft_compatibility_sum"]
        self.write("diagnose/result.json", result)
        report = summary.read_summary(self.root)
        self.assertEqual(report["retrieval"][0]["class_soft_compatibility"], [None] * 4)
        self.assertIn("t128_top8.class_soft_compatibility_sum", report["missing_fields"])


if __name__ == "__main__":
    unittest.main()
