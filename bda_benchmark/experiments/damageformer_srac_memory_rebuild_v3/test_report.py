"""Report scope, aggregate denominators and rejection of unsupported claims."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from .diagnose import FORMAT
from .report import read_report, render, retrieval_means, overlap_means, weighted_means


class ReportTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def fixture(self, *, formal=False, count=2):
        summary = dict(format=FORMAT, complete=formal, full_validation_evidence=formal,
                       execution_complete=True, image_count=count)
        result = dict(summary, training=False, test_accessed=False,
            historical_reproduction_claim=False, labelvote_is_untrained_diagnostic=True,
            smoke=not formal, paired_bank_smoke=not formal,
            metrics={"a1_pixel": dict(valid_miou=.2, iou=[.3, .1, .2, .3])},
            validation={key: True for key in ("protocol", "bank", "split", "labels", "reference", "native", "pool", "artifacts", "immutable")})
        (self.root / "summary.json").write_text(json.dumps(summary))
        (self.root / "result.json").write_text(json.dumps(result))
        return summary, result

    def test_smoke_render_marks_scores_as_untrained_nonformal(self):
        self.fixture()
        text = render(self.root)
        self.assertIn("未训练", text)
        self.assertIn("不能当作完整 Val450", text)
        self.assertIn("不代表历史旧库复现", text)
        self.assertIn("a1_pixel\t0.2", text)

    def test_formal_requires_450_and_all_validation_gates(self):
        _summary, result = self.fixture(formal=True, count=450)
        self.assertIn("完整 Val450 新库配对诊断通过", render(self.root))
        result["validation"]["labels"] = False
        (self.root / "result.json").write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "evidence gates"):
            read_report(self.root)

    def test_empty_completed_claim_at_smoke_count_rejected(self):
        self.fixture(formal=True, count=2)
        with self.assertRaisesRegex(ValueError, "evidence gates"):
            read_report(self.root)

    def test_summary_result_consistency_checked(self):
        _summary, result = self.fixture()
        result["image_count"] = 3
        (self.root / "result.json").write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "disagree"):
            read_report(self.root)

    def test_failed_without_result_keeps_error_and_does_not_show_scores(self):
        (self.root / "summary.json").write_text(json.dumps(dict(format=FORMAT, complete=False, error="ValueError")))
        text = render(self.root)
        self.assertIn("ValueError", text)
        self.assertIn("尚无完成", text)
        self.assertNotIn("foreground_mIoU", text)

    def test_historical_reproduction_or_training_claim_rejected(self):
        for claim in ("historical_reproduction_claim", "training", "test_accessed"):
            _summary, result = self.fixture()
            result[claim] = True
            (self.root / "result.json").write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "claim/scope"):
                read_report(self.root)

    def test_retrieval_denominators_use_aggregated_counts(self):
        row = dict(query_count=10, supported_query_count=8, soft_compatibility_sum=4,
                   best_neighbor_soft_compatibility_sum=6, dominant_label_agreement_count=5,
                   dominant_class_covered_count=7, quality_sum=[0, 0, 0, 0, 6, 0])
        values = retrieval_means(row)
        self.assertAlmostEqual(values["support_coverage"], .8)
        self.assertAlmostEqual(values["soft_compatibility_all_queries"], .4)
        self.assertAlmostEqual(values["soft_compatibility_supported_queries"], .5)
        self.assertAlmostEqual(values["weighted_support_purity_supported_queries"], .75)
        self.assertIsNone(retrieval_means({})["support_coverage"])

    def test_overlap_and_mass_denominators_are_explicit(self):
        values = overlap_means(dict(query_count=10, both_supported_query_count=5,
            neighbor_jaccard_sum=2, top1_agreement_count=1))
        self.assertAlmostEqual(values["mean_jaccard_all_queries"], .2)
        self.assertAlmostEqual(values["mean_jaccard_both_supported_queries"], .4)
        values = weighted_means(dict(queries=10, supported_queries=5, valid_mass=20,
            supported_valid_mass=8, mass_weighted_compatibility_sum=4))
        self.assertAlmostEqual(values["supported_mass_fraction"], .4)
        self.assertAlmostEqual(values["mass_weighted_soft_compatibility"], .5)


if __name__ == "__main__":
    unittest.main()
