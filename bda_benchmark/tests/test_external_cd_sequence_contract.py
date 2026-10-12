import unittest
from pathlib import Path


class ExternalCDSequenceContractTests(unittest.TestCase):
    def test_sequence_keeps_fca_as_explicit_stage(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "run" / "run_ExternalCD_BRIGHT_sequence.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("baselines)", text)
        self.assertIn("fca)", text)
        self.assertIn("run_experiment x2 noFCA 0", text)
        self.assertIn('run_experiment x2 FCA001 "${FCA_WEIGHT:-0.01}"', text)
        self.assertIn("training_summary.json", text)
        self.assertIn("selected_step", text)
        self.assertIn("event_worst_pp", text)
        self.assertIn("output_dir", text)
        self.assertNotIn("eval_test", text)

    def test_training_summary_exposes_declared_selection_metrics(self):
        root = Path(__file__).resolve().parents[1]
        text = (
            root
            / "script"
            / "standard_ML"
            / "train_DamageFormer_RGBSAR_Heterogeneous.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"selected_step": best_step', text)
        self.assertIn(
            'metrics["event_worst_official_style_4class_miou"]',
            text,
        )


if __name__ == "__main__":
    unittest.main()
