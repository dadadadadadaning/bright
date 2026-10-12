"""CPU-only comparisons; no BRIGHT image or checkpoint is loaded."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .compare import ARMS, checked_eval, compare


def fake_eval(arm="baseline"):
    metric = {"valid_miou": 0.5, "iou": [0.9, 0.5, 0.4, 0.6]}
    return {
        "format": "SRAC_Q_FACTORIAL_EVAL_V1",
        "complete": True, "full_validation_evidence": True,
        "image_count": 450, "reproduction": {"passed": True},
        "native_a1_replay_ok": True, "test_accessed": False,
        "immutable": {"files": True, "models": True},
        "protocol_sha256": "a" * 64, "source_sha256": "b" * 64,
        "numerical_profile": "legacy", "pool_mode": "native",
        "checkpoint_sha256": "c" * 64, "arm": arm,
        "checkpoint_steps": 2000,
        "training_hyperparameters": {"steps": 2000, "eval_every": 2000,
                                      "batch": 2, "accum": 8, "workers": 4,
                                      "lr": 1e-4, "aux_weight": 0.1,
                                      "cue_gain": 0.1, "amp": "bf16"},
        "metrics": {name: dict(metric) for name in ARMS},
        "per_event": {"event": {name: dict(metric) for name in ARMS}},
    }


class ComparisonTests(unittest.TestCase):
    def test_rejects_partial_failed_or_test_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            for mutation in (
                {"image_count": 449}, {"full_validation_evidence": False},
                {"reproduction": {"passed": False}}, {"test_accessed": True},
                {"immutable": {"files": False}}, {"native_a1_replay_ok": False},
            ):
                row = fake_eval()
                row.update(mutation)
                path.write_text(json.dumps(row), encoding="utf-8")
                with self.assertRaises(ValueError):
                    checked_eval(path)

    def test_compares_four_classes_and_rejects_changed_protocol(self):
        baseline = fake_eval()
        candidate = fake_eval("semantic_reconstruction")
        candidate["metrics"]["q_grid_probq"]["valid_miou"] = 0.51
        candidate["metrics"]["q_grid_probq"]["iou"][2] = 0.42
        row = compare(baseline, candidate)
        self.assertAlmostEqual(row["differences_pp"]["q_grid_probq"]["foreground_miou_pp"], 1.0)
        self.assertAlmostEqual(row["differences_pp"]["q_grid_probq"]["iou_pp"][2], 2.0)
        candidate["protocol_sha256"] = "d" * 64
        with self.assertRaises(ValueError):
            compare(baseline, candidate)

    def test_rejects_unmatched_training_budget(self):
        baseline = fake_eval()
        candidate = fake_eval("sar_cue")
        candidate["checkpoint_steps"] = 1000
        with self.assertRaises(ValueError):
            compare(baseline, candidate)
        candidate = fake_eval("sar_cue")
        candidate["training_hyperparameters"]["cue_gain"] = 0.25
        with self.assertRaises(ValueError):
            compare(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
