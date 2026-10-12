from __future__ import annotations

import csv
import json

import torch

from scripts.calibrate_cached_rgid import main as calibrate_main
from scripts.compare_rgid_results import main as compare_main
from scripts.eval_cached_rgid import main as evaluate_main
from scripts.train_cached_rgid import main as train_main


def _sample(index: int):
    generator = torch.Generator().manual_seed(100 + index)
    instance_ids = torch.zeros(64, 64, dtype=torch.long)
    instance_ids[8:30, 8:28] = 1
    instance_ids[32:56, 34:58] = 2
    parent_mask = instance_ids > 0
    return {
        "fused_roi": torch.randn(256, 16, 16, generator=generator),
        "instance_ids": instance_ids,
        "parent_mask": parent_mask,
        "name": f"synthetic_{index}",
    }


def test_two_sample_cached_pipeline_runs_end_to_end(tmp_path):
    train_cache = tmp_path / "train_cache"
    val_cache = tmp_path / "val_cache"
    output_dir = tmp_path / "train"
    train_cache.mkdir()
    val_cache.mkdir()
    for index in range(2):
        sample = _sample(index)
        torch.save(sample, train_cache / f"{index:06d}.pt")
        torch.save(sample, val_cache / f"{index:06d}.pt")

    assert (
        train_main(
            [
                "--train-cache",
                str(train_cache),
                "--output-dir",
                str(output_dir),
                "--steps",
                "2",
                "--batch-size",
                "2",
                "--accumulation",
                "1",
                "--device",
                "cpu",
                "--checkpoint-steps",
                "2",
                "--num-workers",
                "0",
            ]
        )
        == 0
    )
    trained = output_dir / "rgid_step_2.pth"
    calibrated = output_dir / "rgid_step_2_calibrated.pth"
    assert trained.is_file()
    assert (
        calibrate_main(
            [
                "--checkpoint",
                str(trained),
                "--train-cache",
                str(train_cache),
                "--output",
                str(calibrated),
                "--batch-size",
                "2",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    calibrated_payload = torch.load(calibrated, map_location="cpu")
    assert "boundary_calibration" in calibrated_payload
    assert "boundary_gate_threshold" in calibrated_payload["boundary_calibration"]

    results_path = output_dir / "eval.json"
    assert (
        evaluate_main(
            [
                "--checkpoint",
                str(calibrated),
                "--val-cache",
                str(val_cache),
                "--output",
                str(results_path),
                "--batch-size",
                "2",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    with open(results_path, "r", encoding="utf-8") as file:
        results = json.load(file)
    assert all(method in results for method in ("g0", "g1", "g2", "g3"))

    csv_path = output_dir / "comparison.csv"
    assert (
        compare_main(
            ["--results", str(results_path), "--output-csv", str(csv_path)]
        )
        == 0
    )
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["Method"] for row in rows] == ["G0", "G1", "G2", "G3"]
