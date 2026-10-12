from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(arguments):
    command = [str(item) for item in arguments]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def audit_command(args, val_list, output, gate_checkpoint="", expected_images=450, max_images=0, calibration=False):
    command = [
        sys.executable, "-u", "-m", "bda_benchmark.experiments.spi_risk_gate_v1.audit",
        "--base-checkpoint", args.base_checkpoint, "--m4-checkpoint", args.m4_checkpoint,
        "--m4-steps", args.m4_steps, "--dataset-path", args.dataset_path,
        "--val-list", val_list, "--sar-stats-json", args.sar_stats_json,
        "--output-json", output, "--device", args.device, "--num-workers", args.num_workers,
        "--expected-images", expected_images, "--bootstrap-iterations", args.bootstrap_iterations,
        "--seed", args.seed,
    ]
    if gate_checkpoint:
        command += ["--gate-checkpoint", gate_checkpoint]
    if max_images:
        command += ["--max-images", max_images]
    if calibration:
        command += ["--calibration-mode"]
    run(command)


def train_gate(args, train_list, output_dir, steps):
    run([
        sys.executable, "-u", "-m", "bda_benchmark.experiments.spi_risk_gate_v1.train_gate",
        "--base-checkpoint", args.base_checkpoint, "--m4-checkpoint", args.m4_checkpoint,
        "--m4-steps", args.m4_steps, "--dataset-path", args.dataset_path,
        "--train-list", train_list, "--output-dir", output_dir,
        "--device", args.device, "--seed", args.seed, "--crop-size", args.crop_size,
        "--batch-size", args.batch_size, "--accumulation-steps", args.accumulation_steps,
        "--optimizer-steps", steps, "--num-workers", args.num_workers,
        "--learning-rate", args.learning_rate,
    ])


def main(argv=None):
    parser = argparse.ArgumentParser("Restart-safe D0-D5 controller")
    parser.add_argument(
        "--mode",
        choices=("stage-a", "smoke", "auto", "explore-d4"),
        default="auto",
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--m4-checkpoint", required=True)
    parser.add_argument("--m4-steps", type=int, default=500)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--gate-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args(argv)
    output_root = Path(args.output_root)
    checkpoint_root = Path(args.checkpoint_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    split_root = checkpoint_root / f"split_seed{args.seed}"
    run([sys.executable, "-m", "bda_benchmark.experiments.spi_risk_gate_v1.make_split",
         "--source-list", args.train_list, "--output-dir", split_root, "--seed", args.seed])
    split_manifest = json.loads((split_root / "split_manifest.json").read_text(encoding="utf-8"))
    gate_train_list = split_root / "gate_train.txt"
    gate_calibration_list = split_root / "gate_calibration.txt"

    if args.mode == "smoke":
        smoke_gate = checkpoint_root / "smoke2"
        train_gate(args, gate_train_list, smoke_gate, 2)
        audit_command(args, args.val_list, output_root / f"D0_D5_smoke2_seed{args.seed}.json",
                      smoke_gate / "gate_last.pth", max_images=2)
        return 0

    stage_a_output = output_root / f"D0_D3_D5_STAGE_A_val450_seed{args.seed}.json"
    if args.mode == "explore-d4" and stage_a_output.is_file():
        existing = json.loads(stage_a_output.read_text(encoding="utf-8"))
        if existing.get("protocol") != "SPI_RISK_GATE_PURE_SEMANTIC_VAL450_v1":
            raise RuntimeError(f"refusing incompatible Stage-A result: {stage_a_output}")
        print(f"Reusing locked Stage-A result: {stage_a_output}", flush=True)
    else:
        audit_command(args, args.val_list, stage_a_output)
    if args.mode == "stage-a":
        return 0
    stage_a = json.loads(stage_a_output.read_text(encoding="utf-8"))
    if stage_a["final"]["decision"] != "PROCEED_TO_D4_GATE_TRAINING" and args.mode != "explore-d4":
        failed = [
            name for name, passed in stage_a["stage_a"]["checks"].items()
            if not passed
        ]
        print(
            "Automatic stop: Stage-A did not satisfy every preregistered safety "
            f"condition. Failed checks: {failed}. This does not by itself mean "
            "that M4 contains no complementary evidence; inspect the Oracle and "
            "change audit separately.",
            flush=True,
        )
        return 0
    if stage_a["final"]["decision"] != "PROCEED_TO_D4_GATE_TRAINING":
        print(
            "Exploratory override: training D4 on Train2918 only. The failed "
            "Stage-A gate remains recorded and is not reclassified as passed. "
            "Val450 will still be protected by the internal calibration gate.",
            flush=True,
        )

    gate_root = checkpoint_root / f"gate{args.gate_steps}"
    train_gate(args, gate_train_list, gate_root, args.gate_steps)
    gate_checkpoint = gate_root / "gate_last.pth"
    calibration_output = output_root / f"D4_internal_calibration_seed{args.seed}.json"
    audit_command(args, gate_calibration_list, calibration_output, gate_checkpoint,
                  expected_images=split_manifest["calibration"], calibration=True)
    calibration = json.loads(calibration_output.read_text(encoding="utf-8"))
    if calibration["final"]["decision"] != "PROCEED_TO_VAL450_D4_AUDIT":
        print("Automatic stop: D4 failed the internal training calibration split; Val450 was not accessed again.", flush=True)
        return 0

    audit_command(args, args.val_list, output_root / f"D0_D5_D4_final_val450_seed{args.seed}.json", gate_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
