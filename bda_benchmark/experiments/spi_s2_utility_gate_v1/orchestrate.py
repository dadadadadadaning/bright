from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


MODULE = "bda_benchmark.experiments.spi_s2_utility_gate_v1"


def run(items) -> None:
    command = [str(item) for item in items]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def audit(args, list_path, output, policy, gate, expected, *, calibration=False, max_images=0, bootstrap=10000):
    command = [
        sys.executable, "-u", "-m", f"{MODULE}.audit",
        "--checkpoint", args.checkpoint,
        "--gate-checkpoint", gate,
        "--direction-policy-json", policy,
        "--dataset-path", args.dataset_path,
        "--list-path", list_path,
        "--sar-stats-json", args.sar_stats_json,
        "--output-json", output,
        "--device", args.device,
        "--num-workers", args.num_workers,
        "--expected-images", expected,
        "--minimum-area", args.minimum_area,
        "--gate-threshold", args.gate_threshold,
        "--bootstrap-iterations", bootstrap,
        "--seed", args.seed,
    ]
    if calibration:
        command.append("--calibration-mode")
    if max_images:
        command += ["--max-images", max_images]
    if not calibration:
        command += ["--expected-p3-miou", args.expected_p3_miou]
    run(command)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser("Leakage-safe S2 utility-gate controller")
    parser.add_argument("--mode", choices=("auto", "smoke", "fit-policy", "train-only"), default="auto")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--optimizer-steps", type=int, default=500)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-area", type=int, default=32)
    parser.add_argument("--gate-threshold", type=float, default=0.50)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--expected-p3-miou", type=float, default=0.605020)
    parser.add_argument(
        "--policy-max-images",
        type=int,
        default=-1,
        help="-1 uses the mode default; 0 means all direction-policy images",
    )
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    checkpoint_root = Path(args.checkpoint_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    split_root = checkpoint_root / f"split_seed{args.seed}"
    run([
        sys.executable, "-m", "bda_benchmark.experiments.spi_risk_gate_v1.make_split",
        "--source-list", args.train_list,
        "--output-dir", split_root,
        "--seed", args.seed,
    ])
    split = json.loads((split_root / "split_manifest.json").read_text(encoding="utf-8"))
    gate_train = split_root / "gate_train.txt"
    calibration = split_root / "gate_calibration.txt"
    policy_path = checkpoint_root / f"direction_policy_seed{args.seed}.json"
    policy_max = (
        args.policy_max_images
        if args.policy_max_images >= 0
        else (2 if args.mode == "smoke" else 0)
    )
    run([
        sys.executable, "-u", "-m", f"{MODULE}.fit_direction_policy",
        "--checkpoint", args.checkpoint,
        "--dataset-path", args.dataset_path,
        "--list-path", gate_train,
        "--sar-stats-json", args.sar_stats_json,
        "--output-json", policy_path,
        "--device", args.device,
        "--num-workers", args.num_workers,
        "--expected-images", split["train"],
        "--max-images", policy_max,
        "--minimum-area", args.minimum_area,
    ])
    if args.mode == "fit-policy":
        return 0

    steps = 2 if args.mode == "smoke" else args.optimizer_steps
    gate_root = checkpoint_root / f"gate_step{steps}_seed{args.seed}"
    run([
        sys.executable, "-u", "-m", f"{MODULE}.train_gate",
        "--checkpoint", args.checkpoint,
        "--dataset-path", args.dataset_path,
        "--train-list", gate_train,
        "--sar-stats-json", args.sar_stats_json,
        "--output-dir", gate_root,
        "--device", args.device,
        "--seed", args.seed,
        "--optimizer-steps", steps,
        "--accumulation-steps", args.accumulation_steps,
        "--num-workers", args.num_workers,
        "--learning-rate", args.learning_rate,
        "--minimum-area", args.minimum_area,
    ])
    if args.mode == "train-only":
        return 0
    gate_checkpoint = gate_root / "gate_last.pth"
    calibration_output = output_root / f"internal_calibration_step{steps}_seed{args.seed}.json"
    audit(
        args, calibration, calibration_output, policy_path, gate_checkpoint,
        split["calibration"], calibration=True,
        max_images=2 if args.mode == "smoke" else 0,
        bootstrap=10 if args.mode == "smoke" else args.bootstrap_iterations,
    )
    if args.mode == "smoke":
        print("Smoke completed. Its pass/fail decision is not scientific evidence.")
        return 0
    result = json.loads(calibration_output.read_text(encoding="utf-8"))
    if result["final"]["decision"] != "PROCEED_TO_VAL450":
        failed = [name for name, passed in result["final"]["checks"].items() if not passed]
        print(f"Automatic stop after internal calibration: {failed}", flush=True)
        print("Val450 was not accessed.", flush=True)
        return 0
    audit(
        args, args.val_list,
        output_root / f"validation_full_step{steps}_seed{args.seed}.json",
        policy_path, gate_checkpoint, 450,
        bootstrap=args.bootstrap_iterations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
