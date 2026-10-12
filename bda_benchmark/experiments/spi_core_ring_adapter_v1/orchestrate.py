from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command):
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def train_all(args, steps, checkpoint_root):
    run([
        sys.executable, "-u", "-m", "bda_benchmark.experiments.spi_core_ring_adapter_v1.train_joint",
        "--dataset-path", args.dataset_path, "--train-list", args.train_list,
        "--base-checkpoint", args.base_checkpoint, "--output-dir", checkpoint_root,
        "--seed", args.seed, "--device", args.device, "--crop-size", args.crop_size,
        "--batch-size", args.batch_size, "--accumulation-steps", args.accumulation_steps,
        "--optimizer-steps", steps, "--num-workers", args.num_workers,
        "--learning-rate", args.learning_rate, "--save-every", args.save_every,
    ])


def audit(args, steps, suffix, checkpoint_root):
    output = Path(args.result_root) / f"SPI_CORE_RING_{suffix}_seed{args.seed}.json"
    run([
        sys.executable, "-u", "-m", "bda_benchmark.experiments.spi_core_ring_adapter_v1.audit",
        "--base-checkpoint", args.base_checkpoint, "--checkpoint-root", checkpoint_root,
        "--dataset-path", args.dataset_path, "--val-list", args.val_list,
        "--sar-stats-json", args.sar_stats_json, "--output-json", output,
        "--device", args.device, "--num-workers", args.num_workers,
        "--expected-images", args.expected_images, "--expected-steps", steps,
        "--bootstrap-iterations", args.bootstrap_iterations, "--seed", args.seed,
    ])
    return output


def main(argv=None):
    parser = argparse.ArgumentParser("Sequential, restart-safe RTX 4090 experiment controller")
    parser.add_argument("--mode", choices=("smoke", "pilot", "auto", "full"), default="auto")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--sar-stats-json", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--pilot-steps", type=int, default=500)
    parser.add_argument("--full-steps", type=int, default=3000)
    parser.add_argument("--expected-images", type=int, default=450)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args(argv)

    Path(args.checkpoint_root).mkdir(parents=True, exist_ok=True)
    Path(args.result_root).mkdir(parents=True, exist_ok=True)
    if args.mode == "smoke":
        args.bootstrap_iterations = 100
        args.expected_images = 450
        args.pilot_steps = 2
        smoke_root = Path(args.checkpoint_root) / "smoke2"
        train_all(args, 2, smoke_root)
        # Smoke still uses only two validation images, directly invoke audit with max-images.
        output = Path(args.result_root) / f"SPI_CORE_RING_smoke2_seed{args.seed}.json"
        command = [
            sys.executable, "-u", "-m", "bda_benchmark.experiments.spi_core_ring_adapter_v1.audit",
            "--base-checkpoint", args.base_checkpoint, "--checkpoint-root", smoke_root,
            "--dataset-path", args.dataset_path, "--val-list", args.val_list,
            "--sar-stats-json", args.sar_stats_json, "--output-json", output,
            "--device", args.device, "--num-workers", args.num_workers,
            "--expected-steps", 2, "--max-images", 2, "--bootstrap-iterations", 100,
            "--seed", args.seed,
        ]
        run(command)
        return 0

    pilot_root = Path(args.checkpoint_root) / f"pilot{args.pilot_steps}"
    full_root = Path(args.checkpoint_root) / f"full{args.full_steps}"
    if args.mode in ("pilot", "auto"):
        train_all(args, args.pilot_steps, pilot_root)
        pilot_output = audit(args, args.pilot_steps, f"pilot{args.pilot_steps}", pilot_root)
        if args.mode == "pilot":
            return 0
        decision = json.loads(pilot_output.read_text(encoding="utf-8"))["decision"]["decision"]
        if decision != "PROCEED_TO_FULL_CONFIRMATION":
            print(f"Automatic stop after pilot: {decision}", flush=True)
            print("No 3000-step jobs were launched; inspect the first failed causal check.", flush=True)
            return 0

    # The confirmatory 3000-step run starts from P3 rather than continuing the
    # screened 500-step optimizer state. This avoids data-order reuse and keeps
    # the final run interpretable; interruptions within this phase still resume.
    train_all(args, args.full_steps, full_root)
    audit(args, args.full_steps, f"full{args.full_steps}", full_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
