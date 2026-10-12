#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Discover and run all supported BRIGHT checkpoints on Windows CPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


BEST_NAMES = (
    "best_valid_miou.pth",
    "best_balanced_score.pth",
    "best_damaged_iou.pth",
    "best_destroyed_iou.pth",
    "best_model.pth",
    "best.pth",
)
LATEST_NAME = "latest_resume.pth"
FALLBACK_NAMES = ("final_model.pth", LATEST_NAME)
PRIMARY_MODELS = {
    "M01_BGCA_none",
    "M02_A4_noROI_MDDMbase",
    "M03_A4_MDDMbase_ROId5",
    "M04_A4DG_b10",
    "M05_A4D_MDDMlarge",
    "A4_DamageExistHead",
}
SECONDARY_MODELS = {
    "BGCA_soft",
    "BGCA_dilated_soft",
    "BGCA_soft_align",
    "BGCA_dilated_soft_align",
    "M06C_dilatedROI_warmupAlign_lam0001",
}
BOOL_TRUE_FLAGS = {
    "use_align",
    "use_shared_diff",
    "use_modality_prompt",
    "use_reliability_gate",
    "use_diff_branch",
    "use_mddm",
    "mddm_use_roi",
    "use_inner_ring_context",
    "use_damage_exist_head",
    "use_hdc",
    "use_intact_consistency",
    "ic_normalize",
    "use_damage_margin",
}
BOOL_FALSE_FLAGS = {
    "detach_mask": "--no_detach_mask",
    "use_residual_diff": "--no_use_residual_diff",
    "gate_use_diff": "--no_gate_use_diff",
    "dg_use_roi": "--no_dg_use_roi",
}
VALUE_ARGS = (
    "roi_mode",
    "roi_alpha",
    "roi_dilate",
    "roi_fusion",
    "roi_beta",
    "align_type",
    "shared_diff_dim",
    "gamma_diff",
    "mddm_variant",
    "mddm_gamma_init",
    "dg_gate_reduction",
    "dg_gamma_init",
    "dg_gamma_mode",
    "dg_large_beta",
    "dg_warmup_iters",
    "dg_gate_bias_init",
    "ring_kernel_size",
    "irc_gamma_init",
    "lambda_exist",
    "exist_pos_weight",
    "exist_loss_type",
    "exist_focal_gamma",
    "hdc_mode",
    "hdc_alpha",
    "lambda_hdc",
    "lambda_severity",
    "lambda_ic",
    "ic_dim",
    "lambda_margin",
    "margin_damaged",
    "margin_destroyed",
)


@dataclass
class EvalItem:
    priority: int
    standard_model: str
    old_dir_name: str
    checkpoint_path: str
    checkpoint_name: str
    param_status: str
    inferred_type: str
    save_dir: str
    command: str
    args_json: str = ""


def model_priority(model: str, old_dir_name: str) -> int:
    normalized = old_dir_name.replace("\\", "/").lower()
    if any(token in normalized for token in ("exp01_", "exp02_", "exp03_", "exp04_", "m06c_")):
        return 2
    if any(token in normalized for token in ("bgca_baseline_align/", "a4_irc_", "bright/", "bright_mymodel/")):
        return 3
    if model in PRIMARY_MODELS:
        return 1
    if model in SECONDARY_MODELS or "M06C_" in old_dir_name:
        return 2
    return 3


def quote_command(parts: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in parts])


def nearest_args_json(directory: Path, saved_root: Path) -> Optional[Path]:
    current = directory
    while current == saved_root or saved_root in current.parents:
        candidate = current / "args.json"
        if candidate.is_file():
            return candidate
        if current == saved_root:
            break
        current = current.parent
    return None


def load_args(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"WARNING: cannot read {path}: {exc}")
        return None


def a4_args() -> List[str]:
    return [
        "--roi_mode", "dilated_soft",
        "--roi_fusion", "multiplicative",
        "--roi_beta", "1.0",
        "--roi_alpha", "0.1",
        "--roi_dilate", "5",
        "--use_mddm",
        "--mddm_variant", "base",
        "--mddm_gamma_init", "0.0",
        "--mddm_use_roi",
    ]


def classify_path(path: Path) -> Tuple[str, str, str, List[str]]:
    text = str(path).lower().replace("-", "_")
    if "unet" in text or "mymodel" in text:
        return "PARAM_UNCERTAIN_UNKNOWN", "UNSUPPORTED", "PARAM_UNCERTAIN_UNKNOWN", []
    if "hdc" in text or "m11_a4hdc" in text:
        return (
            "M11_A4HDC_a03", "DamageFormer_BGCA", "PATH_RULE",
            a4_args() + [
                "--use_hdc", "--hdc_mode", "blend", "--hdc_alpha", "0.3",
                "--lambda_exist", "0.1", "--lambda_severity", "0.2",
                "--lambda_hdc", "0.5", "--exist_loss_type", "bce",
            ],
        )
    if "irc" in text or "innerring" in text:
        return (
            "M06_A4IRC", "DamageFormer_BGCA", "PATH_RULE",
            a4_args() + ["--use_inner_ring_context", "--ring_kernel_size", "15"],
        )
    if "exist" in text or "damageexist" in text:
        extra = ["--use_damage_exist_head", "--lambda_exist", "0.1"]
        if "focal" in text:
            extra += ["--exist_loss_type", "focal", "--exist_focal_gamma", "2.0"]
        else:
            extra += ["--exist_loss_type", "bce"]
        return "A4_DamageExistHead", "DamageFormer_BGCA", "PATH_RULE", a4_args() + extra
    if "a4dg" in text or "dual_scale_gate" in text or "dual_scale" in text:
        beta = "0.3" if any(token in text for token in ("b03", "beta0.3", "beta03")) else "1.0"
        name = "M07_A4DG_b03" if beta == "0.3" else "M04_A4DG_b10"
        return (
            name, "DamageFormer_BGCA", "PATH_RULE",
            a4_args()[:-6] + [
                "--use_mddm", "--mddm_variant", "dual_scale_gate",
                "--mddm_gamma_init", "0.0", "--mddm_use_roi",
                "--dg_gate_reduction", "4", "--dg_gamma_init", "0.0",
                "--dg_gamma_mode", "sigmoid01", "--dg_large_beta", beta,
            ],
        )
    if any(token in text for token in ("m05_a4d", "mddmlarge", "largescale", "large_scale", "a4_d_")):
        args = a4_args()
        args[args.index("base")] = "large_scale"
        return "M05_A4D_MDDMlarge", "DamageFormer_BGCA", "PATH_RULE", args
    if any(token in text for token in ("m03_a4", "a4_mddm_dilatedroi", "mddmbase_roid5")):
        return "M03_A4_MDDMbase_ROId5", "DamageFormer_BGCA", "PATH_RULE", a4_args()
    if "a4_noroi" in text or "noroi_mddm" in text or "m02_a4" in text:
        return (
            "M02_A4_noROI_MDDMbase", "DamageFormer_BGCA", "PATH_RULE",
            ["--roi_mode", "none", "--use_mddm", "--mddm_variant", "base"],
        )
    if "bgca_none" in text or "a0_bgca_none" in text or "m01_bgca_none" in text:
        return "M01_BGCA_none", "DamageFormer_BGCA", "PATH_RULE", ["--roi_mode", "none"]
    if "damageformer" in text and "bgca" not in text:
        if "buildingguidedcontrast" in text:
            return "PARAM_UNCERTAIN_UNKNOWN", "UNSUPPORTED", "PARAM_UNCERTAIN_UNKNOWN", []
        return "M00_DamageFormer", "DamageFormer", "PATH_RULE", []
    if "bgca" in text:
        return "PARAM_UNCERTAIN_UNKNOWN", "DamageFormer_BGCA", "PARAM_UNCERTAIN_UNKNOWN", []
    return "PARAM_UNCERTAIN_UNKNOWN", "UNSUPPORTED", "PARAM_UNCERTAIN_UNKNOWN", []


def args_from_json(config: Dict[str, Any]) -> List[str]:
    result: List[str] = []
    for key in VALUE_ARGS:
        if key in config and config[key] is not None:
            result.extend([f"--{key}", str(config[key])])
    for key in BOOL_TRUE_FLAGS:
        if config.get(key) is True:
            result.append(f"--{key}")
    for key, false_flag in BOOL_FALSE_FLAGS.items():
        if config.get(key) is False:
            result.append(false_flag)
    return result


def standard_name_from_config(path: Path, config: Dict[str, Any]) -> str:
    if "m06c_" in str(path).lower():
        return "M06C_dilatedROI_warmupAlign_lam0001"
    variant = str(config.get("mddm_variant", "base"))
    if config.get("use_hdc"):
        return "M11_A4HDC_a03"
    if config.get("use_inner_ring_context"):
        return "M06_A4IRC"
    if config.get("use_damage_exist_head"):
        return "A4_DamageExistHead"
    if config.get("use_mddm") and variant == "dual_scale_gate":
        beta = float(config.get("dg_large_beta", 1.0))
        return "M07_A4DG_b03" if abs(beta - 0.3) < 1e-8 else "M04_A4DG_b10"
    if config.get("use_mddm") and variant == "large_scale":
        return "M05_A4D_MDDMlarge"
    if config.get("use_mddm") and config.get("mddm_use_roi"):
        return "M03_A4_MDDMbase_ROId5"
    if config.get("use_mddm"):
        return "M02_A4_noROI_MDDMbase"
    if "bgca" in str(path).lower():
        roi_mode = str(config.get("roi_mode", "none")).lower()
        if roi_mode == "none":
            return "M01_BGCA_none"
        suffix = "_align" if config.get("use_align") else ""
        return f"BGCA_{roi_mode}{suffix}"
    return "M00_DamageFormer"


def discover_checkpoint_dirs(saved_root: Path) -> Iterable[Tuple[Path, List[Path]]]:
    for directory, _subdirs, filenames in os.walk(saved_root):
        names = set(filenames)
        selected = [Path(directory) / name for name in BEST_NAMES if name in names]
        if not selected:
            selected = [Path(directory) / name for name in FALLBACK_NAMES if name in names][:1]
        if selected:
            yield Path(directory), selected


def make_items(args: argparse.Namespace) -> List[EvalItem]:
    saved_root = args.saved_root.resolve()
    items: List[EvalItem] = []
    for directory, checkpoints in discover_checkpoint_dirs(saved_root):
        args_path = nearest_args_json(directory, saved_root)
        config = load_args(args_path)
        fallback_name, fallback_type, fallback_status, fallback_args = classify_path(directory)
        if config is not None:
            model_type = "DamageFormer_BGCA" if "bgca" in str(directory).lower() or config.get("use_mddm") else fallback_type
            if model_type not in {"DamageFormer", "DamageFormer_BGCA"}:
                model_type = fallback_type
            standard_model = standard_name_from_config(directory, config)
            specific_args = args_from_json(config)
            if (
                config.get("use_mddm")
                and str(config.get("mddm_variant", "")).lower() == "dual_scale_gate"
                and "dg_large_beta" not in config
            ):
                # Older training args.json files omitted this newer constructor
                # argument.  The trained M04/A4-DG default is beta=1.0.
                specific_args.extend(["--dg_large_beta", "1.0"])
            status = "ARGS_JSON"
        else:
            standard_model, model_type, status, specific_args = (
                fallback_name, fallback_type, fallback_status, fallback_args
            )

        for checkpoint in checkpoints:
            old_dir_name = str(directory.relative_to(saved_root))
            digest = hashlib.sha1(str(checkpoint.resolve()).encode("utf-8")).hexdigest()[:8]
            tag = f"{checkpoint.stem}_{digest}"
            save_dir = args.results_root / standard_model / tag
            command_parts: List[str] = [
                args.python,
                str(args.project_root / "full_evaluate_unified_local_cpu.py"),
                "--eval_protocol", "official",
                "--model_type", model_type if model_type != "UNSUPPORTED" else "DamageFormer_BGCA",
                "--model_path", str(checkpoint.resolve()),
                "--dataset_path", str(args.dataset_root.resolve()),
                "--data_list_path", str(args.test_list.resolve()),
                "--image_size", str(args.image_size),
                "--batch_size", str(args.batch_size),
                "--num_workers", "0",
                "--save_dir", str(save_dir.resolve()),
            ]
            if args.num_batches is not None:
                command_parts.extend(["--num_batches", str(args.num_batches)])
            command_parts.extend(specific_args)
            items.append(EvalItem(
                priority=model_priority(standard_model, old_dir_name),
                standard_model=standard_model,
                old_dir_name=old_dir_name,
                checkpoint_path=str(checkpoint.resolve()),
                checkpoint_name=checkpoint.name,
                param_status=status,
                inferred_type=model_type,
                save_dir=str(save_dir.resolve()),
                command=quote_command(command_parts),
                args_json=str(args_path.resolve()) if args_path else "",
            ))
    return sorted(
        items,
        key=lambda item: (
            item.priority,
            item.standard_model,
            BEST_NAMES.index(item.checkpoint_name) if item.checkpoint_name in BEST_NAMES else 99,
            item.checkpoint_path,
        ),
    )


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_items(items: Sequence[EvalItem], analysis_root: Path, overwrite: bool) -> None:
    logs = analysis_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    failures: List[Dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {item.standard_model} / {item.checkpoint_name}")
        if item.param_status.startswith("PARAM_UNCERTAIN") or item.inferred_type == "UNSUPPORTED":
            failures.append({
                "model": item.standard_model,
                "checkpoint_path": item.checkpoint_path,
                "param_status": item.param_status,
                "returncode": "SKIPPED",
                "log_path": "",
                "command": item.command,
                "error_summary": f"{item.param_status}: model architecture/parameters require confirmation",
            })
            print("  SKIP: parameter or architecture uncertain")
            continue

        save_dir = Path(item.save_dir)
        summary_path = save_dir / "summary_official.json"
        if summary_path.is_file() and not overwrite:
            print("  SKIP: summary already exists")
            continue
        save_dir.mkdir(parents=True, exist_ok=True)
        metadata = asdict(item)
        (save_dir / "local_eval_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log_path = logs / f"{item.standard_model}_{save_dir.name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                item.command,
                cwd=str(Path(__file__).resolve().parents[2]),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
        if completed.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
            failures.append({
                "model": item.standard_model,
                "checkpoint_path": item.checkpoint_path,
                "param_status": item.param_status,
                "returncode": completed.returncode,
                "log_path": str(log_path.resolve()),
                "command": item.command,
                "error_summary": " | ".join(tail),
            })
            print(f"  FAILED: see {log_path}")
        else:
            print(f"  OK: {save_dir}")

    write_csv(
        analysis_root / "eval_failed_models.csv",
        ["model", "checkpoint_path", "param_status", "returncode", "log_path", "error_summary", "command"],
        failures,
    )


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--saved-root", type=Path, default=project_root / "saved_weights")
    parser.add_argument("--dataset-root", type=Path, default=project_root / "bda_benchmark" / "dataset")
    parser.add_argument(
        "--test-list", type=Path,
        default=project_root / "bda_benchmark" / "dataset" / "splitname" / "standard_ML" / "test_set.txt",
    )
    parser.add_argument("--results-root", type=Path, default=project_root / "eval_results_official_local_cpu")
    parser.add_argument("--analysis-root", type=Path, default=project_root / "eval_analysis_cpu")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--num-batches", type=int, default=None,
        help="Optional developer smoke-test limit; normal 20/878-image runs leave this unset.",
    )
    parser.add_argument(
        "--max-checkpoints", type=int, default=None,
        help="Optional developer limit applied after discovery.",
    )
    parser.add_argument(
        "--priority-max", type=int, choices=[1, 2, 3], default=3,
        help="Run/register models up to this priority (1=main line only).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.project_root = args.project_root.resolve()
    args.results_root = args.results_root.resolve()
    args.analysis_root = args.analysis_root.resolve()
    args.analysis_root.mkdir(parents=True, exist_ok=True)

    items = make_items(args)
    items = [item for item in items if item.priority <= args.priority_max]
    if args.max_checkpoints is not None:
        items = items[: max(0, args.max_checkpoints)]
    registry_fields = list(EvalItem.__dataclass_fields__.keys())
    write_csv(args.analysis_root / "local_eval_registry.csv", registry_fields, [asdict(item) for item in items])
    (args.analysis_root / "local_eval_commands.txt").write_text(
        "\n".join(item.command for item in items) + ("\n" if items else ""), encoding="utf-8"
    )

    uncertain = sum(item.param_status.startswith("PARAM_UNCERTAIN") for item in items)
    print(f"Discovered checkpoints: {len(items)}")
    print(f"Parameter/architecture uncertain: {uncertain}")
    print(f"Registry: {args.analysis_root / 'local_eval_registry.csv'}")
    print(f"Commands: {args.analysis_root / 'local_eval_commands.txt'}")
    if args.dry_run:
        (args.analysis_root / "logs").mkdir(parents=True, exist_ok=True)
        failed_path = args.analysis_root / "eval_failed_models.csv"
        if not failed_path.exists():
            write_csv(
                failed_path,
                ["model", "checkpoint_path", "param_status", "returncode", "log_path", "error_summary", "command"],
                [],
            )
    else:
        run_items(items, args.analysis_root, args.overwrite)


if __name__ == "__main__":
    main()
