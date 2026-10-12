#!/usr/bin/env python
"""List models or validate a teacher/student pair and its checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bda_benchmark.model import list_model_specs
from bda_benchmark.script.common.paths import resolve_artifact_paths


def _json_object(value: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("constructor kwargs must be a JSON object")
    return parsed


def _parameter_counts(model) -> Dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def _print_registry() -> None:
    print(
        f"{'name':43} {'family':24} {'input':12} {'output'}"
    )
    print("-" * 105)
    for spec in list_model_specs():
        print(
            f"{spec.name:43} {spec.family:24} "
            f"{spec.input_mode:12} {spec.output_contract}"
        )
        if spec.required_kwargs:
            print(f"  required: {', '.join(spec.required_kwargs)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect the unified BRIGHT model registry or a distillation pair"
    )
    parser.add_argument("--list", action="store_true", help="list registered models")
    parser.add_argument("--teacher")
    parser.add_argument("--student")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--student-checkpoint")
    parser.add_argument("--teacher-kwargs", type=_json_object, default={})
    parser.add_argument("--student-kwargs", type=_json_object, default={})
    parser.add_argument(
        "--offline",
        action="store_true",
        help="disable pretrained Swin initialization where supported",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="allow missing or unexpected checkpoint keys",
    )
    parser.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
        help="allow pickle loading for a trusted legacy checkpoint",
    )
    parser.add_argument("--experiment")
    parser.add_argument("--protocol", default="distillation")
    parser.add_argument("--output-root")
    parser.add_argument(
        "--create-artifact-dirs",
        action="store_true",
        help="create the resolved checkpoint/log/evaluation directories",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.list:
        _print_registry()
        if not args.teacher and not args.student:
            return 0

    if not args.teacher or not args.student:
        raise SystemExit("--teacher and --student are required unless only --list is used")

    try:
        from bda_benchmark.model.distillation import DistillationPair, load_checkpoint
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "PyTorch is required to construct a teacher/student pair. "
                "Registry listing with --list does not require PyTorch."
            ) from exc
        raise

    pair = DistillationPair.from_registry(
        args.teacher,
        args.student,
        teacher_kwargs=args.teacher_kwargs,
        student_kwargs=args.student_kwargs,
        offline=args.offline,
    )

    checkpoint_reports: Dict[str, Any] = {}
    strict = not args.non_strict
    for role, model, checkpoint in (
        ("teacher", pair.teacher, args.teacher_checkpoint),
        ("student", pair.student, args.student_checkpoint),
    ):
        if not checkpoint:
            continue
        report = load_checkpoint(
            model,
            checkpoint,
            strict=strict,
            allow_unsafe=args.allow_unsafe_checkpoint,
        )
        checkpoint_reports[role] = {
            "path": str(report.path),
            "state_key": report.state_key,
            "missing_keys": list(report.missing_keys),
            "unexpected_keys": list(report.unexpected_keys),
            "exact_match": report.exact_match,
        }

    artifact_paths = None
    if args.experiment:
        artifact_paths = resolve_artifact_paths(
            args.experiment,
            protocol=args.protocol,
            output_root=args.output_root,
        )
        if args.create_artifact_dirs:
            artifact_paths.create()

    summary = {
        "teacher": {
            "name": pair.teacher_spec.name,
            "class": type(pair.teacher).__name__,
            "parameters": _parameter_counts(pair.teacher),
            "training": pair.teacher.training,
        },
        "student": {
            "name": pair.student_spec.name,
            "class": type(pair.student).__name__,
            "parameters": _parameter_counts(pair.student),
            "training": pair.student.training,
        },
        "checkpoints": checkpoint_reports,
    }
    if artifact_paths is not None:
        summary["artifacts"] = {
            "checkpoint_dir": str(artifact_paths.checkpoint_dir),
            "log_dir": str(artifact_paths.log_dir),
            "evaluation_dir": str(artifact_paths.evaluation_dir),
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
