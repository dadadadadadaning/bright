"""Portable project and artifact paths for BRIGHT scripts."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


BDA_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BDA_ROOT.parent
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ArtifactPaths:
    """Canonical locations for one experiment."""

    experiment: str
    protocol: str
    checkpoint_dir: Path
    log_dir: Path
    evaluation_dir: Path

    def create(self) -> "ArtifactPaths":
        for directory in (
            self.checkpoint_dir,
            self.log_dir,
            self.evaluation_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def ensure_import_paths() -> None:
    """Make both new package imports and legacy ``model.*`` imports available."""

    for path in (PROJECT_ROOT, BDA_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _validate_segment(value: str, label: str) -> str:
    value = value.strip()
    if not _SAFE_NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            f"{label} must be a single safe path segment, got {value!r}"
        )
    return value


def resolve_artifact_paths(
    experiment: str,
    *,
    protocol: str = "distillation",
    output_root: str | Path | None = None,
) -> ArtifactPaths:
    """Resolve checkpoint/log/evaluation directories without creating them.

    ``BRIGHT_OUTPUT_ROOT`` may relocate all generated artifacts while preserving
    the same internal layout.
    """

    experiment = _validate_segment(experiment, "experiment")
    protocol = _validate_segment(protocol, "protocol")
    configured_root = output_root or os.environ.get("BRIGHT_OUTPUT_ROOT")
    root = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else PROJECT_ROOT
    )
    return ArtifactPaths(
        experiment=experiment,
        protocol=protocol,
        checkpoint_dir=root / "saved_weights" / protocol / experiment,
        log_dir=root / "logs" / protocol / experiment,
        evaluation_dir=root / "eval_results_official" / protocol / experiment,
    )

