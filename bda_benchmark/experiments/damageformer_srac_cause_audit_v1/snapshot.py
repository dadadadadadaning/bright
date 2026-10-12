"""Copy one atomically saved parent front to a new, diagnostic-only directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from .support import digest, validate_snapshot_payload


def create_snapshot(source, output):
    """Read one source handle, verify the local copy, then publish its manifest.

    The parent training writer uses atomic replacement. Keeping its source file
    handle open across the whole copy selects one saved version even if a later
    checkpoint replaces the pathname. The referenced A1 file is never copied.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "front_snapshot.pth"
    temporary = output / "front_snapshot.pth.partial"
    checksum = hashlib.sha256()
    copied = 0
    with source.open("rb") as reader, temporary.open("xb") as writer:
        before = os.fstat(reader.fileno())
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            checksum.update(block)
            writer.write(block)
            copied += len(block)
        writer.flush()
        os.fsync(writer.fileno())
        after = os.fstat(reader.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or copied != before.st_size:
        raise ValueError("Source was modified in place while copying; no successful snapshot manifest written")
    snapshot_sha256 = checksum.hexdigest()
    if digest(temporary) != snapshot_sha256:
        raise ValueError("Snapshot copy checksum mismatch")
    # Parse and validate before publishing either a final .pth or success manifest.
    payload = torch.load(temporary, map_location="cpu", weights_only=False)
    validate_snapshot_payload(payload)
    manifest = dict(
        format="SRAC_CAUSE_DIAGNOSTIC_SNAPSHOT_V1",
        complete=True,
        diagnostic_only=True,
        snapshot_file=destination.name,
        snapshot_sha256=snapshot_sha256,
        source=str(source),
        source_sha256=snapshot_sha256,
        source_bytes=copied,
        source_mtime_ns=before.st_mtime_ns,
        copied_at_utc=datetime.now(timezone.utc).isoformat(),
        front_format=payload["format"],
        source_hash=payload["source_hash"],
        a1_sha256=payload["a1_sha256"],
        a1_copied=False,
        step=payload["step"],
        declared_steps=payload["arguments"]["steps"],
        run_complete=bool(payload.get("run_complete", False)),
        completed_steps=payload.get("completed_steps"),
        smoke=payload["smoke"],
        valid_miou=payload.get("valid_miou"),
        test_accessed=False,
        scope="Read-only cause diagnosis; does not satisfy or modify parent completion gates",
    )
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary.rename(destination)
    with (output / "snapshot.json").open("x", encoding="utf-8") as handle:
        handle.write(manifest_text)
    return manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", required=True, help="Actual parent front/best.pth")
    parser.add_argument("--output", required=True, help="A new directory; existing paths are rejected")
    args = parser.parse_args()
    result = create_snapshot(args.source, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
