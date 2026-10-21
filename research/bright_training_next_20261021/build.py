#!/usr/bin/env python3
"""Bundle the byte-identical tested resume helpers plus the scoped extension."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

BASE_SHA = "0f7fd0a85ca2db5727509875c84c1beba1011d2bd4300a8b6b5ea86b59e75f07"
HERE = Path(__file__).resolve().parent


def build(base):
    base = Path(base)
    if hashlib.sha256(base.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError("Require the exact fixed NLL resume archive")
    output = HERE / "bright_training_next_v1.pyz"
    with zipfile.ZipFile(base) as source:
        files = {n: source.read(n) for n in source.namelist() if n != "__main__.py"}
    ledger = {n: hashlib.sha256(data).hexdigest() for n, data in files.items()}
    for name in ("training_entry.py", "training_common.py", "no_read_baseline.py"):
        files[name] = (HERE / name).read_bytes()
    observation = HERE.parent / "bright_grid_probability_probe_20261021"
    for name in ("grid_probe.py", "evaluate_checkpoint.py"):
        files[name] = (observation / name).read_bytes()
    files["BASE_HELPERS.json"] = json.dumps(dict(source_sha256=BASE_SHA, files=ledger), indent=2).encode()
    files["__main__.py"] = b"from training_entry import main\nraise SystemExit(main())\n"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".pyz.sha256").write_text(digest + "  " + output.name + "\n")
    print(digest, output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base_archive")
    build(p.parse_args().base_archive)
