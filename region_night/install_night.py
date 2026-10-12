#!/usr/bin/env python3
"""Fetch this experiment's files at ONE exact GitHub commit into a new folder."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import urllib.request

FILES = frozenset(("night_run.py", "region_residual.py", "residual_core.py", "residual_eval.py",
                   "test_night_run.py", "test_region_residual.py", "test_residual_core.py",
                   "test_residual_eval.py", "install_night.py", "test_install_night.py", "README.md"))
REPO = "https://raw.githubusercontent.com/dadadadadadaning/bright/"


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def install(ref, destination, fetcher=fetch):
    if re.fullmatch(r"[0-9a-f]{40}", ref) is None:
        raise ValueError("Use the exact 40-character commit from the instructions")
    base = REPO + ref + "/region_night/"
    ledger_bytes = fetcher(base + "SHA256SUMS.json")
    ledger = json.loads(ledger_bytes)
    if (not isinstance(ledger, dict) or set(ledger) != FILES
            or any(not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None
                   for v in ledger.values())):
        raise ValueError("Unexpected filenames or hashes in release manifest")
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        matching = destination.is_dir() and all(
            (destination / name).is_file()
            and hashlib.sha256((destination / name).read_bytes()).hexdigest() == value
            for name, value in ledger.items())
        if matching:
            try:
                matching = json.loads((destination / "release.json").read_text()).get("commit") == ref
            except (OSError, ValueError):
                matching = False
        if matching:
            print("Already installed unchanged: " + str(destination), flush=True)
            return destination
        raise FileExistsError("Destination contains different files; choose a NEW --dest: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bright_night_download_", dir=destination.parent) as temporary:
        staging = Path(temporary) / "files"
        staging.mkdir()
        for name in sorted(FILES):
            data = fetcher(base + name)
            if hashlib.sha256(data).hexdigest() != ledger[name]:
                raise ValueError("Download checksum mismatch: " + name)
            (staging / name).write_bytes(data)
        (staging / "SHA256SUMS.json").write_bytes(ledger_bytes)
        (staging / "release.json").write_text(json.dumps(dict(commit=ref, source=base), indent=2))
        staging.rename(destination)
    print("Installed verified experiment files: " + str(destination), flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--dest", default="/root/shared-nvme/BRIGHT-region-night-tools_20261012")
    args = parser.parse_args()
    install(args.ref, args.dest)


if __name__ == "__main__":
    main()
