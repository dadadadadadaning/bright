"""Build using byte-identical helpers from the recorded continuation artifact."""
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
BASE_SHA = "029137749200f9b4f4d0d5d26e4015dcb5b43a4c3a667fb616c97b8d4fed3570"


def build():
    base = HERE.parent / "bright_route_failure_resume_20261021/bright_route_failure_resume_v1.pyz"
    if hashlib.sha256(base.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError("Recorded continuation archive hash differs")
    with zipfile.ZipFile(base) as archive:
        files = {n: archive.read(n) for n in archive.namelist() if n != "__main__.py"}
    ledger = {n: hashlib.sha256(data).hexdigest() for n, data in files.items()}
    for name in ("index_observer.py", "index_eval_probe.py"):
        files[name] = (HERE / name).read_bytes()
    files["INDEX_PROBE_BASE.json"] = json.dumps(dict(source_sha256=BASE_SHA, files=ledger), indent=2).encode()
    files["__main__.py"] = b"from index_eval_probe import main\nraise SystemExit(main())\n"
    output = HERE / "bright_index_eval2000_v1.pyz"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".pyz.sha256").write_text(digest + "  " + output.name + "\n")
    print(digest, output)


if __name__ == "__main__":
    build()
