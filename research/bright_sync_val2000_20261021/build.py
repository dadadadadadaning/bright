"""Pin all inherited checkpoint/evidence helpers to the tested index diagnostic."""
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
BASE_SHA = "cc639347f9c07409c10721fa8b12728219ec5d6893d98b97b238bbd6dfb3a164"


def build():
    base = HERE.parent / "bright_index_eval_probe_20261021/bright_index_eval2000_v1.pyz"
    if hashlib.sha256(base.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError("Pinned index diagnostic archive differs")
    with zipfile.ZipFile(base) as archive:
        files = {n: archive.read(n) for n in archive.namelist() if n != "__main__.py"}
    ledger = {n: hashlib.sha256(data).hexdigest() for n, data in files.items()}
    for name in ("sync_val2000.py", "evaluation_observer.py"):
        files[name] = (HERE / name).read_bytes()
    files["SYNC_VAL_BASE.json"] = json.dumps(dict(source_sha256=BASE_SHA, files=ledger), indent=2).encode()
    files["__main__.py"] = b"from sync_val2000 import main\nraise SystemExit(main())\n"
    output = HERE / "bright_sync_val2000_v1.pyz"
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
