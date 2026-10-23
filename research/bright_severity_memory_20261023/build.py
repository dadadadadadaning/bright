"""Build a small deterministic source ZIP; no weights or training data."""
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent


def main():
    files = sorted(p for p in HERE.iterdir() if p.suffix in (".py", ".md"))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (HERE / "SHA256SUMS.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    files.append(HERE / "SHA256SUMS.json")
    path = HERE.parent / (HERE.name + ".zip")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for item in sorted(files):
            info = zipfile.ZipInfo(HERE.name + "/" + item.name, date_time=(2026, 10, 23, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, item.read_bytes())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".zip.sha256").write_text(digest + "  " + path.name + "\n")
    print(json.dumps(dict(path=str(path), bytes=path.stat().st_size, sha256=digest)))


if __name__ == "__main__":
    main()
