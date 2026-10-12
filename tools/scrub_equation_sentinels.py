from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path


def scrub(docx_path: Path) -> tuple[int, int]:
    temp_path = docx_path.with_suffix(".sentinel-clean.docx")
    sentinel = "¤".encode("utf-8")
    placeholders_before = 0
    removed = 0

    with zipfile.ZipFile(docx_path, "r") as source, zipfile.ZipFile(
        temp_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "word/document.xml":
                placeholders_before = data.count(b"[[EQ")
                removed = data.count(sentinel)
                data = data.replace(sentinel, b"")
            target.writestr(item, data)

    os.replace(temp_path, docx_path)
    return placeholders_before, removed


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: scrub_equation_sentinels.py <document.docx>")
    path = Path(sys.argv[1]).resolve()
    placeholders, removed = scrub(path)
    print(f"placeholders={placeholders}")
    print(f"sentinels_removed={removed}")


if __name__ == "__main__":
    main()
