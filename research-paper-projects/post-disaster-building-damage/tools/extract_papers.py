from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pypdf import PdfReader


def safe_stem(path: Path) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", path.stem).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for pdf_path in sorted(args.input_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        metadata = reader.metadata or {}
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(
                f"\n\n===== PDF_PAGE_{page_number:03d} =====\n\n{text.strip()}\n"
            )
        output_path = args.output_dir / f"{safe_stem(pdf_path)}.txt"
        output_path.write_text("".join(pages), encoding="utf-8")
        records.append(
            {
                "file_name": pdf_path.name,
                "path": str(pdf_path),
                "page_count": len(reader.pages),
                "title": metadata.get("/Title"),
                "author": metadata.get("/Author"),
                "subject": metadata.get("/Subject"),
                "extracted_text": str(output_path),
            }
        )
        print(f"{pdf_path.name}\t{len(reader.pages)}\t{output_path.name}")

    (args.output_dir / "manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
