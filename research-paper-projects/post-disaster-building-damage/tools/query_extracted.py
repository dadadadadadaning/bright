from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PAGE_RE = re.compile(r"===== PDF_PAGE_(\d{3}) =====")


def load_pages(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    parts = PAGE_RE.split(text)
    pages: list[str] = []
    for index in range(1, len(parts), 2):
        page_no = int(parts[index])
        while len(pages) < page_no:
            pages.append("")
        pages[page_no - 1] = parts[index + 1].strip()
    return pages


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--pages", default="")
    parser.add_argument("--pattern", default="")
    parser.add_argument("--context", type=int, default=260)
    parser.add_argument("--max-hits", type=int, default=30)
    parser.add_argument("--outline", action="store_true")
    args = parser.parse_args()

    pages = load_pages(args.path)
    if args.pages:
        requested = []
        for token in args.pages.split(","):
            if "-" in token:
                start, end = (int(x) for x in token.split("-", 1))
                requested.extend(range(start, end + 1))
            else:
                requested.append(int(token))
        for page_no in requested:
            print(f"\n===== PDF PAGE {page_no} =====\n")
            print(pages[page_no - 1])

    if args.outline:
        section_re = re.compile(
            r"^(?:\d+(?:\.\d+)*\.?\s+|[A-Z]\.?\s+)?"
            r"(?:abstract|introduction|related work|method(?:ology)?|approach|"
            r"experiments?|results?|discussion|limitations?|conclusion|"
            r"appendix|dataset|implementation details|ablation(?: study)?|"
            r"loss(?: function)?|training)[^\n]{0,100}$",
            re.IGNORECASE,
        )
        caption_re = re.compile(r"^(?:Fig(?:ure)?\.?|Table)\s*\d+[^\n]{0,220}$", re.I)
        for page_no, page in enumerate(pages, start=1):
            for line in page.splitlines():
                line = compact(line)
                if section_re.match(line) or caption_re.match(line):
                    print(f"p{page_no:02d}\t{line}")

    if args.pattern:
        pattern = re.compile(args.pattern, re.IGNORECASE)
        hits = 0
        for page_no, page in enumerate(pages, start=1):
            for match in pattern.finditer(page):
                start = max(0, match.start() - args.context)
                end = min(len(page), match.end() + args.context)
                print(f"\n--- p{page_no} ---\n{compact(page[start:end])}")
                hits += 1
                if hits >= args.max_hits:
                    return


if __name__ == "__main__":
    main()
