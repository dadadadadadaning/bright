from pathlib import Path
from pypdf import PdfReader

ROOT = Path(r"D:\software\dowmload\BRIGHT-master\research-paper-projects\mcuv-background\sources")
PAPERS = {
    Path(r"D:\software\文献\A01-BRIGHT-2025.pdf"): ROOT / "A01-BRIGHT-2025.txt",
    Path(r"D:\software\文献\A09-BRIGHT-Challenge-2026.pdf"): ROOT / "A09-BRIGHT-Challenge-2026.txt",
}

for pdf_path, text_path in PAPERS.items():
    reader = PdfReader(pdf_path)
    chunks = []
    for page_no, page in enumerate(reader.pages, start=1):
        chunks.append(f"\n===== PDF PAGE {page_no} =====\n")
        chunks.append(page.extract_text() or "[NO EXTRACTABLE TEXT]")
    text_path.write_text("\n".join(chunks), encoding="utf-8")
    print(f"{pdf_path.name}: {len(reader.pages)} pages -> {text_path}")
