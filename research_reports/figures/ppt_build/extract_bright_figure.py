from pathlib import Path
from pypdf import PdfReader
root = Path('D:/software/dowmload/BRIGHT-master/research_reports/figures/pdf_assets')
root.mkdir(exist_ok=True)
r = PdfReader('D:/software/文献/A01-BRIGHT-2025.pdf')
im = r.pages[11].images[0]
(root/'BRIGHT_Fig6_original.jpg').write_bytes(im.data)
print(im.image.size)
