"""Reproducible zipapp build; no runtime dependencies copied into the artifact."""
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'bright_calibration_resume_v1.pyz'
with zipfile.ZipFile(OUTPUT, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for name in ('__main__.py', 'launch.py', 'helpers.py', 'preflight.py', 'resume.py', 'lean_adapter.py'):
        info = zipfile.ZipInfo(name, (2026, 10, 24, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, (ROOT / name).read_bytes())
digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
OUTPUT.with_suffix('.pyz.sha256').write_text(digest+'  '+OUTPUT.name+'\n')
print(digest, OUTPUT.stat().st_size)
