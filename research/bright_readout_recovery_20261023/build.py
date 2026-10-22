"""Reproducible standard-library zipapp builder."""
import hashlib
from pathlib import Path
import zipfile
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'bright_readout_recovery_v1.pyz'
files = {name: (ROOT / name).read_bytes() for name in
         ('recovery.py', 'recovery_observer.py', 'compare_recovery.py')}
files['__main__.py'] = b'from recovery import main\nmain()\n'
with zipfile.ZipFile(OUTPUT, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for name, data in sorted(files.items()):
        info = zipfile.ZipInfo(name, (2026, 10, 23, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, data)
digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
OUTPUT.with_suffix('.pyz.sha256').write_text(digest+'  '+OUTPUT.name+'\n')
print(digest, OUTPUT.stat().st_size)
