"""Build the standalone evaluation launcher deterministically (stdlib only)."""
from pathlib import Path
import hashlib
import zipfile

root = Path(__file__).resolve().parent
output = root / "bright_checkpoint_val450_v1.pyz"
with zipfile.ZipFile(output, "w") as archive:
    files = [("__main__.py", b"from evaluate_checkpoint import main\nraise SystemExit(main())\n")]
    files += [(name, (root / name).read_bytes()) for name in
              ("evaluate_checkpoint.py", "grid_probe.py", "EVALUATION_ONLY.md")]
    for name, content in files:
        info = zipfile.ZipInfo(name, date_time=(2026, 10, 21, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, content)
digest = hashlib.sha256(output.read_bytes()).hexdigest()
output.with_suffix(".pyz.sha256").write_text(digest + "  " + output.name + "\n", encoding="ascii")
print(digest, output.name, output.stat().st_size)
