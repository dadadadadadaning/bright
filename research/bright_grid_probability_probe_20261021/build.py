from pathlib import Path
import hashlib
import zipfile

root = Path(__file__).resolve().parent
output = root / "bright_grid_probability_probe_v1.pyz"
with zipfile.ZipFile(output, "w") as z:
    for name, content in [
        ("__main__.py", b"from grid_probe import main\nraise SystemExit(main())\n"),
        ("grid_probe.py", (root / "grid_probe.py").read_bytes()),
        ("README.md", (root / "README.md").read_bytes()),
    ]:
        info = zipfile.ZipInfo(name, date_time=(2026, 10, 21, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        z.writestr(info, content)
digest = hashlib.sha256(output.read_bytes()).hexdigest()
output.with_suffix(".pyz.sha256").write_text(digest + "  " + output.name + "\n")
print(digest, output.name, output.stat().st_size)
