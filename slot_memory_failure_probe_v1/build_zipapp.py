"""Build the diagnostic app with deterministic metadata and explicit contents."""
from pathlib import Path
import hashlib
import zipfile


def main():
    source = Path(__file__).resolve().parent
    output = source.parent / "bright_slot_memory_failure_probe_v1.pyz"
    names = ("__main__.py", "probe.py", "nll_trace.py", "inspect_failure.py", "README.md")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (source / name).read_bytes())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".pyz.sha256").write_text(digest + "  " + output.name + "\n")
    print(digest, output.name, output.stat().st_size)


if __name__ == "__main__":
    main()
