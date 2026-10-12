"""Install the already delivered, hash-pinned ZIP without renameat2."""
from pathlib import Path
import hashlib
import os
import zipfile

ARCHIVE = Path("/root/shared-nvme/bright_memory_key_compare_v1.zip")
REPO = Path("/root/shared-nvme/BRIGHT-master")
EXPECTED = "39d1cd66ff604d022d0384479834bcbf1d4aec2a586ede08be5a9f1042ae8b1d"


def install(archive=ARCHIVE, repo=REPO):
    archive, repo = Path(archive), Path(repo).resolve(strict=True)
    if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != EXPECTED:
        raise SystemExit("ZIP missing or checksum differs; no installation performed.")
    namespace = {"__name__": "verified_memory_installer"}
    with zipfile.ZipFile(archive) as zipped:
        exec(compile(zipped.read("INSTALL_SERVER.py"), "INSTALL_SERVER.py", "exec"), namespace)
    payload = namespace["read_verified_archive"](archive)
    parent = (repo / "bda_benchmark/experiments").resolve(strict=True)
    if repo not in parent.parents:
        raise SystemExit("Experiment parent resolves outside the repository.")
    target = parent / namespace["PACKAGE"]
    if os.path.lexists(target):
        namespace["_destination_identical"](target, payload)
        print("Existing code is identical; no files changed:", target)
        return target
    target.mkdir(mode=0o755, exist_ok=False)
    marker = target / ".install_in_progress"
    marker_bytes = b"Verified ZIP installation in progress; do not execute this directory.\n"
    try:
        with marker.open("xb") as stream:
            stream.write(marker_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        for name, data in payload.items():
            file = target / name
            with file.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
                os.fchmod(stream.fileno(), 0o755 if file.suffix == ".sh" else 0o644)
        namespace["_destination_identical"](target, {**payload, marker.name: marker_bytes})
        marker.unlink()
        print("Verified compatibility installation complete:", target)
        return target
    except BaseException:
        print("Installation incomplete; preserve directory and marker:", target)
        raise


if __name__ == "__main__":
    install()
