"""Find a unique completed old Val450 region report bound to this E0."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def find_reference(protocol, root=None):
    path = Path(protocol)
    path = path / "protocol.json" if path.is_dir() else path
    path = path.resolve()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    search = Path(root) if root else path.parent.parent
    matches = []
    for summary in sorted(search.rglob("summary.json")):
        try:
            s = json.loads(summary.read_text(encoding="utf-8"))
            r = json.loads((summary.parent / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        checks = r.get("immutable", {})
        if (s.get("complete") is True and s.get("full_validation_evidence") is True
                and r.get("format") == "SRAC_REGION_VAL_FORMAL_V1"
                and r.get("complete") is True and r.get("full_validation_evidence") is True
                and r.get("image_count") == 450 and r.get("test_accessed") is False
                and r.get("protocol_sha256") == sha
                and r.get("reproduction", {}).get("passed") is True
                and checks and all(v is True for v in checks.values())
                and r.get("native_a1_replay_ok") is True):
            matches.append(summary.parent.resolve())
    if len(matches) != 1:
        listing = "\n".join(str(p) for p in matches) or "(none)"
        raise ValueError(f"Expected one completed matching Val report under {search}; found {len(matches)}. "
                         "Set REGION_VAL_OUTPUT to the desired completed directory.\n" + listing)
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--root")
    args = parser.parse_args()
    print(find_reference(args.protocol, args.root))


if __name__ == "__main__":
    main()
