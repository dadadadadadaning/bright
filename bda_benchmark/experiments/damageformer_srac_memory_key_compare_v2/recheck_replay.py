"""Read one saved token replay evidence on CPU; do not run the model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    import torch
    from .replay import stored_token_replay
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="replay_evidence/NNNNN.pt saved by V2")
    parser.add_argument("--sha256", help="Expected hash from the source summary/images journal")
    args = parser.parse_args()
    path = Path(args.input)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.sha256 and args.sha256 != digest:
        raise ValueError("Replay evidence checksum differs")
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record.get("format") != "SRAC_MEMORY_KEY_REPLAY_EVIDENCE_V2":
        raise ValueError("Expected V2 saved replay evidence")
    report = stored_token_replay(record["replayed_fp32"], record["archived_tokens"])
    output = dict(image=record["image"], evidence_sha256=digest,
                  protocol_sha256=record["protocol_sha256"],
                  original_bank_sha256=record["original_bank_sha256"], report=report,
                  interpretation="Storage compatibility is not exact historical FP32 reproduction; no model ran")
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
