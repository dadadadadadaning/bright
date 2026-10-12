"""Read-only summaries, including same-step Q comparisons (not pooled bests)."""
import argparse
import json
from pathlib import Path


def read_rows(path):
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                print("Warning: incomplete log row ignored")
    return result


def scores(row):
    return {name: value["valid_miou"] for name, value in row.get("metrics", {}).items()
            if isinstance(value, dict) and "valid_miou" in value}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", required=True, help="front or joint output directory")
    p.add_argument("--reference", help="optional unchanged parent front directory")
    args = p.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit("Output does not exist: " + str(root))
    for filename in ("manifest.json", "summary.json"):
        path = root / filename
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if filename == "summary.json":
                print("Summary:", json.dumps(data, indent=2))
            else:
                print("smoke:", data.get("smoke"), "arguments:", data.get("arguments"))
    rows = read_rows(root / "validation.jsonl")
    for row in rows:
        print("step", row["step"], {k: round(v * 100, 4) for k, v in scores(row).items()})
    if rows:
        final = rows[-1]
        print("Last full metrics (fraction units):", json.dumps(final.get("metrics"), indent=2))
        print("Compare j3-j2 (search metric), j2-j1 (Memory), j1-j0 (region context).")
        print("Read retrieval diagnostics separately; kNN is NOT full-image mIoU.")
    if args.reference:
        other = {r["step"]: r for r in read_rows(Path(args.reference) / "validation.jsonl")}
        print("Q paired checkpoints: new final minus reference final, pp")
        for row in rows:
            if row["step"] in other:
                first, second = scores(row), scores(other[row["step"]])
                if "final" in first and "final" in second:
                    print(row["step"], round(100 * (first["final"] - second["final"]), 4))
        print("Check matching A1 hash, data, seed, optimizer updates, batch/accum and initialization before claiming a causal gain.")


if __name__ == "__main__":
    main()
