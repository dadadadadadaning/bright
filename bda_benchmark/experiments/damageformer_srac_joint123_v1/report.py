"""Read-only progress for the exactly-three-head comparison."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", required=True, help="J123_ROOT/joint output directory")
    args = parser.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit("No joint output yet; the metric/probe stage may still be running: " + str(root))
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    print("complete:", summary.get("complete", False), "probe:", summary.get("probe", False))
    rows = []
    path = root / "validation.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                if line.strip():
                    rows.append(json.loads(line))
            except json.JSONDecodeError:
                print("Incomplete last log line ignored")
    for row in rows:
        metrics = row["metrics"]
        print("step", row["step"], {name: round(100 * metrics[name]["valid_miou"], 4)
                                    for name in ("a1_pixel", "j1", "j2", "j3")})
    if rows:
        final = rows[-1]
        m = final["metrics"]
        print("Last same-step differences (pp):",
              {"j2-j1": 100 * (m["j2"]["valid_miou"] - m["j1"]["valid_miou"]),
               "j3-j2": 100 * (m["j3"]["valid_miou"] - m["j2"]["valid_miou"]),
               "j3-j1": 100 * (m["j3"]["valid_miou"] - m["j1"]["valid_miou"])})
        print("Last per-class metrics:", json.dumps({name: m[name] for name in ("j1", "j2", "j3")}, indent=2))
        print("Retrieval (not mIoU):", json.dumps(final.get("retrieval"), indent=2))
        print("Damaged J3 vs J2:", json.dumps(final.get("damaged_j3_vs_j2"), indent=2))
    else:
        print("No validation yet; inspect the unified log for artifact validation / initial A1 check / training")


if __name__ == "__main__":
    main()
