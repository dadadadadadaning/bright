#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .core import write_json


def main() -> None:
    p = argparse.ArgumentParser("Select K using Validation DI AUROC")
    p.add_argument("--root", required=True)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--ks", default="1,2,4")
    args = p.parse_args()
    root = Path(args.root)
    rows = []
    for k in [int(x) for x in args.ks.split(",") if x.strip()]:
        d = root / f"k{k}_seed{args.seed}"
        s = d / "summary.json"
        c = d / "best.pth"
        if not s.exists() or not c.exists():
            continue
        data = json.loads(s.read_text(encoding="utf-8"))
        auc = float(data["metrics"]["di_auroc"])
        rows.append((auc, k, d, c))
    if not rows:
        raise RuntimeError("no completed multi-prototype runs found")
    rows.sort(reverse=True)
    auc, k, d, c = rows[0]
    dst = root / "best_tail.pth"
    shutil.copy2(c, dst)
    payload = {
        "selection_metric": "Validation DI AUROC",
        "selected_k": k,
        "selected_auroc": auc,
        "selected_dir": str(d),
        "checkpoint": str(dst),
        "candidates": [{"k": kk, "di_auroc": aa} for aa, kk, _, _ in rows],
        "test_accessed": False,
    }
    write_json(root / "best_selection.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
