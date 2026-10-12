from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--eval-root",required=True)
    args=ap.parse_args()
    root=Path(args.eval_root)
    files=sorted(root.glob("E3_*.json"))
    if not files:
        print("No E3 json files found."); return
    print("=== E3 causal comparison (fixed 2% coverage) ===")
    for p in files:
        d=json.loads(p.read_text(encoding="utf-8"))
        print(f"\n[{p.name}] control={d.get('control')}")
        for k,v in d["results"].items():
            print(
                f"{k:42s} "
                f"valid={100*v['metrics']['valid_miou']:.4f}% "
                f"gain={v['delta_valid_miou_pp']:+.4f}pp "
                f"D-IoU={100*v['metrics']['damaged_iou']:.4f}% "
                f"D->I={100*v['metrics']['damaged_to_intact']:.2f}%"
            )
        c=d.get("causal_checks",{})
        print(" causal:",json.dumps(c,ensure_ascii=False))
    u=root/"E4_utility.json"
    if u.exists():
        d=json.loads(u.read_text(encoding="utf-8"))["result"]
        print("\n=== E4 Utility ===")
        print(
            f"valid={100*d['metrics']['valid_miou']:.4f}% "
            f"gain={d['delta_valid_miou_pp']:+.4f}pp "
            f"D-IoU={100*d['metrics']['damaged_iou']:.4f}%"
        )

if __name__=="__main__":
    main()
