from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from .cache import load_stage1_cache,candidate_mask
from .metrics import evaluate_score
from .config import DEFAULT_PROTOCOL

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--val-cache",required=True)
    ap.add_argument("--utility-scores",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    va=load_stage1_cache(args.val_cache,"val")
    u=torch.load(args.utility_scores,map_location="cpu",weights_only=False)
    score=torch.full((len(va["labels"]),),-1e9)
    score[u["rows"].long()]=u["predicted_delta_pp"].float()
    out={
        "protocol":"E4_UTILITY_EVAL_V1",
        "result":evaluate_score(va,score,candidate_mask(va),DEFAULT_PROTOCOL.coverage,"E4_utility"),
        "test_accessed":False,
    }
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
