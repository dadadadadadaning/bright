from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .cache import load_stage1_cache, stable_image_split, c1_features, candidate_mask
from .config import DEFAULT_PROTOCOL
from .metrics import average_precision, binary_auroc

class C1Specialist(nn.Module):
    """Self-contained current-evidence specialist.

    Input = base I/D/X probs + uncertainty + RegionMean[128].
    This is a NEW reproducible specialist for the relation pipeline.
    If you need exact legacy C1 scores, export them and use external score mode.
    """
    def __init__(self, in_dim=132, h1=256, h2=128, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.GELU(),
            nn.Linear(h2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def _train_rows(cache, split_mask, min_purity):
    y = cache["labels"].long()
    purity = cache["label_purity"].float()
    # Match legacy supervision domain: clean I/D on Train only.
    return split_mask & (purity >= min_purity) & ((y == 0) | (y == 1))

def _cal_candidate_rows(cache, cal_mask):
    y = cache["labels"].long()
    purity = cache["label_purity"].float()
    return cal_mask & candidate_mask(cache) & (purity >= 0.70) & ((y == 0) | (y == 1))

@torch.no_grad()
def score_cache(model, cache, batch_size=4096, device="cuda"):
    x = c1_features(cache)
    out = torch.empty(len(x), dtype=torch.float32)
    model.eval()
    for s in range(0, len(x), batch_size):
        xb = x[s:s+batch_size].to(device, non_blocking=True)
        out[s:s+len(xb)] = model(xb).float().cpu()
    return out

def train_specialist(train_cache, output_dir, seed=3407, steps=3000, batch_size=512, lr=1e-3):
    torch.manual_seed(seed)
    p = DEFAULT_PROTOCOL
    fit_mask, cal_mask = stable_image_split(train_cache, p.fit_ratio, seed)
    rows = _train_rows(train_cache, fit_mask, p.min_purity)
    x = c1_features(train_cache)[rows]
    y = (train_cache["labels"].long()[rows] == 1).float()

    ds = TensorDataset(x, y)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=False)
    it = iter(loader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = C1Specialist(in_dim=x.shape[1]).to(device)

    n_pos = float(y.sum()); n_neg = float(len(y) - y.sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)
    best_ap = -1.0; best = None

    for step in range(1, steps+1):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader); xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)
        loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        if step % 250 == 0 or step == steps:
            scores = score_cache(model, train_cache, device=device)
            cr = _cal_candidate_rows(train_cache, cal_mask)
            cy = (train_cache["labels"].long()[cr] == 1).long()
            cs = scores[cr]
            ap = average_precision(cy, cs)
            auc = binary_auroc(cy, cs)
            row = {"step":step, "loss":float(loss), "cal_ap":ap, "cal_auroc":auc}
            print(json.dumps(row), flush=True)
            if ap > best_ap:
                best_ap = ap
                best = {
                    "state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                    "step": step, "cal_ap": ap, "cal_auroc": auc,
                    "seed": seed, "in_dim": x.shape[1],
                    "fit_mask": fit_mask, "cal_mask": cal_mask,
                }
                torch.save(best, outdir/"best.pth")

    ckpt = torch.load(outdir/"best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"]); model.to(device)
    train_scores = score_cache(model, train_cache, device=device)
    torch.save({
        "protocol":"EVIDENCE_RELATION_C1_SPECIALIST",
        "scores":train_scores,
        "fit_mask":fit_mask, "cal_mask":cal_mask,
        "best_step":ckpt["step"], "cal_ap":ckpt["cal_ap"], "cal_auroc":ckpt["cal_auroc"],
    }, outdir/"train_scores.pt")
    return model, ckpt

def load_specialist(path, device="cuda"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = C1Specialist(in_dim=ckpt.get("in_dim",132))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    tr = load_stage1_cache(args.train_cache, "train")
    va = load_stage1_cache(args.val_cache, "val")
    model, ckpt = train_specialist(tr, args.output_dir, args.seed, args.steps)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_scores = score_cache(model, va, device=device)
    torch.save({
        "protocol":"EVIDENCE_RELATION_C1_SPECIALIST_VAL",
        "scores":val_scores,
        "best_step":ckpt["step"], "cal_ap":ckpt["cal_ap"], "cal_auroc":ckpt["cal_auroc"],
        "test_accessed":False,
    }, Path(args.output_dir)/"val_scores.pt")
    print(json.dumps({
        "best_step":ckpt["step"], "cal_ap":ckpt["cal_ap"], "cal_auroc":ckpt["cal_auroc"],
        "val_scores":str(Path(args.output_dir)/"val_scores.pt"),
    }, indent=2))

if __name__ == "__main__":
    main()
