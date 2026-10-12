from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .utility import UtilityVerifier

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cal", required=True, help="pt dict with features [N,D], targets [N]")
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    obj = torch.load(args.train_cal, map_location="cpu", weights_only=False)
    x,y = obj["features"].float(), obj["targets"].float()
    loader = DataLoader(TensorDataset(x,y),batch_size=args.batch_size,shuffle=True)
    it = iter(loader)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UtilityVerifier(in_dim=x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best = float("inf")
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    for step in range(1,args.steps+1):
        try: xb,yb = next(it)
        except StopIteration:
            it = iter(loader); xb,yb = next(it)
        xb,yb = xb.to(device),yb.to(device)
        loss = F.binary_cross_entropy_with_logits(model(xb),yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step==1 or step%100==0: print({"step":step,"loss":float(loss)},flush=True)
        if float(loss)<best:
            best=float(loss)
            torch.save({"state_dict":model.state_dict(),"in_dim":x.shape[1],"step":step,"best_train_loss":best,"seed":args.seed},args.output)
    print("Saved:",args.output)

if __name__ == "__main__": main()
