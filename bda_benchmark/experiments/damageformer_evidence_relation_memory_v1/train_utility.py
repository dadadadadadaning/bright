from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F

from .cache import load_stage1_cache
from .utility import UtilityVerifier, utility_features, utility_targets, split_rows_by_image, normalize_fit
from .metrics import average_precision

@torch.no_grad()
def _score(model,x,device):
    model.eval(); out=[]
    for s in range(0,len(x),4096):
        sign,delta=model(x[s:s+4096].to(device))
        # Primary ranking = predicted exact delta; sign is auxiliary regularization.
        out.append(delta.float().cpu())
    return torch.cat(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-cache",required=True)
    ap.add_argument("--val-cache",required=True)
    ap.add_argument("--specialist-train-scores",required=True)
    ap.add_argument("--specialist-val-scores",required=True)
    ap.add_argument("--history",required=True)
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--steps",type=int,default=2000)
    ap.add_argument("--seed",type=int,default=3407)
    args=ap.parse_args()

    torch.manual_seed(args.seed)
    tr=load_stage1_cache(args.train_cache,"train")
    va=load_stage1_cache(args.val_cache,"val")
    st=torch.load(args.specialist_train_scores,map_location="cpu",weights_only=False)["scores"].float()
    sv=torch.load(args.specialist_val_scores,map_location="cpu",weights_only=False)["scores"].float()
    hist=torch.load(args.history,map_location="cpu",weights_only=False)
    if hist.get("control","none")!="none":
        raise RuntimeError("Utility must train on real historical evidence, not a control.")

    cal_rows=hist["cal"]["rows"].long()
    val_rows=hist["val"]["rows"].long()
    x=utility_features(tr,cal_rows,st,hist["cal"])
    target=utility_targets(tr,cal_rows)
    sign=(target>0).float()

    util_fit,util_dev=split_rows_by_image(tr,cal_rows,args.seed)
    xn,mean,std=normalize_fit(x,util_fit)
    target_mean=target[util_fit].mean()
    target_std=target[util_fit].std().clamp_min(1e-5)
    tz=(target-target_mean)/target_std

    device="cuda" if torch.cuda.is_available() else "cpu"
    model=UtilityVerifier(x.shape[1]).to(device)
    pos=float(sign[util_fit].sum()); neg=float(util_fit.sum()-sign[util_fit].sum())
    pos_weight=torch.tensor([neg/max(1.0,pos)],device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    fit_idx=torch.where(util_fit)[0]
    g=torch.Generator().manual_seed(args.seed)
    best_ap=-1.0

    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    for step in range(1,args.steps+1):
        take=fit_idx[torch.randint(0,len(fit_idx),(min(512,len(fit_idx)),),generator=g)]
        xb=xn[take].to(device); yb=sign[take].to(device); db=tz[take].to(device)
        slog,dpred=model(xb)
        loss_sign=F.binary_cross_entropy_with_logits(slog,yb,pos_weight=pos_weight)
        loss_delta=F.smooth_l1_loss(dpred,db)
        loss=loss_sign+0.5*loss_delta
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        if step%100==0 or step==args.steps:
            with torch.no_grad():
                slog,dp=model(xn[util_dev].to(device))
                dev_score=dp.float().cpu()
                apv=average_precision(sign[util_dev].long(),dev_score)
                mae=float((dp.cpu()-tz[util_dev]).abs().mean())
            print(json.dumps({"step":step,"loss":float(loss),"dev_ap":apv,"dev_mae_z":mae}),flush=True)
            if apv>best_ap:
                best_ap=apv
                torch.save({
                    "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    "in_dim":x.shape[1],
                    "feature_mean":mean,"feature_std":std,
                    "target_mean":target_mean,"target_std":target_std,
                    "step":step,"dev_ap":apv,
                },outdir/"best.pth")

    ck=torch.load(outdir/"best.pth",map_location="cpu",weights_only=False)
    model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    xv=utility_features(va,val_rows,sv,hist["val"])
    xvn=(xv-ck["feature_mean"])/ck["feature_std"]
    score_z=_score(model,xvn,device)
    score_pp=score_z*ck["target_std"]+ck["target_mean"]
    torch.save({
        "protocol":"E4_UTILITY_VERIFIER_V1",
        "rows":val_rows,
        "predicted_delta_pp":score_pp,
        "best_step":ck["step"],"dev_ap":ck["dev_ap"],
        "test_accessed":False,
    },outdir/"val_utility_scores.pt")
    print(json.dumps({"saved":str(outdir/"val_utility_scores.pt"),"best_step":ck["step"],"dev_ap":ck["dev_ap"]},indent=2))

if __name__=="__main__":
    main()
