# -*- coding: utf-8 -*-
"""Analyze class separation and cross-event kNN quality of an IARM memory."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--memory',required=True); p.add_argument('--topk',type=int,default=5); p.add_argument('--chunk_size',type=int,default=512); p.add_argument('--output_json',required=True); return p.parse_args()

def knn_predict(features,labels,events,k,chunk,cross_event):
    n=features.shape[0]; pred=[]; confidence=[]
    event_arr=np.asarray(events,dtype=object)
    for start in range(0,n,chunk):
        q=features[start:start+chunk]; sims=q@features.t()
        rows=torch.arange(q.shape[0]); cols=torch.arange(start,start+q.shape[0]); sims[rows,cols]=-1e4
        if cross_event:
            mask=torch.from_numpy(event_arr[None,:]==event_arr[start:start+q.shape[0],None]).to(sims.device)
            sims=sims.masked_fill(mask,-1e4)
        kk=min(k,n-1); values,idx=torch.topk(sims,k=kk,dim=1); neighbor_labels=labels[idx]
        votes=[]; conf=[]
        for row in range(neighbor_labels.shape[0]):
            counts=torch.bincount(neighbor_labels[row]-1,minlength=3).float(); cls=int(counts.argmax().item())+1; votes.append(cls); conf.append(float(counts.max().item()/kk))
        pred.extend(votes); confidence.extend(conf)
    return torch.tensor(pred),torch.tensor(confidence)

def main():
    a=parse_args(); m=torch.load(a.memory,map_location='cpu'); f=F.normalize(m['features'].float(),dim=1); y=m['labels'].long(); events=list(m['events']); hard=m.get('hard_to_intact',torch.zeros(len(y),dtype=torch.bool))
    centres=F.normalize(m['class_centres'].float(),dim=1); cosine=(centres@centres.t()).numpy()
    output={'memory_size':len(y),'class_counts':{str(c):int((y==c).sum()) for c in (1,2,3)},'centre_cosine':cosine.tolist()}
    for name,cross in [('standard',False),('cross_event',True)]:
        pred,conf=knn_predict(f,y,events,a.topk,a.chunk_size,cross); acc=float((pred==y).float().mean()); per={str(c):float((pred[y==c]==c).float().mean()) for c in (1,2,3)}; hard_acc=float((pred[hard]==y[hard]).float().mean()) if bool(hard.any()) else None
        output[name]={'accuracy':acc,'per_class_accuracy':per,'hard_to_intact_accuracy':hard_acc,'mean_vote_confidence':float(conf.mean())}
    path=Path(a.output_json); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(output,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
