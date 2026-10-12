#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from bda_benchmark.dataset.make_data_loader import MultimodalDamageAssessmentDatset
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.core import load_a1_parent,seed_worker
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.model import DamageFormerA1CrossModalSuperpixel
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.train import evaluate,format_metrics,read_list

def main():
    p=argparse.ArgumentParser(); p.add_argument('--a1_ckpt',required=True); p.add_argument('--dataset_path',required=True); p.add_argument('--val_list',required=True)
    p.add_argument('--region_stride',type=int,default=8); p.add_argument('--batch_size',type=int,default=2); p.add_argument('--num_workers',type=int,default=8)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--output_json',required=True); a=p.parse_args()
    if 'test' in Path(a.val_list).name.lower(): raise RuntimeError('TEST ACCESS IS FORBIDDEN')
    device=torch.device(a.device if torch.cuda.is_available() else 'cpu'); model=DamageFormerA1CrossModalSuperpixel('c5_aligned_assoc',a.region_stride)
    rep=load_a1_parent(model,a.a1_ckpt); model.to(device).eval(); ds=MultimodalDamageAssessmentDatset(a.dataset_path,read_list(a.val_list),640,type='val')
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=device.type=='cuda',worker_init_fn=seed_worker,persistent_workers=a.num_workers>0)
    m=evaluate(model,loader,device,255,0); out={'parent_report':rep,'metrics':m,'region_gain':float(torch.tanh(model.cmsp.region_gain).detach().cpu()),
      'lca_gain':float(torch.tanh(model.cmsp.lca.gain).detach().cpu()),'test_accessed':False}
    Path(a.output_json).parent.mkdir(parents=True,exist_ok=True); Path(a.output_json).write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Exact replay | '+format_metrics(m)); print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
