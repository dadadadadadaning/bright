import torch
from .schema import MemoryBank
from .memory_retriever import build_global_keys,query_memory
from .utility import exact_utility_delta

def main():
    torch.manual_seed(1); n,k,d=12,5,16
    rt=torch.randn(n,k,d); rm=torch.ones(n,k,dtype=torch.bool); meta=torch.zeros(n,k,4)
    meta[...,0]=1/k; meta[...,1]=.8; meta[...,2]=.2; meta[...,3]=torch.rand(n,k)*.2
    bank=MemoryBank(build_global_keys(rt,rm),rt,rm,meta,torch.tensor([1,2,3,1,2,3,1,2,3,1,2,3]),torch.tensor([0,0,0,1,1,1,2,2,2,3,3,3]),torch.arange(n))
    bank.validate(); out=query_memory(bank.keys[0],rt[0],rm[0],meta[0],0,bank,coarse_k=8,fine_k=5)
    assert not torch.any(bank.event_ids[out["neighbor_indices"]]==0)
    g=torch.tensor([[100,2,1,0],[3,80,5,1],[1,20,30,2],[0,4,3,40]],dtype=torch.float32)
    b=torch.tensor([[0,0,0,0],[0,5,0,0],[0,4,3,0],[0,0,0,0]],dtype=torch.float32)
    delta=exact_utility_delta(g,b); assert torch.isfinite(delta)
    print("SMOKE PASS"); print("delta_DI =",float(out["delta_di"])); print("utility_delta =",float(delta))
if __name__ == "__main__": main()
