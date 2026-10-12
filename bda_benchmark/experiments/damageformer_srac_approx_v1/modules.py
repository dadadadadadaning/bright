"""Local soft superpixels, full-region pooling and labelled support attention."""
import torch
from torch import nn
from torch.nn import functional as F
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.association import (
    candidate_map, masked_local_softmax, soft_pool_unpool, normalized_coordinate_grid)


class RegionModule(nn.Module):
    def __init__(self, channels=256, dim=128, stride=8, learned=True, modulation=True, weighted=True):
        super().__init__()
        self.stride, self.learned, self.modulation, self.weighted = stride, learned, modulation, weighted
        self.association = nn.Sequential(nn.Conv2d(channels+2, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 9, 1))
        self.project = nn.Conv2d(channels, dim, 1)
        self.modulator = nn.Sequential(nn.Conv2d(channels+2, dim, 3, padding=1), nn.GELU(), nn.Conv2d(dim, dim, 1))
        self.score = nn.Conv2d(dim, 1, 1)
        self.main = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.skip = nn.Sequential(nn.LayerNorm(2*channels), nn.Linear(2*channels, dim))
        self.alpha = nn.Parameter(torch.zeros(()))
        self.classifier = nn.Linear(dim, 4)
        nn.init.zeros_(self.modulator[-1].weight); nn.init.zeros_(self.modulator[-1].bias)

    def forward(self, rgb, content, building):
        b,c,h,w = content.shape
        ids, valid, gh, gw = candidate_map(h,w,self.stride,content.device)
        xy = normalized_coordinate_grid(b,h,w,content)
        logits = self.association(torch.cat([rgb,xy],1))
        if not self.learned:
            logits = torch.full_like(logits, -1e4); logits[:,4] = 0
        q = masked_local_softmax(logits.float(), valid)
        hard = ids[None].expand(b,-1,-1).gather(1,q.argmax(1,keepdim=True)).reshape(b,h,w)
        edge = content.new_zeros(b,1,h,w)
        edge[:,:,1:] = (hard[:,1:] != hard[:,:-1]).unsqueeze(1)
        edge[:,:,:,1:] = torch.maximum(edge[:,:,:,1:], (hard[:,:,1:] != hard[:,:,:-1]).unsqueeze(1).float())
        projected = self.project(content)
        if self.modulation:
            projected = projected + self.modulator(torch.cat([content*(0.1+0.9*building.detach()), building.detach(), edge],1))
        weights = self.score(projected).sigmoid().flatten(2) if self.weighted else projected.new_ones(b,1,h*w)
        # Pool every feature-grid location, not a sampled 12-position subset.
        pooled = soft_pool_unpool(projected.float(), q*weights, ids, gh*gw)['tokens']
        stats = soft_pool_unpool(torch.cat([content,content.square()],1).float(),q,ids,gh*gw)
        mean, second = stats['tokens'].chunk(2,-1)
        z = self.main(pooled)
        if self.weighted:
            z = z + self.alpha.tanh()*self.skip(torch.cat([mean,(second-mean.square()).clamp_min(0).add(1e-6).sqrt()],-1))
        region_logits = self.classifier(z)
        pixel = content.new_zeros(b,h*w,4)
        for j in range(9):
            pixel += region_logits.gather(1,ids[j][None,:,None].expand(b,-1,4))*q[:,j,:,None]
        return dict(logits=pixel.transpose(1,2).reshape(b,4,h,w), tokens=z, q=q, ids=ids,
                    count=gh*gw, mass=stats['mass'], xy=xy, hard_ids=hard)


def partition_loss(aux, rgb, location):
    """RGB-feature, coordinate and building reconstruction; no damage-boundary claims."""
    q, ids, k = aux['q'], aux['ids'], aux['count']
    normalized = F.normalize(rgb.detach().float(),dim=1)
    rec = soft_pool_unpool(normalized,q,ids,k)['reconstruction']
    xyrec = soft_pool_unpool(aux['xy'],q,ids,k)['reconstruction']
    target = F.interpolate(location[:,None].float(),rgb.shape[-2:],mode='nearest')
    valid = target != 255
    safe = torch.where(valid,target,torch.zeros_like(target))
    numerator = soft_pool_unpool(safe,q,ids,k)['tokens']
    denominator = soft_pool_unpool(valid.float(),q,ids,k)['tokens']
    ratio = numerator/denominator.clamp_min(1e-6)
    reconstructed = torch.zeros_like(safe).flatten(2).transpose(1,2)
    for j in range(9):
        reconstructed += ratio.gather(1,ids[j][None,:,None].expand(len(rgb),-1,1))*q[:,j,:,None]
    reconstructed = reconstructed.transpose(1,2).reshape_as(safe).clamp(1e-6,1-1e-6)
    building = (F.binary_cross_entropy(reconstructed,safe,reduction='none')*valid).sum()/valid.sum().clamp_min(1)
    return F.mse_loss(rec,normalized)+0.1*F.mse_loss(xyrec,aux['xy'])+building


class RetrievalClassifier(nn.Module):
    """Support metadata only; query GT and query disaster ID are never inputs."""
    def __init__(self, dim=128, disaster_types=1):
        super().__init__()
        self.labels = nn.Embedding(4,dim)
        self.disasters = nn.Embedding(disaster_types,dim)
        self.geometry = nn.Linear(4,dim)
        self.attn = nn.MultiheadAttention(dim,4,batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim,2*dim),nn.GELU(),nn.Linear(2*dim,dim))
        self.out = nn.Linear(dim,4)

    def forward(self, query, support, labels, geometry=None, disaster=None):
        values = support+self.labels(labels)
        if geometry is not None: values = values+self.geometry(geometry)
        if disaster is not None: values = values+self.disasters(disaster)
        context,_ = self.attn(query[:,None],support,values,need_weights=False)
        h = self.norm(query+context[:,0])
        return self.out(h+self.ffn(h))
