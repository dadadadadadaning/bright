from __future__ import annotations
from dataclasses import dataclass
from typing import Dict,List,Tuple
import numpy as np
import torch
CLASS_NAMES={1:"Intact",2:"Damaged",3:"Destroyed"}
@dataclass
class AuditConfig:
    effective_region_threshold:float=0.05
    multi_building_fraction:float=0.10
    min_region_building_mass:float=1e-4

def _entropy3(p):
    p=np.clip(p,1e-12,1.0); return -(p*np.log(p)).sum(axis=-1)/np.log(3.0)

def association_dense_statistics(probabilities,candidate_ids,instance_map,damage_map):
    probs=probabilities.detach().float().cpu().numpy(); ids=candidate_ids.detach().long().cpu().numpy()
    inst=instance_map.detach().long().cpu().numpy().reshape(-1); dam=damage_map.detach().long().cpu().numpy().reshape(-1)
    K=int(ids.max())+1; B=int(inst.max())
    total=np.zeros(K); bg=np.zeros(K); bm=np.zeros((B+1,K)); dm=np.zeros((4,K))
    for j in range(9):
        r=ids[j]; w=probs[j].astype(np.float64,copy=False)
        total += np.bincount(r,weights=w,minlength=K)
        m=inst==0
        if m.any(): bg += np.bincount(r[m],weights=w[m],minlength=K)
        m=inst>0
        if m.any():
            combo=inst[m]*K+r[m]; bc=np.bincount(combo,weights=w[m],minlength=(B+1)*K); bm += bc.reshape(B+1,K)
        for c in (1,2,3):
            m=dam==c
            if m.any(): dm[c]+=np.bincount(r[m],weights=w[m],minlength=K)
    return {"total_mass":total,"background_mass":bg,"building_mass":bm,"damage_mass":dm}

def audit_one_image(sample_name,statistics,instance_map_np,damage_map_np,source,config):
    total=statistics["total_mass"]; bg=statistics["background_mass"]; bm=statistics["building_mass"]; dm=statistics["damage_mass"]
    K=len(total); B=bm.shape[0]-1; denom=np.clip(total,1e-12,None)
    build_frac=bm[1:]/denom[None,:] if B>0 else np.zeros((0,K)); max_b=build_frac.max(0) if B>0 else np.zeros(K)
    dom_b=build_frac.argmax(0)+1 if B>0 else np.zeros(K,dtype=np.int64); bg_frac=bg/denom
    multi=(build_frac>=config.multi_building_fraction).sum(0) if B>0 else np.zeros(K,dtype=np.int64)
    dsum=dm[1:4].sum(0); dp=np.zeros((K,3)); valid=dsum>config.min_region_building_mass; dp[valid]=dm[1:4,valid].T/dsum[valid,None]
    ent=np.ones(K); ent[valid]=_entropy3(dp[valid]); domp=np.zeros(K); domc=np.zeros(K,dtype=np.int64)
    if valid.any(): domp[valid]=dp[valid].max(1); domc[valid]=dp[valid].argmax(1)+1
    rr=[]
    for k in range(K):
        if total[k]<=1e-8: continue
        rr.append({"sample":sample_name,"region_id":int(k),"total_mass":float(total[k]),"dominant_building_id":int(dom_b[k]) if B>0 else 0,
                   "region_building_purity":float(max_b[k]),"background_crossing":float(bg_frac[k]),"multi_building_crossing":bool(multi[k]>=2),
                   "damage_entropy_norm":float(ent[k]),"dominant_damage_class":int(domc[k]),"dominant_damage_name":CLASS_NAMES.get(int(domc[k]),"None"),
                   "dominant_damage_purity":float(domp[k]),"p_intact":float(dp[k,0]),"p_damaged":float(dp[k,1]),"p_destroyed":float(dp[k,2]),"instance_source":source})
    br=[]
    for b in range(1,B+1):
        mask=instance_map_np==b; area=int(mask.sum())
        if area<=0: continue
        labels=damage_map_np[mask]; labels=labels[(labels>=1)&(labels<=3)]
        if labels.size==0: continue
        gt=int(np.argmax(np.bincount(labels.astype(np.int64),minlength=4)[1:4])+1)
        mb=bm[b]; tb=float(mb.sum())
        if tb<=1e-8: continue
        contrib=mb/tb; eff=np.where(contrib>=config.effective_region_threshold)[0]
        if eff.size==0: eff=np.asarray([int(np.argmax(contrib))])
        row={"sample":sample_name,"building_id":int(b),"building_gt_class":gt,"building_gt_name":CLASS_NAMES.get(gt,"Unknown"),"area_feature_pixels":area,
             "effective_region_count":int(eff.size),"max_region_contribution":float(contrib[eff].max()),
             "mean_effective_region_building_purity":float(max_b[eff].mean()),"mean_effective_region_background_crossing":float(bg_frac[eff].mean()),
             "mean_effective_region_damage_entropy":float(ent[eff].mean()),"mean_effective_region_dominant_damage_purity":float(domp[eff].mean()),
             "multi_building_region_fraction":float((multi[eff]>=2).mean()),"instance_source":source}
        for t in (0.5,0.6,0.7): row[f"damaged_region_at_{t:.1f}"]=bool(np.any(dp[eff,1]>=t))
        br.append(row)
    return br,rr

def add_size_bucket(rows):
    if not rows: return {"q33":0.0,"q67":0.0}
    a=np.asarray([float(r["area_feature_pixels"]) for r in rows]); q33,q67=np.quantile(a,[1/3,2/3])
    for r in rows:
        x=float(r["area_feature_pixels"]); r["size_bucket"]="small" if x<=q33 else ("medium" if x<=q67 else "large")
    return {"q33":float(q33),"q67":float(q67)}

def _sum(rows):
    if not rows:return {}
    c=np.asarray([r["effective_region_count"] for r in rows],dtype=float)
    return {"building_count":len(rows),"regions_per_building_mean":float(c.mean()),"regions_per_building_median":float(np.median(c)),
            "regions_per_building_p10":float(np.quantile(c,.1)),"regions_per_building_p25":float(np.quantile(c,.25)),"regions_per_building_p50":float(np.quantile(c,.5)),
            "regions_per_building_p75":float(np.quantile(c,.75)),"regions_per_building_p90":float(np.quantile(c,.9)),"building_1_region_fraction":float((c==1).mean()),
            "building_2_region_fraction":float((c==2).mean()),"building_3_4_region_fraction":float(((c>=3)&(c<=4)).mean()),"building_ge5_region_fraction":float((c>=5).mean()),
            "mean_region_building_purity":float(np.mean([r["mean_effective_region_building_purity"] for r in rows])),
            "mean_background_crossing":float(np.mean([r["mean_effective_region_background_crossing"] for r in rows])),
            "mean_damage_entropy_norm":float(np.mean([r["mean_effective_region_damage_entropy"] for r in rows])),
            "mean_dominant_damage_purity":float(np.mean([r["mean_effective_region_dominant_damage_purity"] for r in rows])),
            "mean_multi_building_region_fraction":float(np.mean([r["multi_building_region_fraction"] for r in rows]))}

def summarize(br,rr):
    out={"all":_sum(br)}
    for c,n in CLASS_NAMES.items(): out[n.lower()]=_sum([r for r in br if int(r["building_gt_class"])==c])
    out["by_size"]={s:_sum([r for r in br if r.get("size_bucket")==s]) for s in ("small","medium","large")}
    d=[r for r in br if int(r["building_gt_class"])==2]; cov={"damaged_building_count":len(d)}
    for t in (0.5,0.6,0.7): cov[f"recall_at_{t:.1f}"]=float(np.mean([bool(r[f"damaged_region_at_{t:.1f}"]) for r in d])) if d else 0.0
    out["damaged_evidence_coverage"]=cov
    out["region_level_all"]={} if not rr else {"region_count":len(rr),"purity_mean":float(np.mean([r["region_building_purity"] for r in rr])),
      "purity_median":float(np.median([r["region_building_purity"] for r in rr])),"purity_ge_0_90_fraction":float(np.mean([r["region_building_purity"]>=.9 for r in rr])),
      "purity_ge_0_75_fraction":float(np.mean([r["region_building_purity"]>=.75 for r in rr])),"purity_lt_0_50_fraction":float(np.mean([r["region_building_purity"]<.5 for r in rr])),
      "background_crossing_mean":float(np.mean([r["background_crossing"] for r in rr])),"multi_building_crossing_fraction":float(np.mean([r["multi_building_crossing"] for r in rr])),
      "damage_entropy_norm_mean":float(np.mean([r["damage_entropy_norm"] for r in rr])),"dominant_damage_purity_mean":float(np.mean([r["dominant_damage_purity"] for r in rr]))}
    return out
