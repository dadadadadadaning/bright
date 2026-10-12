from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

_SUFFIXES = ("_pre_disaster","_post_disaster","_building_damage","_building","_damage")

def canonical_stem(name: str) -> str:
    text = Path(str(name)).stem
    changed = True
    while changed:
        changed = False
        for suffix in _SUFFIXES:
            if text.endswith(suffix):
                text = text[:-len(suffix)]; changed = True
    return text

class InstanceIndex:
    def __init__(self, instance_dir: str | None):
        self.root = Path(instance_dir).expanduser() if instance_dir else None
        self.by_stem: Dict[str, Path] = {}
        if self.root and self.root.exists():
            for p in self.root.rglob("*"):
                if not p.is_file() or p.suffix.lower() not in {".png",".tif",".tiff",".npy",".json"}:
                    continue
                key = canonical_stem(p.name)
                old = self.by_stem.get(key)
                if old is None or (old.suffix.lower()==".json" and p.suffix.lower()!=".json"):
                    self.by_stem[key] = p
    def describe(self):
        return {"instance_dir": str(self.root) if self.root else None,
                "exists": bool(self.root and self.root.exists()),
                "indexed_files": len(self.by_stem),
                "extensions": sorted({p.suffix.lower() for p in self.by_stem.values()})}
    def resolve(self, sample_name: str) -> Optional[Path]:
        key = canonical_stem(sample_name)
        if key in self.by_stem: return self.by_stem[key]
        matches = [p for k,p in self.by_stem.items() if key == k or key in k or k in key]
        return matches[0] if len(matches)==1 else None

def _raster_to_ids(arr: np.ndarray) -> np.ndarray:
    arr=np.asarray(arr)
    if arr.ndim==3:
        if arr.shape[-1] in (3,4): arr=arr[...,0]
        elif arr.shape[0] in (3,4): arr=arr[0]
    arr=np.squeeze(arr)
    if arr.ndim!=2: raise ValueError(f"instance raster must be 2-D, got {arr.shape}")
    arr=arr.astype(np.int64,copy=False); ids=np.unique(arr); ids=ids[ids>0]
    out=np.zeros_like(arr,dtype=np.int32)
    for new,old in enumerate(ids.tolist(),1): out[arr==old]=new
    return out

def _draw_polygons(polygons: Iterable, h:int, w:int)->np.ndarray:
    canvas=Image.new("I",(w,h),0); draw=ImageDraw.Draw(canvas); idx=1
    for poly in polygons:
        if not poly: continue
        pts=[(float(x),float(y)) for x,y in poly]
        if len(pts)<3: continue
        draw.polygon(pts,fill=idx); idx+=1
    return np.asarray(canvas,dtype=np.int32)

def _json_to_ids(path:Path,h:int,w:int)->np.ndarray:
    obj=json.loads(path.read_text(encoding="utf-8")); polygons=[]
    if isinstance(obj,dict) and isinstance(obj.get("shapes"),list):
        polygons=[s.get("points") for s in obj["shapes"] if s.get("points")]
    elif isinstance(obj,dict) and isinstance(obj.get("annotations"),list):
        for ann in obj["annotations"]:
            seg=ann.get("segmentation")
            if isinstance(seg,list):
                for ring in seg:
                    if isinstance(ring,list) and len(ring)>=6 and isinstance(ring[0],(int,float)):
                        polygons.append(list(zip(ring[0::2],ring[1::2])))
    elif isinstance(obj,list):
        polygons=[x.get("points") or x.get("polygon") for x in obj if isinstance(x,dict) and (x.get("points") or x.get("polygon"))]
    if not polygons: raise ValueError("unsupported JSON instance format")
    return _draw_polygons(polygons,h,w)

def semantic_connected_components(damage: np.ndarray)->np.ndarray:
    building=np.isin(damage,[1,2,3]).astype(np.uint8)
    labels,_=ndimage.label(building,structure=np.asarray([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8))
    return labels.astype(np.int32)

def load_instance_map(sample_name:str, damage:np.ndarray, index:InstanceIndex)->Tuple[np.ndarray,str,str|None]:
    h,w=damage.shape; path=index.resolve(sample_name)
    if path is not None:
        try:
            suf=path.suffix.lower()
            if suf==".npy": ids=_raster_to_ids(np.load(path))
            elif suf in {".png",".tif",".tiff"}: ids=_raster_to_ids(np.asarray(Image.open(path)))
            elif suf==".json": ids=_json_to_ids(path,h,w)
            else: raise ValueError(suf)
            if ids.shape!=(h,w):
                ids=np.asarray(Image.fromarray(ids.astype(np.int32),mode="I").resize((w,h),resample=Image.Resampling.NEAREST),dtype=np.int32)
            return ids,f"instance_file:{path.name}",None
        except Exception as exc:
            return semantic_connected_components(damage),"semantic_cc_fallback",f"{type(exc).__name__}: {exc}"
    return semantic_connected_components(damage),"semantic_cc_fallback","no matching instance annotation"

def inspect_instance_dir(instance_dir:str|None,limit:int=30):
    idx=InstanceIndex(instance_dir); info=idx.describe(); info["preview"]=[str(p) for p in list(idx.by_stem.values())[:limit]]; return info
