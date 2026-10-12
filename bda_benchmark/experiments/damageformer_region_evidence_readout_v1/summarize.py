#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from .core import write_json


def main():
    p=argparse.ArgumentParser(); p.add_argument("--audit_root", required=True); p.add_argument("--output_json", required=True)
    a=p.parse_args(); root=Path(a.audit_root); rows={}
    for path in sorted(root.glob("*_audit.json")):
        data=json.loads(path.read_text(encoding="utf-8")); variant=data["variant"]
        rows[variant]={}
        for name,result in data["runs"].items():
            m=result["metrics"]
            rows[variant][name]={"valid_miou":m["valid_miou"], "gain_pp":result["gain_valid_miou_pp"],
                "damaged_iou":m["damaged_iou"], "damaged_f1":m["damaged_f1"],
                "damaged_to_intact":m["damaged_to_intact"], "destroyed_iou":m["destroyed_iou"],
                "rescue_rate":m["rescue_rate"], "harm_rate":m["harm_rate"],
                "accepted_candidate_pixel_fraction":m["accepted_candidate_pixel_fraction"],
                "accepted_candidate_building_fraction":m["accepted_candidate_building_fraction"],
                "net_correction_count":m["net_correction_count"]}
    checks={
        "w5_kd_above_w0_unconstrained": rows.get("w5_logit4_kd",{}).get("global",{}).get("valid_miou",-1) > rows.get("w0_full_residual",{}).get("global",{}).get("valid_miou",1),
        "w5_above_w3_safe_id_readout": rows.get("w5_logit4_kd",{}).get("global",{}).get("valid_miou",-1) > rows.get("w3_id_mass",{}).get("global",{}).get("valid_miou",1),
        "w2_above_w1": rows.get("w2_id_logit",{}).get("global",{}).get("valid_miou",-1) > rows.get("w1_id_calibration",{}).get("global",{}).get("valid_miou",1),
        "w3_above_w2": rows.get("w3_id_mass",{}).get("global",{}).get("valid_miou",-1) > rows.get("w2_id_logit",{}).get("global",{}).get("valid_miou",1),
        "w3_selective_above_global": max([v.get("valid_miou",-1) for k,v in rows.get("w3_id_mass",{}).items() if k.startswith("top_")] or [-1]) > rows.get("w3_id_mass",{}).get("global",{}).get("valid_miou",1),
    }
    payload={"protocol":"DAMAGEFORMER_REGION_EVIDENCE_READOUT_V1_SUMMARY", "rows":rows,
             "causal_checks":checks, "note":"Val-only admission; not Test evidence.", "test_accessed":False}
    write_json(Path(a.output_json),payload); print(json.dumps(payload,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
