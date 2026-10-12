from __future__ import annotations
import torch
from .schema import LABEL_I, LABEL_D
from .prototype import prototype_scores
from .memory_retriever import query_memory
from .utility import make_utility_features

@torch.no_grad()
def late_decision_one(
    *, base_label, specialist_score, query_key, query_regions, query_region_mask,
    query_region_meta, query_event, bank, utility_model, base_uncertainty,
    mean_align_conf, peak_delta, tau_specialist=0.5, tau_utility=0.5, tau_x_veto=0.15
):
    if int(base_label) != LABEL_I:
        return {"accept": False, "final_label": int(base_label), "reason": "base_not_intact"}
    proto = prototype_scores(query_key, query_event, bank)
    mem = query_memory(query_key, query_regions, query_region_mask, query_region_meta, query_event, bank)
    feat = make_utility_features(
        specialist_score, base_uncertainty, proto, mem, mean_align_conf, peak_delta
    )
    utility = torch.sigmoid(utility_model(feat.unsqueeze(0))[0])
    x_conflict = float(mem["x_veto"]) > tau_x_veto or float(proto["proto_X_veto"]) > tau_x_veto
    accept = float(specialist_score) >= tau_specialist and float(utility) >= tau_utility and not x_conflict
    return {
        "accept": bool(accept),
        "final_label": LABEL_D if accept else LABEL_I,
        "specialist_score": float(specialist_score),
        "utility": float(utility),
        "prototype": {k: float(v) for k,v in proto.items()},
        "memory": {k: float(v) for k,v in mem.items() if k not in {"neighbor_indices","neighbor_scores"}},
        "neighbor_indices": mem["neighbor_indices"].cpu(),
        "neighbor_scores": mem["neighbor_scores"].cpu(),
    }
