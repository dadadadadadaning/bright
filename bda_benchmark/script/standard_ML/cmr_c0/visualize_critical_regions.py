# -*- coding: utf-8 -*-
"""Generate fixed-layout CMR-C0 critical-region diagnostic case sheets."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

from dataset.make_data_loader_p4_native_sar import img_loader, to_single_channel_sar
from model.cmr_c0.case_card import validate_case_cache
from model.cmr_c0.visual_diagnostics import (
    VIS_SCORE_PROTOCOL,
    apply_json_calibrators,
    cosine_similarity_matrix,
    expanded_crop_box,
    score_validation_queries,
    select_diagnostic_samples,
    validate_visual_score_payload,
)
from model.cmr_v1.region_cache import unpack_component, validate_region_cache
from script.standard_ML.rm_region_retrieval_runtime import binary_auroc


LABEL_NAMES = {0: "Intact", 1: "Damaged", 2: "Destroyed"}
OBSERVATION_COLUMNS = (
    "hotspot_location",
    "hotspot_modality",
    "hotspot_shape",
    "top_ratio_change",
    "nms_effect",
    "intact_case_match_quality",
    "damaged_case_match_quality",
    "why_all_is_better",
    "manual_judgement",
    "notes",
)
RAW_IMAGE_CACHE_LIMIT = 24


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "case"


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_case_cache(path: str, role: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_case_cache(payload, role)
    return payload


def _load_region_cache(path: str, role: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_region_cache(payload, role)
    return payload


def _region_row_lookup(region: Mapping[str, object]) -> Dict[int, int]:
    return {
        int(source): row
        for row, source in enumerate(region["source_region_indices"].tolist())
    }


def _component_for_case(
    cache: Mapping[str, object],
    row: int,
    region: Mapping[str, object],
    region_lookup: Mapping[int, int],
) -> np.ndarray:
    source = int(cache["source_region_indices"][row])
    if source not in region_lookup:
        raise RuntimeError(f"region cache is missing source row {source}")
    region_row = int(region_lookup[source])
    return unpack_component(
        region["packed_component_masks"][region_row],
        region["bboxes"][region_row].tolist(),
        region["image_shapes"][region_row].tolist(),
    )


def _load_raw_pair(dataset_root: str | Path, image_id: str) -> tuple[np.ndarray, np.ndarray]:
    root = Path(dataset_root)
    pre = np.asarray(
        img_loader(str(root / "pre-event" / f"{image_id}_pre_disaster.tif")),
        dtype=np.float32,
    )[..., :3]
    sar = to_single_channel_sar(img_loader(
        str(root / "post-event" / f"{image_id}_post_disaster.tif")
    )).astype(np.float32)
    if pre.shape[:2] != sar.shape:
        raise RuntimeError(f"unaligned visual pair for {image_id}: {pre.shape} vs {sar.shape}")
    return pre, sar


def _robust_rgb(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32)
    low = np.nanpercentile(value, 1.0, axis=(0, 1), keepdims=True)
    high = np.nanpercentile(value, 99.0, axis=(0, 1), keepdims=True)
    return np.clip((value - low) / np.maximum(high - low, 1e-6), 0.0, 1.0)


def _robust_sar(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32)
    low, high = np.nanpercentile(value, (2.0, 98.0))
    return np.clip((value - low) / max(float(high - low), 1e-6), 0.0, 1.0)


def _token_pixels(cache: Mapping[str, object], row: int) -> np.ndarray:
    shape = cache["image_shapes"][row].float()
    scale = torch.tensor(
        [max(int(shape[0]) - 1, 1), max(int(shape[1]) - 1, 1)],
        dtype=torch.float32,
    )
    return (cache["normalized_xy"][row].float() * scale).numpy()


def _crop(value: np.ndarray, box: Sequence[int]) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    return value[y1:y2, x1:x2]


def _axis_image(ax, image: np.ndarray, title: str, *, grayscale: bool = False) -> None:
    ax.imshow(image, cmap="gray" if grayscale else None, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_component(ax, component: np.ndarray, box: Sequence[int], color: str = "white") -> None:
    local = _crop(component.astype(np.float32), box)
    if bool(local.any()) and bool((local < 0.5).any()):
        ax.contour(local, levels=[0.5], colors=[color], linewidths=1.0)


def _scatter_tokens(
    ax,
    pixels_yx: np.ndarray,
    margins: np.ndarray,
    indices: np.ndarray,
    box: Sequence[int],
    *,
    vlim: float,
    marker: str = "s",
    size: float = 16.0,
    edgecolor: str = "black",
) -> None:
    x1, y1, _, _ = map(int, box)
    selected = np.asarray(indices, dtype=np.int64)
    if selected.size == 0:
        return
    y = pixels_yx[selected, 0] - y1
    x = pixels_yx[selected, 1] - x1
    ax.scatter(
        x, y, c=margins[selected], cmap="RdBu_r", vmin=-vlim, vmax=vlim,
        marker=marker, s=size, linewidths=0.35, edgecolors=edgecolor, alpha=0.88,
    )


def _case_visual_data(
    cache: Mapping[str, object],
    row: int,
    region: Mapping[str, object],
    region_lookup: Mapping[int, int],
    dataset_root: str,
    raw_cache: Dict[str, tuple[np.ndarray, np.ndarray]],
) -> Dict[str, object]:
    image_id = str(cache["image_ids"][row])
    if image_id not in raw_cache:
        while len(raw_cache) >= RAW_IMAGE_CACHE_LIMIT:
            raw_cache.pop(next(iter(raw_cache)))
        raw_cache[image_id] = _load_raw_pair(dataset_root, image_id)
    else:
        # Plain dictionaries preserve insertion order in supported Python;
        # reinsertion provides a small dependency-free LRU cache.
        raw_cache[image_id] = raw_cache.pop(image_id)
    pre, sar = raw_cache[image_id]
    component = _component_for_case(cache, row, region, region_lookup)
    if pre.shape[:2] != component.shape:
        raise RuntimeError(
            f"raw/component shape mismatch for {image_id}: {pre.shape[:2]} vs {component.shape}"
        )
    box = expanded_crop_box(
        cache["bboxes"][row].tolist(), cache["image_shapes"][row].tolist()
    )
    return {
        "image_id": image_id,
        "pre": _robust_rgb(pre),
        "sar": _robust_sar(sar),
        "component": component,
        "box": box,
        "pixels": _token_pixels(cache, row),
        "margins": cache["di_margin"][row].float().numpy(),
    }


def _overlay_panel(
    ax,
    visual: Mapping[str, object],
    cache: Mapping[str, object],
    row: int,
    variant: str,
    title: str,
    *,
    background: str,
    vlim: float,
) -> None:
    image = visual[background]
    _axis_image(
        ax, _crop(image, visual["box"]), title,
        grayscale=(background == "sar"),
    )
    _draw_component(ax, visual["component"], visual["box"])
    _scatter_tokens(
        ax, visual["pixels"], visual["margins"],
        cache["critical_indices"][variant][row].long().numpy(),
        visual["box"], vlim=vlim,
    )


def _render_case_sheet(
    record: Mapping[str, object],
    group: str,
    query: Mapping[str, object],
    support: Mapping[str, object],
    query_region: Mapping[str, object],
    support_region: Mapping[str, object],
    query_region_lookup: Mapping[int, int],
    support_region_lookup: Mapping[int, int],
    dataset_root: str,
    selected_variant: str,
    global_vlim: float,
    output_base: Path,
    raw_cache: Dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    save_pdf: bool,
) -> None:
    query_row = int(record["query_row"])
    intact_row = int(record["variants"][selected_variant]["best_intact"]["support_row"])
    damaged_row = int(record["variants"][selected_variant]["best_damaged"]["support_row"])
    qv = _case_visual_data(
        query, query_row, query_region, query_region_lookup, dataset_root, raw_cache
    )
    iv = _case_visual_data(
        support, intact_row, support_region, support_region_lookup, dataset_root, raw_cache
    )
    dv = _case_visual_data(
        support, damaged_row, support_region, support_region_lookup, dataset_root, raw_cache
    )
    local_vlim = max(float(np.nanpercentile(np.abs(qv["margins"]), 98.0)), 1e-4)

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(3, 8, figsize=(20, 8.6), constrained_layout=True)

    # Row 1: identity and inputs.
    _axis_image(axes[0, 0], _crop(qv["pre"], qv["box"]), "Query pre-event RGB")
    _axis_image(axes[0, 1], _crop(qv["sar"], qv["box"]), "Query post-event SAR", grayscale=True)
    _axis_image(axes[0, 2], _crop(qv["component"], qv["box"]), "Frozen P3 component", grayscale=True)
    _axis_image(axes[0, 3], _crop(qv["pre"], qv["box"]), "RGB + component")
    _draw_component(axes[0, 3], qv["component"], qv["box"], color="#F0E442")
    _axis_image(axes[0, 4], _crop(qv["sar"], qv["box"]), "SAR + component", grayscale=True)
    _draw_component(axes[0, 4], qv["component"], qv["box"], color="#F0E442")
    probability = record["base_probabilities"]
    metadata = (
        f"Group: {group}\nEvent: {record['event']}\nDisaster: {record['disaster_type']}\n"
        f"GT: {LABEL_NAMES[int(record['label'])]}\nP3: {LABEL_NAMES[int(record['base_prediction'])]}\n"
        f"P3 p(I/D/X): {probability[0]:.3f}/{probability[1]:.3f}/{probability[2]:.3f}\n"
        f"BBox: {record['bbox']}\nArea: {record['area']}\nIdentity: {record['identity_mode']}"
    )
    axes[0, 5].axis("off")
    axes[0, 5].text(0.0, 1.0, metadata, va="top", ha="left", fontsize=8)
    axes[0, 6].axis("off")
    axes[0, 7].axis("off")

    # Row 2: D-I and fixed selections. First heatmap is sample-normalised;
    # overlays and selection panels use one global scale across all cases.
    axes[1, 0].imshow(
        np.zeros_like(_crop(qv["component"], qv["box"]), dtype=np.float32),
        cmap="gray", vmin=0, vmax=1,
    )
    axes[1, 0].set_title("D-I heatmap (local scale)")
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
    _draw_component(axes[1, 0], qv["component"], qv["box"])
    _scatter_tokens(
        axes[1, 0], qv["pixels"], qv["margins"],
        np.arange(len(qv["margins"])), qv["box"], vlim=local_vlim,
        edgecolor="none", size=20,
    )
    _overlay_panel(
        axes[1, 1], qv, query, query_row, "all", "D-I on RGB (global)",
        background="pre", vlim=global_vlim,
    )
    _overlay_panel(
        axes[1, 2], qv, query, query_row, "all", "D-I on SAR (global)",
        background="sar", vlim=global_vlim,
    )
    for axis, variant, title in zip(
        axes[1, 3:7],
        ("di_top10", "di_top20", "di_top30", "di_top20_nms"),
        ("Top 10%", "Top 20%", "Top 30%", "Top 20% + NMS"),
    ):
        _overlay_panel(
            axis, qv, query, query_row, variant, title,
            background="pre", vlim=global_vlim,
        )
    _axis_image(axes[1, 7], _crop(qv["pre"], qv["box"]), "Core / boundary")
    _draw_component(axes[1, 7], qv["component"], qv["box"])
    boundary = query["is_boundary"][query_row].bool().numpy()
    all_indices = np.arange(boundary.size)
    crop_x1, crop_y1, _, _ = map(int, qv["box"])
    for indices, marker, color, size in (
        (all_indices[~boundary], "o", "#009E73", 12),
        (all_indices[boundary], "x", "#E69F00", 15),
    ):
        axes[1, 7].scatter(
            qv["pixels"][indices, 1] - crop_x1,
            qv["pixels"][indices, 0] - crop_y1,
            marker=marker,
            s=size,
            color=color,
            linewidths=0.7,
        )

    # Row 3: best frozen counterfactual cases and local affinity matrices.
    _overlay_panel(
        axes[2, 0], iv, support, intact_row, selected_variant,
        "Best Intact: RGB + critical", background="pre", vlim=global_vlim,
    )
    _overlay_panel(
        axes[2, 1], iv, support, intact_row, selected_variant,
        "Best Intact: SAR + critical", background="sar", vlim=global_vlim,
    )
    _overlay_panel(
        axes[2, 2], dv, support, damaged_row, selected_variant,
        "Best Damaged: RGB + critical", background="pre", vlim=global_vlim,
    )
    _overlay_panel(
        axes[2, 3], dv, support, damaged_row, selected_variant,
        "Best Damaged: SAR + critical", background="sar", vlim=global_vlim,
    )
    q_index = query["critical_indices"][selected_variant][query_row]
    i_index = support["critical_indices"][selected_variant][intact_row]
    d_index = support["critical_indices"][selected_variant][damaged_row]
    sim_i = cosine_similarity_matrix(
        query["features"][query_row], support["features"][intact_row], q_index, i_index
    ).numpy()
    sim_d = cosine_similarity_matrix(
        query["features"][query_row], support["features"][damaged_row], q_index, d_index
    ).numpy()
    axes[2, 4].imshow(sim_i, cmap="viridis", vmin=-1.0, vmax=1.0, aspect="auto")
    axes[2, 4].set_title("Query→Intact affinity")
    axes[2, 4].set_xlabel("Memory token"); axes[2, 4].set_ylabel("Query token")
    axes[2, 5].imshow(sim_d, cmap="viridis", vmin=-1.0, vmax=1.0, aspect="auto")
    axes[2, 5].set_title("Query→Damaged affinity")
    axes[2, 5].set_xlabel("Memory token"); axes[2, 5].set_ylabel("Query token")

    all_row = record["variants"]["all"]
    selected_row = record["variants"][selected_variant]
    score_text = (
        f"ALL\nS_I={all_row['s_i']:.4f}\nS_D={all_row['s_d']:.4f}\n"
        f"Delta={all_row['delta_d_i']:.4f}\nCombined={all_row['combined_probability']:.4f}\n\n"
        f"{selected_variant}\nS_I={selected_row['s_i']:.4f}\nS_D={selected_row['s_d']:.4f}\n"
        f"Delta={selected_row['delta_d_i']:.4f}\nCombined={selected_row['combined_probability']:.4f}\n\n"
        f"Best I rank={selected_row['best_intact']['rank']}\n"
        f"Best D rank={selected_row['best_damaged']['rank']}"
    )
    axes[2, 6].axis("off")
    axes[2, 6].text(0.0, 1.0, score_text, va="top", ha="left", fontsize=8)
    axes[2, 7].axis("off")
    axes[2, 7].text(
        0.0, 1.0,
        "Colour coding\nRed: supports Damaged\nBlue: supports Intact\n"
        "Yellow outline: frozen component\nGreen circle: core token\nOrange x: boundary token\n"
        "Affinity: cosine similarity",
        va="top", ha="left", fontsize=8,
    )

    norm = Normalize(vmin=-global_vlim, vmax=global_vlim)
    mapper = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    mapper.set_array([])
    fig.colorbar(
        mapper, ax=list(axes[1, :7]), orientation="horizontal",
        fraction=0.035, pad=0.02, label="P3 local logit margin: Damaged − Intact",
    )
    fig.suptitle(
        f"CMR-C0 diagnostic | {group} | {record['event']} | {record['image_id']} | row={query_row}",
        fontsize=10,
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _observation_row(
    record: Mapping[str, object], group: str, selected_variant: str, figure: str
) -> Dict[str, object]:
    all_row = record["variants"]["all"]
    critical = record["variants"][selected_variant]
    row: Dict[str, object] = {
        "group": group,
        "query_row": int(record["query_row"]),
        "image_id": record["image_id"],
        "event": record["event"],
        "disaster_type": record["disaster_type"],
        "ground_truth": LABEL_NAMES[int(record["label"])],
        "p3_prediction": LABEL_NAMES[int(record["base_prediction"])],
        "identity_mode": record["identity_mode"],
        "all_s_i": all_row["s_i"],
        "all_s_d": all_row["s_d"],
        "all_delta_d_i": all_row["delta_d_i"],
        "all_combined_probability": all_row["combined_probability"],
        "critical_variant": selected_variant,
        "critical_s_i": critical["s_i"],
        "critical_s_d": critical["s_d"],
        "critical_delta_d_i": critical["delta_d_i"],
        "critical_combined_probability": critical["combined_probability"],
        "best_intact_rank": critical["best_intact"]["rank"],
        "best_intact_memory_index": critical["best_intact"]["memory_index"],
        "best_damaged_rank": critical["best_damaged"]["rank"],
        "best_damaged_memory_index": critical["best_damaged"]["memory_index"],
        "figure": figure,
    }
    row.update({name: "" for name in OBSERVATION_COLUMNS})
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _verify_against_frozen_audit(
    records: Sequence[Mapping[str, object]],
    audit: Mapping[str, object],
    variants: Sequence[str],
) -> None:
    expected_count = int(audit["validation_availability"]["retained_rows"])
    if len(records) != expected_count:
        raise RuntimeError(
            f"visual scoring retained {len(records)} rows; frozen audit retained {expected_count}"
        )
    target = torch.tensor([int(record["label"]) == 1 for record in records])
    for variant in variants:
        score = torch.tensor([
            float(record["variants"][variant]["combined_probability"])
            for record in records
        ])
        actual = float(binary_auroc(score, target))
        expected = float(audit["results"][f"{variant}_combined"]["validation"]["auroc"])
        if abs(actual - expected) > 1e-6:
            raise RuntimeError(
                f"{variant} visual-score AUROC {actual:.9f} does not replay "
                f"frozen audit {expected:.9f}"
            )
        print(f"Verified frozen {variant} Validation AUROC: {actual:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser("CMR-C0 critical-region visual diagnosis")
    parser.add_argument("--support", required=True)
    parser.add_argument("--validation_queries", required=True)
    parser.add_argument("--support_region", required=True)
    parser.add_argument("--validation_region", required=True)
    parser.add_argument("--audit_json", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--score_cache", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--neighbor_k", type=int, default=3)
    parser.add_argument("--class_temperature", type=float, default=0.10)
    parser.add_argument("--per_event", type=int, default=2)
    parser.add_argument("--maximum_per_group", type=int, default=24)
    parser.add_argument("--force_scores", action="store_true")
    parser.add_argument("--no_pdf", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = _read_json(args.audit_json)
    if audit.get("protocol") != "CMR_C0_ISOLATED_frozen_critical_verifier_v1":
        raise RuntimeError("visual diagnosis requires the frozen CMR-C0 audit")
    if bool(audit.get("test_accessed", True)):
        raise RuntimeError("visual diagnosis refuses a Test-derived audit")
    selected_variant = str(audit["selected_variant"])
    if selected_variant == "all":
        raise RuntimeError(
            "Train selected 'all'; there is no frozen critical-token variant to diagnose"
        )
    variants = ("all", selected_variant)

    support = _load_case_cache(args.support, "support")
    validation = _load_case_cache(args.validation_queries, "validation_query")
    if support["p3_checkpoint_sha256"] != validation["p3_checkpoint_sha256"]:
        raise RuntimeError("support/query P3 hashes differ")
    if bool(support.get("test_accessed", True)) or bool(validation.get("test_accessed", True)):
        raise RuntimeError("visual diagnosis refuses Test-derived case caches")

    score_path = Path(args.score_cache)
    if score_path.exists() and not args.force_scores:
        score_payload = _read_json(score_path)
        validate_visual_score_payload(score_payload)
        if score_payload["p3_checkpoint_sha256"] != validation["p3_checkpoint_sha256"]:
            raise RuntimeError("visual-score cache uses a different P3")
        if list(score_payload["variants"]) != list(variants):
            raise RuntimeError("visual-score variants differ from current Train selection")
        records = score_payload["records"]
        print(f"Loaded frozen visual-score cache: {score_path}")
    else:
        records = score_validation_queries(
            validation, support, variants=variants, device=args.device,
            neighbor_k=args.neighbor_k,
            class_temperature=args.class_temperature,
        )
        apply_json_calibrators(records, audit, variants)
        score_payload = {
            "protocol": VIS_SCORE_PROTOCOL,
            "scope": "Validation-only frozen Top20 diagnostic; no method selection",
            "variants": list(variants),
            "selected_variant_from_train": selected_variant,
            "p3_checkpoint_sha256": validation["p3_checkpoint_sha256"],
            "records": records,
            "test_accessed": False,
        }
        validate_visual_score_payload(score_payload)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = score_path.with_suffix(score_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(score_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(score_path)
        print(f"Saved frozen visual-score cache: {score_path}")

    _verify_against_frozen_audit(records, audit, variants)

    selected = select_diagnostic_samples(
        records, selected_variant=selected_variant,
        per_event=args.per_event, maximum_per_group=args.maximum_per_group,
    )
    record_lookup = {int(record["query_row"]): record for record in records}
    query_region = _load_region_cache(args.validation_region, "validation_query")
    support_region = _load_region_cache(args.support_region, "support")
    query_region_lookup = _region_row_lookup(query_region)
    support_region_lookup = _region_row_lookup(support_region)

    all_margin = torch.cat([value.float().abs() for value in validation["di_margin"]])
    global_vlim = max(float(torch.quantile(all_margin, 0.98)), 1e-4)
    del all_margin
    gc.collect()

    raw_cache: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    observation_rows: list[Dict[str, object]] = []
    figure_manifest: Dict[str, list[str]] = defaultdict(list)
    for group, rows in selected.items():
        group_dir = output_dir / "case_sheets" / group
        for order, query_row in enumerate(rows, 1):
            record = record_lookup[int(query_row)]
            stem = (
                f"{order:03d}_{_safe_name(record['event'])}_"
                f"{_safe_name(record['image_id'])}_row{query_row}"
            )
            output_base = group_dir / stem
            _render_case_sheet(
                record, group, validation, support,
                query_region, support_region,
                query_region_lookup, support_region_lookup,
                args.dataset_root, selected_variant, global_vlim,
                output_base, raw_cache, save_pdf=not args.no_pdf,
            )
            relative_png = str(output_base.with_suffix(".png").relative_to(output_dir))
            observation_rows.append(
                _observation_row(record, group, selected_variant, relative_png)
            )
            figure_manifest[group].append(relative_png)
            print(f"CMR-C0-VIS rendered: {relative_png}", flush=True)

    _write_csv(output_dir / "CMR_C0_VIS_manual_observations.csv", observation_rows)
    event_rows: list[Dict[str, object]] = []
    count = Counter((row["group"], row["event"]) for row in observation_rows)
    for (group, event), value in sorted(count.items()):
        event_rows.append({"group": group, "event": event, "selected_cases": value})
    _write_csv(output_dir / "CMR_C0_VIS_event_selection_summary.csv", event_rows)
    manifest = {
        "protocol": "CMR_C0_ISOLATED_visual_diagnostic_manifest_v1",
        "scope": "qualitative diagnosis only; GT used only for offline grouping",
        "selected_variant_from_train": selected_variant,
        "global_di_absolute_p98": global_vlim,
        "groups": {key: list(value) for key, value in selected.items()},
        "figures": dict(figure_manifest),
        "observation_csv": "CMR_C0_VIS_manual_observations.csv",
        "event_summary_csv": "CMR_C0_VIS_event_selection_summary.csv",
        "pixel_level_damage_ground_truth_claimed": False,
        "raw_image_cache_limit": RAW_IMAGE_CACHE_LIMIT,
        "p3_changed": False,
        "test_accessed": False,
    }
    (output_dir / "CMR_C0_VIS_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=" * 96)
    print("CMR-C0 visual diagnosis completed")
    print(f"Selected variant : {selected_variant}")
    print(f"Rendered sheets  : {len(observation_rows)}")
    for group, rows in selected.items():
        print(f"  {group:<24}: {len(rows)}")
    print(f"Output directory : {output_dir}")
    print("P3 changed       : False")
    print("Test accessed    : False")


if __name__ == "__main__":
    main()
