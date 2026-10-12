#!/usr/bin/env python3
"""Build quantitative plots and qualitative BRIGHT comparison panels.

Run this after ``evaluate_model.py`` has produced one directory per model.
Only real saved predictions are used; the script never fabricates placeholders.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


MODEL_ORDER = ("unet", "siamcrnn", "siamattnunet", "deeplabv3plus", "damageformer", "changemamba")
DISPLAY_NAMES = {
    "unet": "U-Net",
    "siamcrnn": "SiamCRNN",
    "siamattnunet": "SiamAttnUNet",
    "deeplabv3plus": "DeepLabV3+",
    "damageformer": "DamageFormer",
    "changemamba": "ChangeMamba",
}
CLASS_NAMES = ("Background", "Intact", "Damaged", "Destroyed")
COLORS = np.asarray(
    [(245, 246, 248), (70, 181, 121), (228, 189, 89), (182, 70, 69)], dtype=np.float64
) / 255.0
CLASS_CMAP = ListedColormap(COLORS)
MODEL_COLORS = ("#7895B2", "#92B4A7", "#D7A86E", "#B7A6CC", "#E07A5F", "#355C7D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--suffix", default=".tif")
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--samples", nargs="*", default=None,
                        help="Optional exact sample IDs. Otherwise informative cases are selected automatically.")
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value) -> float:
    if value in (None, "", "None", "nan", "NaN"):
        return float("nan")
    return float(value)


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_inputs(args: argparse.Namespace) -> None:
    missing = []
    for model in args.models:
        for relative in ("metrics.json", "per_image_metrics.csv", "per_event_metrics.csv"):
            path = args.results_root / model / relative
            if not path.exists():
                missing.append(str(path))
    if not args.dataset_root.exists():
        missing.append(str(args.dataset_root))
    if missing:
        raise FileNotFoundError("Evaluation outputs are incomplete:\n  " + "\n  ".join(missing))


def build_comparison_rows(results_root: Path, models: list[str]) -> list[dict]:
    rows = []
    for model in models:
        payload = read_json(results_root / model / "metrics.json")
        metrics = payload["metrics"]
        timing = payload["timing"]
        row = {
            "model": DISPLAY_NAMES[model],
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "num_images": payload["num_images"],
            "mIoU_4class_pct": 100 * metrics["mean_iou_4class"],
            "mIoU_foreground_pct": 100 * metrics["foreground_miou_3class"],
            "building_F1_pct": 100 * metrics["building_f1"],
            "intact_IoU_pct": 100 * metrics["iou"][1],
            "damaged_IoU_pct": 100 * metrics["iou"][2],
            "destroyed_IoU_pct": 100 * metrics["iou"][3],
            "damaged_F1_pct": 100 * metrics["f1"][2],
            "destroyed_F1_pct": 100 * metrics["f1"][3],
            "damaged_to_intact_pct": 100 * metrics["damaged_to_intact_rate"],
            "sec_per_image": timing["mean_seconds_per_image"],
            "peak_cuda_memory_mb": timing["peak_cuda_memory_mb"],
            "tta": payload["protocol"]["tta"],
            "amp": payload["protocol"]["amp"],
        }
        rows.append(row)
    return rows


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D8DEE7", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def plot_overall(rows: list[dict], output_dir: Path) -> None:
    names = [row["model"] for row in rows]
    metric_keys = ("mIoU_4class_pct", "mIoU_foreground_pct", "damaged_F1_pct")
    metric_labels = ("4-class mIoU", "Foreground mIoU", "Damaged F1")
    x = np.arange(len(names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.2, 4.8), constrained_layout=True)
    shades = ("#7E9AB8", "#E6B566", "#C96F68")
    for offset, (key, label, color) in enumerate(zip(metric_keys, metric_labels, shades)):
        values = [row[key] for row in rows]
        bars = ax.bar(x + (offset - 1) * width, values, width, label=label, color=color, edgecolor="white")
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax.set_xticks(x, names, rotation=18, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title("Unified BRIGHT evaluation")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    style_axes(ax)
    fig.savefig(output_dir / "overall_metrics.png", dpi=300)
    fig.savefig(output_dir / "overall_metrics.pdf")
    plt.close(fig)


def plot_per_class_iou(rows: list[dict], output_dir: Path) -> None:
    names = [row["model"] for row in rows]
    x = np.arange(len(names))
    width = 0.25
    keys = ("intact_IoU_pct", "damaged_IoU_pct", "destroyed_IoU_pct")
    colors = (COLORS[1], COLORS[2], COLORS[3])
    fig, ax = plt.subplots(figsize=(10.2, 4.8), constrained_layout=True)
    for offset, (key, class_name, color) in enumerate(zip(keys, CLASS_NAMES[1:], colors)):
        bars = ax.bar(x + (offset - 1) * width, [row[key] for row in rows], width,
                      label=class_name, color=color, edgecolor="white")
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax.set_xticks(x, names, rotation=18, ha="right")
    ax.set_ylabel("IoU (%)")
    ax.set_title("Foreground class IoU")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    style_axes(ax)
    fig.savefig(output_dir / "per_class_iou.png", dpi=300)
    fig.savefig(output_dir / "per_class_iou.pdf")
    plt.close(fig)


def plot_event_heatmap(results_root: Path, models: list[str], output_dir: Path) -> None:
    by_model = {}
    events = set()
    for model in models:
        rows = read_csv(results_root / model / "per_event_metrics.csv")
        by_model[model] = {row["event"]: 100 * as_float(row["foreground_miou_3class"]) for row in rows}
        events.update(by_model[model])
    events = sorted(events)
    matrix = np.asarray([[by_model[model].get(event, np.nan) for event in events] for model in models])
    fig_width = max(10.5, 0.67 * len(events))
    fig, ax = plt.subplots(figsize=(fig_width, 4.1), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    ax.set_xticks(np.arange(len(events)), events, rotation=38, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(models)), [DISPLAY_NAMES[model] for model in models])
    ax.set_title("Foreground mIoU by disaster event")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if np.isfinite(matrix[row, col]):
                normalized = (matrix[row, col] - np.nanmin(matrix)) / max(np.nanmax(matrix) - np.nanmin(matrix), 1e-8)
                ax.text(col, row, f"{matrix[row, col]:.1f}", ha="center", va="center",
                        fontsize=6.5, color="white" if normalized > 0.58 else "#1F2937")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.018)
    colorbar.set_label("Foreground mIoU (%)")
    fig.savefig(output_dir / "event_foreground_miou_heatmap.png", dpi=300)
    fig.savefig(output_dir / "event_foreground_miou_heatmap.pdf")
    plt.close(fig)


def plot_confusion_matrices(results_root: Path, models: list[str], output_dir: Path) -> None:
    cols = 3
    rows = math.ceil(len(models) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10.5, 3.35 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for index, model in enumerate(models):
        ax = axes.flat[index]
        cm = np.loadtxt(results_root / model / "confusion_matrix.csv", delimiter=",")
        row_sum = cm.sum(axis=1, keepdims=True)
        normalized = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0) * 100
        ax.imshow(normalized, cmap="Blues", vmin=0, vmax=100)
        for row in range(4):
            for col in range(4):
                ax.text(col, row, f"{normalized[row, col]:.1f}", ha="center", va="center",
                        fontsize=8, color="white" if normalized[row, col] > 55 else "#1F2937")
        ax.set_xticks(range(4), CLASS_NAMES, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(4), CLASS_NAMES, fontsize=8)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Ground truth")
        ax.set_title(DISPLAY_NAMES[model])
    for index in range(len(models), rows * cols):
        axes.flat[index].axis("off")
    fig.savefig(output_dir / "normalized_confusion_matrices.png", dpi=300)
    fig.savefig(output_dir / "normalized_confusion_matrices.pdf")
    plt.close(fig)


def select_cases(results_root: Path, models: list[str], count: int) -> list[dict]:
    per_model = {}
    for model in models:
        rows = read_csv(results_root / model / "per_image_metrics.csv")
        per_model[model] = {row["name"]: row for row in rows}
    common = set.intersection(*(set(items) for items in per_model.values()))
    candidates = []
    for name in common:
        reference = per_model[models[0]][name]
        fg_scores = np.asarray([as_float(per_model[model][name]["foreground_miou_3class"]) for model in models])
        damaged_pixels = int(float(reference["gt_pixels_damaged"]))
        destroyed_pixels = int(float(reference["gt_pixels_destroyed"]))
        intact_pixels = int(float(reference["gt_pixels_intact"]))
        if not np.isfinite(fg_scores).any():
            continue
        spread = float(np.nanmax(fg_scores) - np.nanmin(fg_scores))
        mean_score = float(np.nanmean(fg_scores))
        candidates.append(
            {
                "name": name,
                "event": reference["event"],
                "damaged_pixels": damaged_pixels,
                "destroyed_pixels": destroyed_pixels,
                "intact_pixels": intact_pixels,
                "spread": spread,
                "mean_fg_miou": mean_score,
            }
        )

    selections = []
    used_names = set()
    used_events = set()

    def take(label: str, predicate, sort_key, reverse=True):
        ranked = sorted((item for item in candidates if predicate(item)), key=sort_key, reverse=reverse)
        for item in ranked:
            if item["name"] not in used_names and item["event"] not in used_events:
                chosen = dict(item)
                chosen["selection_reason"] = label
                selections.append(chosen)
                used_names.add(item["name"])
                used_events.add(item["event"])
                return
        for item in ranked:
            if item["name"] not in used_names:
                chosen = dict(item)
                chosen["selection_reason"] = label
                selections.append(chosen)
                used_names.add(item["name"])
                return

    if count >= 1:
        take("Damaged-rich disagreement", lambda x: x["damaged_pixels"] > 200,
             lambda x: (x["spread"], x["damaged_pixels"]))
    if count >= 2:
        take("Destroyed-rich disagreement", lambda x: x["destroyed_pixels"] > 200,
             lambda x: (x["spread"], x["destroyed_pixels"]))
    if count >= 3:
        take("Mixed-class scene", lambda x: x["damaged_pixels"] > 50 and x["destroyed_pixels"] > 50,
             lambda x: (x["spread"], x["damaged_pixels"] + x["destroyed_pixels"]))
    if count >= 4:
        take("Common hard case", lambda x: x["damaged_pixels"] + x["destroyed_pixels"] > 100,
             lambda x: x["mean_fg_miou"], reverse=False)
    while len(selections) < count:
        before = len(selections)
        take("High model disagreement", lambda _x: True, lambda x: x["spread"])
        if len(selections) == before:
            break
    return selections[:count]


def robust_uint8(array: np.ndarray, per_channel: bool = False) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    data = array.astype(np.float32)
    if per_channel and data.ndim == 3:
        output = np.empty_like(data)
        for channel in range(data.shape[2]):
            low, high = np.percentile(data[..., channel], (1, 99))
            output[..., channel] = np.clip((data[..., channel] - low) / max(high - low, 1e-6), 0, 1)
        return (255 * output).astype(np.uint8)
    low, high = np.percentile(data, (1, 99))
    return (255 * np.clip((data - low) / max(high - low, 1e-6), 0, 1)).astype(np.uint8)


def load_inputs(dataset_root: Path, name: str, suffix: str):
    pre = imageio.imread(dataset_root / "pre-event" / f"{name}_pre_disaster{suffix}")
    post = imageio.imread(dataset_root / "post-event" / f"{name}_post_disaster{suffix}")
    gt = imageio.imread(dataset_root / "target" / f"{name}_building_damage{suffix}")
    pre = robust_uint8(pre[..., :3], per_channel=True)
    if post.ndim == 3:
        post = post[..., 0]
    post = robust_uint8(post)
    if gt.ndim == 3:
        gt = gt[..., 0]
    return pre, post, gt.astype(np.uint8)


def show_panel(ax, image, title: str, *, mask: bool = False) -> None:
    if mask:
        ax.imshow(image, cmap=CLASS_CMAP, vmin=-0.5, vmax=3.5, interpolation="nearest")
    else:
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_edgecolor("#BBC3CD")


def plot_qualitative_cases(
    results_root: Path,
    dataset_root: Path,
    models: list[str],
    selections: list[dict],
    suffix: str,
    output_dir: Path,
) -> None:
    columns = 3 + len(models)
    rows = len(selections)
    fig, axes = plt.subplots(rows, columns, figsize=(2.05 * columns, 2.12 * rows + 0.5), squeeze=False)
    for row_index, selection in enumerate(selections):
        name = selection["name"]
        pre, post, gt = load_inputs(dataset_root, name, suffix)
        show_panel(axes[row_index, 0], pre, "Pre-event RGB")
        show_panel(axes[row_index, 1], post, "Post-event SAR")
        show_panel(axes[row_index, 2], gt, "Ground truth", mask=True)
        for col_offset, model in enumerate(models, start=3):
            prediction_path = results_root / model / "predictions/raw" / f"{name}.png"
            if not prediction_path.exists():
                raise FileNotFoundError(f"Prediction required for qualitative panel: {prediction_path}")
            prediction = np.asarray(imageio.imread(prediction_path), dtype=np.uint8)
            show_panel(axes[row_index, col_offset], prediction, DISPLAY_NAMES[model], mask=True)
        axes[row_index, 0].set_ylabel(
            f"{selection['event']}\n{selection['selection_reason']}", fontsize=8.2, rotation=90, labelpad=8
        )
    legend = [Patch(facecolor=COLORS[index], edgecolor="#7A8491", label=name)
              for index, name in enumerate(CLASS_NAMES)]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.006))
    fig.suptitle("BRIGHT qualitative comparison under one evaluation protocol", fontsize=13, y=1.002)
    fig.tight_layout(rect=(0, 0.035, 1, 0.985), pad=0.55)
    fig.savefig(output_dir / "qualitative_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "qualitative_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    case_dir = output_dir / "qualitative_cases"
    case_dir.mkdir(exist_ok=True)
    for selection in selections:
        name = selection["name"]
        pre, post, gt = load_inputs(dataset_root, name, suffix)
        fig, axes = plt.subplots(1, columns, figsize=(2.1 * columns, 2.45), squeeze=False)
        show_panel(axes[0, 0], pre, "Pre-event RGB")
        show_panel(axes[0, 1], post, "Post-event SAR")
        show_panel(axes[0, 2], gt, "Ground truth", mask=True)
        for col_offset, model in enumerate(models, start=3):
            prediction = imageio.imread(results_root / model / "predictions/raw" / f"{name}.png")
            show_panel(axes[0, col_offset], prediction, DISPLAY_NAMES[model], mask=True)
        fig.suptitle(f"{name} · {selection['selection_reason']}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.91), pad=0.45)
        fig.savefig(case_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_markdown(rows: list[dict], selections: list[dict], output_dir: Path) -> None:
    ordered = sorted(rows, key=lambda row: row["mIoU_foreground_pct"], reverse=True)
    lines = [
        "# BRIGHT统一评估报告",
        "",
        "所有模型采用同一测试列表、预处理、四分类映射和全局像素混淆矩阵。预测未与定位头相乘。",
        "",
        "| Model | 4-class mIoU | Foreground mIoU | Damaged IoU | Damaged F1 | Damaged→Intact | sec/image |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['model']} | {row['mIoU_4class_pct']:.2f} | {row['mIoU_foreground_pct']:.2f} | "
            f"{row['damaged_IoU_pct']:.2f} | {row['damaged_F1_pct']:.2f} | "
            f"{row['damaged_to_intact_pct']:.2f} | {row['sec_per_image']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 自动选择的定性案例",
            "",
            "案例选择只依据GT类别像素量、模型间差异和难度，不依据某个指定模型是否获胜。",
            "",
        ]
    )
    for item in selections:
        lines.append(f"- `{item['name']}` ({item['event']}): {item['selection_reason']}")
    lines.extend(
        [
            "",
            "## 图件",
            "",
            "- `overall_metrics.png/pdf`: 总体指标对比。",
            "- `per_class_iou.png/pdf`: 前景三类IoU。",
            "- `event_foreground_miou_heatmap.png/pdf`: 分事件前景mIoU。",
            "- `normalized_confusion_matrices.png/pdf`: 行归一化混淆矩阵。",
            "- `qualitative_comparison.png/pdf`: RGB、SAR、GT与六模型同图对比。",
            "",
            "说明：SAR仅为逐图1%–99%拉伸后的显示效果；指标使用原始模型输入。",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    output_dir = args.output_dir or (args.results_root / "report")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = build_comparison_rows(args.results_root, args.models)
    write_csv(output_dir / "model_comparison.csv", comparison_rows)
    plot_overall(comparison_rows, output_dir)
    plot_per_class_iou(comparison_rows, output_dir)
    plot_event_heatmap(args.results_root, args.models, output_dir)
    plot_confusion_matrices(args.results_root, args.models, output_dir)

    if args.samples:
        selections = [
            {"name": name, "event": "manual", "selection_reason": "User-selected case"}
            for name in args.samples
        ]
    else:
        selections = select_cases(args.results_root, args.models, args.num_cases)
    if selections:
        plot_qualitative_cases(
            args.results_root, args.dataset_root, args.models, selections, args.suffix, output_dir
        )
    with (output_dir / "selected_samples.json").open("w", encoding="utf-8") as handle:
        json.dump(selections, handle, ensure_ascii=False, indent=2)
    write_markdown(comparison_rows, selections, output_dir)
    print(f"[DONE] Benchmark report: {output_dir}")


if __name__ == "__main__":
    main()
