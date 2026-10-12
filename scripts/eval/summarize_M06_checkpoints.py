import re
import csv
from pathlib import Path

root = Path("/root/shared-nvme/BRIGHT-master/eval_results/BGCA_final/M06_BGCA_dilatedROI_align_noOrdinal_checkpoints")
out_csv = root / "M06_checkpoint_summary_from_log.csv"

def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")

def find_float(pattern: str, text: str):
    m = re.search(pattern, text)
    return m.group(1) if m else ""

def checkpoint_sort_key(name: str):
    if name == "best_model":
        return 999999
    m = re.search(r"iter_(\d+)", name)
    return int(m.group(1)) if m else 999998

rows = []

for d in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: checkpoint_sort_key(p.name)):
    log_path = d / "eval.log"
    txt_path = d / "summary.txt"

    text = read_text(log_path)
    if not text:
        text = read_text(txt_path)

    row = {
        "checkpoint": d.name,
        "loc_f1": "",
        "loc_iou": "",
        "valid_mean_f1": "",
        "valid_miou": "",
        "final_miou": "",
        "intact_f1": "",
        "intact_iou": "",
        "damaged_f1": "",
        "damaged_iou": "",
        "destroyed_f1": "",
        "destroyed_iou": "",
        "damaged_to_destroyed_rate": "",
        "destroyed_to_damaged_rate": "",
        "damaged_destroyed_pair_confusion_rate": "",
        "ordinal_mae_valid_pred": "",
        "ordinal_severe_error_rate_valid_pred": "",
        "pred_background_on_gt_building_rate": "",
        "log_exists": "yes" if text else "no",
    }

    if text:
        row["loc_f1"] = find_float(r"Loc F1:\s*([0-9.eE+-]+)", text)
        row["loc_iou"] = find_float(r"Loc F1:\s*[0-9.eE+-]+,\s*Loc IoU:\s*([0-9.eE+-]+)", text)

        row["valid_mean_f1"] = find_float(r"Valid Mean F1:\s*([0-9.eE+-]+)", text)
        row["valid_miou"] = find_float(r"Valid Mean F1:\s*[0-9.eE+-]+,\s*Valid mIoU:\s*([0-9.eE+-]+)", text)

        row["final_miou"] = find_float(r"Final mIoU:\s*([0-9.eE+-]+)", text)

        row["intact_f1"] = find_float(r"1-intact:\s*F1=([0-9.eE+-]+)", text)
        row["intact_iou"] = find_float(r"1-intact:\s*F1=[0-9.eE+-]+,\s*IoU=([0-9.eE+-]+)", text)

        row["damaged_f1"] = find_float(r"2-damaged:\s*F1=([0-9.eE+-]+)", text)
        row["damaged_iou"] = find_float(r"2-damaged:\s*F1=[0-9.eE+-]+,\s*IoU=([0-9.eE+-]+)", text)

        row["destroyed_f1"] = find_float(r"3-destroyed:\s*F1=([0-9.eE+-]+)", text)
        row["destroyed_iou"] = find_float(r"3-destroyed:\s*F1=[0-9.eE+-]+,\s*IoU=([0-9.eE+-]+)", text)

        row["damaged_to_destroyed_rate"] = find_float(r"damaged\s*->\s*destroyed rate:\s*([0-9.eE+-]+)", text)
        row["destroyed_to_damaged_rate"] = find_float(r"destroyed\s*->\s*damaged rate:\s*([0-9.eE+-]+)", text)
        row["damaged_destroyed_pair_confusion_rate"] = find_float(r"damaged/destroyed pair confusion rate:\s*([0-9.eE+-]+)", text)

        row["ordinal_mae_valid_pred"] = find_float(r"ordinal MAE\(valid pred\):\s*([0-9.eE+-]+)", text)
        row["ordinal_severe_error_rate_valid_pred"] = find_float(r"severe ordinal error rate\(valid pred\):\s*([0-9.eE+-]+)", text)
        row["pred_background_on_gt_building_rate"] = find_float(r"pred background on GT building rate:\s*([0-9.eE+-]+)", text)

    rows.append(row)

fieldnames = list(rows[0].keys()) if rows else []

with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved: {out_csv}")