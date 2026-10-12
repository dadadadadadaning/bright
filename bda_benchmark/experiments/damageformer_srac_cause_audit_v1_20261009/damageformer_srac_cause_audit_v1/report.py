"""Readable results from cause audit; no model loading or GPU required."""
import argparse
import json
from pathlib import Path


def percent(value):
    return "N/A" if value is None else f"{100*value:.4f}%"


def describe(root):
    result = json.loads((Path(root)/"audit.json").read_text(encoding="utf-8"))
    print("complete:", result["complete"], "smoke:", result["smoke"], "checkpoint step:", result["checkpoint_step"])
    print("reproduction:", result["reproduction"])
    if not result["complete"] or result["smoke"]:
        print("CAUTION: failed or 2-image smoke results are NOT full-Val method evidence.")
    print("\nFull-image foreground mIoU / Damaged P, R, F1")
    for name, m in result["metrics"].items():
        print(f"{name:24s} {percent(m['valid_miou']):>10s}  "
              f"P={percent(m['precision'][2])} R={percent(m['recall'][2])} F1={percent(m['f1'][2])}")
    for left, right in (("a1", "a1_resize"), ("a1_resize", "a1_q"), ("a1_q", "region"),
                        ("region", "region_read_sharp"), ("region", "region_pool_sharp"),
                        ("region", "region_both_sharp"), ("region", "region_split")):
        a, b = result["metrics"][left], result["metrics"][right]
        if a["valid_miou"] is not None and b["valid_miou"] is not None:
            print(f"{right} - {left}: {100*(b['valid_miou']-a['valid_miou']):+.4f} pp")
    print("\nClass IoU (Background, Intact, Damaged, Destroyed):")
    for name in ("a1", "a1_q", "region", "region_both_sharp", "region_split"):
        print(name, [percent(v) for v in result["metrics"][name]["iou"]])
    print("\nPaired harm/rescue (GT Damaged: correct D -> wrong I / wrong I -> correct D):")
    for name, m in result["paired"].items():
        print(name, "harm/rescue/net", m["harmed"], m["rescued"], m["net"],
              "D->I harm / rescue", m["damaged_to_intact_harm"], m["damaged_from_intact_rescue"])
    print("\nConditional errors (counts, not full-image mIoU):")
    for group in ("roi_false_negative", "roi_false_positive", "boundary", "interior", "proxy_small",
                  "B_0_01", "B_01_04", "B_04_06", "B_06_09", "B_09_1", "GT_purity_lt080"):
        print(group)
        for pair, m in result["groups"].get(group, {}).items():
            print(" ", pair, "net", m["net"], "harm_rate", m["harm_rate"],
                  "GT Damaged D->I harm", m["damaged_to_intact_harm"])
    print("\nRouting means (same slot ceiling does not mean same occupied region budget):")
    print(json.dumps(result["mean_routing_per_image"], indent=2))
    print("\nClass-conditional GT mixing and Q entropy:")
    print(json.dumps(result["mixing_by_GT_class"], indent=2))
    print("\nPer-event foreground mIoU: A1 / region / both-sharp / predicted split")
    for ev, modes in result["per_event"].items():
        print(ev, " / ".join(percent(modes[key]["valid_miou"]) for key in
                             ("a1", "region", "region_both_sharp", "region_split")))
    print("\nThis audit does not test unfreezing. Use a separate matched training probe for fusion adaptation.")
    print("Sharpen/split results are inference interventions, not retrained ablations. GT components are proxies, not instances.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    describe(parser.parse_args().root)
