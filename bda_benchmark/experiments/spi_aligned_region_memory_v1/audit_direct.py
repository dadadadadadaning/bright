#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path

import torch

from bda_benchmark.experiments.spi_factorized_cross_modal_token_v1.core import FactorizedKeyEncoder, groups_from_payload
from bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.audit_retrieval import report


def main() -> None:
    parser = argparse.ArgumentParser("Region-only Direct Head audit; no Memory is loaded")
    parser.add_argument("--records", required=True); parser.add_argument("--metric", required=True)
    parser.add_argument("--output_json", required=True); parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0"); args = parser.parse_args()
    records = torch.load(args.records, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.metric, map_location="cpu", weights_only=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = FactorizedKeyEncoder(checkpoint["group_dims"], checkpoint["variant"],
                                 key_dim=checkpoint["key_dim"], hidden_dim=checkpoint["hidden_dim"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    predictions = []
    with torch.inference_mode():
        for start in range(0, records["labels"].numel(), args.batch_size):
            stop = min(start + args.batch_size, records["labels"].numel())
            index = torch.arange(start, stop)
            _, logits = model(groups_from_payload(records, index, device))
            predictions.append((logits.argmax(1) + 1).cpu())
    direct = report(records["labels"], torch.cat(predictions), records["events"], include_event_class_metrics=True)
    payload = {"protocol": "SPI_ALIGNED_REGION_MEMORY_V1_DIRECT_AUDIT",
               "region_variant": records["metadata"]["region_variant"],
               "direct_metric_classifier": direct, "memory_loaded": False, "test_accessed": False}
    output = Path(args.output_json); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()

