"""Validation-only trainer for the controlled building-prior ablation."""
from __future__ import annotations

import json
import os

import torch

from changedetection.checkpoints import load_model_weights
from changedetection.datasets import build_eval_loader
from changedetection.logging_utils import format_log_block
from changedetection.script.script_utils import get_vssm_kwargs
from changedetection.tasks.bright import BRIGHTTrainer

from .model import build_model


def _jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


class BuildingPriorAblationTrainer(BRIGHTTrainer):
    """Same BRIGHT optimisation for CM0 and CM1; Validation only."""

    save_iteration_checkpoints = False

    def __init__(self, args):
        self.variant = args.variant
        self.initialization_report = None
        super().__init__(args)
        self.eval_interval = int(args.eval_every)

    def _resolve_model_save_path(self):
        return os.path.abspath(self.args.output_dir)

    def build_model(self, config):
        model = build_model(
            self.variant,
            pretrained=self.args.encoder_pretrained_path,
            roi_floor=self.args.roi_floor,
            **get_vssm_kwargs(config),
        )
        if self.args.init_checkpoint:
            report = load_model_weights(model, self.args.init_checkpoint)
            expected_missing = (
                set()
                if self.variant == "cm0_original"
                else {"building_transition"}
            )
            observed_missing = set(report["missing_keys"])
            if (
                observed_missing != expected_missing
                or report["unexpected_keys"]
                or report["mismatched_keys"]
            ):
                raise RuntimeError(
                    "full-model initialisation contract failed: "
                    f"variant={self.variant}, expected_missing={sorted(expected_missing)}, "
                    f"report={report}"
                )
            self.initialization_report = report
        return model

    def build_eval_loaders(self):
        # Deliberately do not construct a Test loader during model selection.
        return {
            "Validation": build_eval_loader(
                self.args.dataset,
                dataset_path=self.args.val_dataset_path,
                data_list=self.args.val_data_name_list,
                crop_size=self.args.crop_size,
                split="val",
                batch_size=self.args.eval_batch_size,
                num_workers=self.args.eval_num_workers,
            )
        }

    def evaluate_loader(self, split_name, data_loader):
        metrics = super().evaluate_loader(split_name, data_loader)
        transition = getattr(self.model, "building_transition", None)
        metrics["building_transition"] = (
            None
            if transition is None
            else float(torch.tanh(transition.detach()).cpu())
        )
        return metrics

    def format_best_result(self, best_record):
        validation = best_record["results"]["Validation"]
        return format_log_block(
            "BEST Validation",
            {
                "loc_F1_pct": 100 * validation["loc_f1"],
                "clf_F1_pct": 100 * validation["clf_f1"],
                "OA_pct": 100 * validation["oa"],
                "mIoU_pct": 100 * validation["miou"],
                "sub_IoU_pct": 100 * torch.as_tensor(validation["sub_iou"]),
                "building_transition": validation["building_transition"],
            },
            meta={"iter": best_record["iteration"], "score": best_record["score"]},
        )

    def training(self):
        os.makedirs(self.model_save_path, exist_ok=True)
        contract = {
            "protocol": "CHANGEMAMBA_BUILDING_PRIOR_ABLATION_V1",
            "variant": self.variant,
            "training_steps": self.args.max_iters,
            "eval_every": self.eval_interval,
            "selection": "Validation four-class mIoU",
            "test_during_training": False,
            "save_policy": "best_model.pth plus overwritten latest.pth only",
            "seed": self.args.seed,
            "roi_floor": self.args.roi_floor,
            "encoder_pretrained_path": self.args.encoder_pretrained_path,
            "init_checkpoint": self.args.init_checkpoint,
            "initialization_report": self.initialization_report,
        }
        with open(
            os.path.join(self.model_save_path, "experiment_contract.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(contract, handle, ensure_ascii=False, indent=2)

        super().training()

        checkpoint = torch.load(
            os.path.join(self.model_save_path, self.best_checkpoint_name),
            map_location="cpu",
            weights_only=False,
        )
        summary = {
            "contract": contract,
            "selected_iteration": checkpoint.get("iteration"),
            "best_score": checkpoint.get("best_score"),
            "best_record": checkpoint.get("best_record"),
            "test_accessed": False,
        }
        with open(
            os.path.join(self.model_save_path, "training_summary.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(_jsonable(summary), handle, ensure_ascii=False, indent=2)
