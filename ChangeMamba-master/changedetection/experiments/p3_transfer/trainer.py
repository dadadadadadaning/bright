"""Validation-only trainer for controlled P3-to-ChangeMamba transfers."""
from __future__ import annotations

import json
import os

import torch
import torch.optim as optim

from changedetection.experiments.native_sar.dataset import (
    build_native_sar_loader,
    load_sar_stats,
)
from changedetection.experiments.native_sar.trainer import evaluate_native_sar_model
from changedetection.logging_utils import format_log_block
from changedetection.script.script_utils import get_vssm_kwargs
from changedetection.tasks.bright import BRIGHTTrainer

from .model import (
    ChangeMambaP3Transfer,
    freeze_for_p3_transfer,
    load_parent_checkpoint,
)


class P3TransferChangeMambaTrainer(BRIGHTTrainer):
    """Train only NativeSAR and declared P3 structural transfer parameters."""

    def __init__(self, args):
        self.sar_stats = load_sar_stats(args.sar_stats_path)
        self.migration_report = None
        self.trainable_report = None
        super().__init__(args)
        self.eval_interval = int(args.eval_every)
        expected = 31 if args.variant == "cm_p2" else 289
        if self.trainable_report is None:
            raise RuntimeError("P3-transfer freeze policy was not applied")
        if self.trainable_report["trainable"] != expected:
            raise RuntimeError(
                "P3-transfer trainable-parameter contract changed: "
                f"expected={expected}, observed={self.trainable_report}"
            )

    def _resolve_model_save_path(self):
        return os.path.abspath(self.args.output_dir)

    def build_model(self, config):
        model = ChangeMambaP3Transfer(
            output_building=2,
            output_damage=4,
            pretrained=None,
            sar_mean=self.sar_stats["mean"],
            sar_std=self.sar_stats["std"],
            variant=self.args.variant,
            roi_floor=self.args.roi_floor,
            **get_vssm_kwargs(config),
        )
        self.migration_report = load_parent_checkpoint(
            model, self.args.parent_checkpoint
        )
        # BaseTrainer builds the optimizer immediately after build_model, so
        # the freeze contract must be established here rather than afterwards.
        self.trainable_report = freeze_for_p3_transfer(model)
        return model

    def build_train_loader(self):
        return build_native_sar_loader(
            dataset_path=self.args.train_dataset_path,
            data_list=self.args.train_data_name_list,
            crop_size=self.args.crop_size,
            sar_stats_path=self.args.sar_stats_path,
            split="train",
            batch_size=self.args.train_batch_size,
            num_workers=self.args.num_workers,
            max_iters=self.args.max_iters,
            shuffle=True,
        )

    def build_eval_loaders(self):
        return {
            "Validation": build_native_sar_loader(
                dataset_path=self.args.val_dataset_path,
                data_list=self.args.val_data_name_list,
                crop_size=self.args.crop_size,
                sar_stats_path=self.args.sar_stats_path,
                split="val",
                batch_size=self.args.eval_batch_size,
                num_workers=max(1, self.args.num_workers // 2),
                shuffle=False,
            )
        }

    def build_optimizer(self):
        parameters = [value for value in self.model.parameters() if value.requires_grad]
        return optim.AdamW(
            parameters,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def train_step(self, batch):
        # Keep the accepted body deterministic while retaining its differentiable
        # path to the adapter, ROI transition, and optional hierarchy heads.
        self.model.eval()
        self.model.sar_adapter.train()
        if self.model.variant == "cm_p3":
            self.model.occurrence_delta.train()
            self.model.severity_delta.train()
        return super().train_step(batch)

    def evaluate_loader(self, split_name, data_loader):
        metrics = evaluate_native_sar_model(self.model, data_loader, self.device)
        metrics["building_transition"] = float(
            torch.tanh(self.model.building_transition.detach()).cpu()
        )
        return metrics

    def format_best_result(self, best_record):
        metrics = best_record["results"]["Validation"]
        return format_log_block(
            "BEST Validation",
            {
                "loc_F1_pct": 100 * metrics["loc_f1"],
                "clf_F1_pct": 100 * metrics["clf_f1"],
                "OA_pct": 100 * metrics["oa"],
                "mIoU_pct": 100 * metrics["miou"],
                "sub_IoU_pct": 100 * torch.as_tensor(metrics["sub_iou"]),
                "building_transition": metrics["building_transition"],
            },
            meta={"iter": best_record["iteration"], "score": best_record["score"]},
        )

    def _admit_initial_baseline(self):
        metrics = self.evaluate_loader("Validation", self.eval_loaders["Validation"])
        observed = 100.0 * metrics["miou"]
        if abs(observed - self.args.expected_parent_miou) > self.args.baseline_tolerance_pp:
            raise RuntimeError(
                "step-0 function-preserving guard failed: "
                f"observed={observed:.4f}, expected={self.args.expected_parent_miou:.4f} "
                f"+/- {self.args.baseline_tolerance_pp:.4f} pp"
            )
        record = {
            "iteration": 0,
            "score": metrics["miou"],
            "results": {"Validation": metrics},
            "checkpoint_path": os.path.join(
                self.model_save_path, self.best_checkpoint_name
            ),
        }
        self.best_score = metrics["miou"]
        self.best_record = record
        self.save_checkpoint(
            self.best_checkpoint_name,
            0,
            self.best_score,
            record,
            extra_state={"safe_initial_state": True, "test_accessed": False},
        )
        self.emit_log(
            self.format_eval_result(
                "Initial Validation", 0, len(self.train_loader), metrics
            )
        )

    def training(self):
        os.makedirs(self.model_save_path, exist_ok=True)
        contract = {
            **self.model.experiment_metadata(),
            "parent_checkpoint": os.path.abspath(self.args.parent_checkpoint),
            "trainable": self.trainable_report,
            "training_steps": self.args.max_iters,
            "selection": "Validation mIoU including exact step-0 parent",
            "test_during_training": False,
            "migration": self.migration_report,
        }
        with open(
            os.path.join(self.model_save_path, "experiment_contract.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(contract, handle, ensure_ascii=False, indent=2)
        self._admit_initial_baseline()
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
            json.dump(summary, handle, ensure_ascii=False, indent=2)


__all__ = ["P3TransferChangeMambaTrainer"]
