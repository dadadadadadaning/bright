"""Validation-only training protocol for the isolated native-SAR adapter."""
from __future__ import annotations

import json
import os

import torch
import torch.optim as optim

from changedetection.evaluation import (
    BinaryChangeEvaluator,
    DamageClassificationEvaluator,
    MultiClassEvaluator,
)
from changedetection.logging_utils import format_log_block
from changedetection.models.ChangeMambaMMBDA import ChangeMambaMMBDA
from changedetection.script.script_utils import get_vssm_kwargs
from changedetection.tasks.bright import BRIGHTTrainer

from .dataset import build_native_sar_loader, load_sar_stats
from .model import (
    ChangeMambaMMBDANativeSAR,
    freeze_except_native_adapter,
    load_baseline_as_native,
)


def evaluate_native_sar_model(model, data_loader, device):
    evaluator_loc = BinaryChangeEvaluator()
    evaluator_clf = DamageClassificationEvaluator(num_classes=4)
    evaluator_total = MultiClassEvaluator(num_classes=4)
    model.eval()
    with torch.no_grad():
        for pre_img, post_sar, label_loc, label_clf, _ in data_loader:
            pre_img = pre_img.to(device)
            post_sar = post_sar.to(device)
            label_loc = label_loc.to(device).long()
            label_clf = label_clf.to(device).long()
            output_loc, output_clf = model(pre_img, post_sar)
            pred_loc = output_loc.argmax(dim=1).cpu().numpy()
            pred_clf = output_clf.argmax(dim=1).cpu().numpy()
            label_loc_np = label_loc.cpu().numpy()
            label_clf_np = label_clf.cpu().numpy()
            evaluator_loc.add_batch(label_loc_np, pred_loc)
            evaluator_clf.add_batch(
                label_clf_np[label_loc_np > 0], pred_clf[label_loc_np > 0]
            )
            evaluator_total.add_batch(label_clf_np, pred_clf)
    loc = evaluator_loc.compute()
    damage = evaluator_clf.compute()
    total = evaluator_total.compute()
    return {
        "loc_f1": float(loc.f1),
        "clf_f1": float(damage.harmonic_mean_f1),
        "oa": float(total.oa),
        "miou": float(total.miou),
        "sub_iou": [float(value) for value in total.iou_per_class],
    }


class NativeSARChangeMambaTrainer(BRIGHTTrainer):
    """Only the 30-parameter SAR adapter is trainable for 3000 steps."""

    def __init__(self, args):
        self.sar_stats = load_sar_stats(args.sar_stats_path)
        self.migration_report = None
        super().__init__(args)
        self.eval_interval = int(args.eval_every)
        self.trainable_report = freeze_except_native_adapter(self.model)
        if self.trainable_report["trainable"] != 30:
            raise RuntimeError(
                "native adapter trainable-parameter contract changed: "
                f"{self.trainable_report}"
            )

    def _resolve_model_save_path(self):
        return os.path.abspath(self.args.output_dir)

    def build_model(self, config):
        model = ChangeMambaMMBDANativeSAR(
            output_building=2,
            output_damage=4,
            pretrained=None,
            sar_mean=self.sar_stats["mean"],
            sar_std=self.sar_stats["std"],
            **get_vssm_kwargs(config),
        )
        self.migration_report = load_baseline_as_native(
            model, self.args.baseline_checkpoint
        )
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
        # Test is intentionally absent during training.
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
        return optim.AdamW(
            self.model.sar_adapter.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def train_step(self, batch):
        # The accepted ChangeMamba body is a fixed teacher-like feature path.
        # Keep DropPath/dropout deterministic while retaining gradients through
        # it to the small input adapter.
        self.model.eval()
        self.model.sar_adapter.train()
        return super().train_step(batch)

    def evaluate_loader(self, split_name, data_loader):
        return evaluate_native_sar_model(self.model, data_loader, self.device)

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
            },
            meta={"iter": best_record["iteration"], "score": best_record["score"]},
        )

    def _admit_initial_baseline(self):
        metrics = self.evaluate_loader("Validation", self.eval_loaders["Validation"])
        observed = 100.0 * metrics["miou"]
        if abs(observed - self.args.expected_baseline_miou) > self.args.baseline_tolerance_pp:
            raise RuntimeError(
                "step-0 function-preserving guard failed: "
                f"observed={observed:.4f}, expected={self.args.expected_baseline_miou:.4f} "
                f"+/- {self.args.baseline_tolerance_pp:.4f} pp"
            )
        record = {
            "iteration": 0,
            "score": metrics["miou"],
            "results": {"Validation": metrics},
            "checkpoint_path": os.path.join(self.model_save_path, self.best_checkpoint_name),
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
        self.emit_log(self.format_eval_result("Initial Validation", 0, len(self.train_loader), metrics))

    def training(self):
        os.makedirs(self.model_save_path, exist_ok=True)
        contract = {
            "family": "ChangeMamba NativeSAR-1ch input-only adaptation",
            "baseline_checkpoint": os.path.abspath(self.args.baseline_checkpoint),
            "trainable": ["sar_adapter.weight", "sar_adapter.bias"],
            "trainable_parameters": self.trainable_report["trainable"],
            "training_steps": self.args.max_iters,
            "selection": "Validation mIoU including exact step-0 baseline",
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
