"""Validation-only trainer for the controlled building-prior ablation."""
from __future__ import annotations

import json
import math
import os

import torch
from torch.utils.data import DataLoader

from changedetection.checkpoints import load_model_weights
from changedetection.datasets import build_dataset, build_eval_loader
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
        self.stable_training = self.variant == "cm1s_building_prior_stable"
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

    def build_train_loader(self):
        """Build micro-batches while keeping max_iters as optimizer steps."""
        micro_iterations = int(self.args.max_iters)
        if self.stable_training:
            micro_iterations *= int(self.args.accumulation_steps)
        dataset = build_dataset(
            self.args.dataset,
            dataset_path=self.args.train_dataset_path,
            data_list=self.args.train_data_name_list,
            crop_size=self.args.crop_size,
            max_iters=micro_iterations,
            batch_size=self.args.train_batch_size,
            split=self.args.type,
        )
        return DataLoader(
            dataset,
            batch_size=self.args.train_batch_size,
            shuffle=self.args.shuffle,
            num_workers=self.args.num_workers,
            drop_last=False,
        )

    def build_optimizer(self):
        if not self.stable_training:
            return super().build_optimizer()
        transition = []
        shared = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name == "building_transition":
                transition.append(parameter)
            else:
                shared.append(parameter)
        if len(transition) != 1:
            raise RuntimeError(
                "stable CM1-S expects exactly one building_transition parameter, "
                f"found {len(transition)}"
            )
        return torch.optim.AdamW(
            [
                {
                    "params": shared,
                    "lr": self.args.backbone_learning_rate,
                    "group_name": "shared_model",
                },
                {
                    "params": transition,
                    "lr": self.args.new_module_learning_rate,
                    "group_name": "building_prior",
                },
            ],
            weight_decay=self.args.weight_decay,
        )

    def build_scheduler(self):
        if not self.stable_training:
            return super().build_scheduler()
        warmup = int(self.args.warmup_steps)
        total = int(self.args.max_iters)

        def multiplier(step):
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(warmup)
            progress = float(step - warmup) / float(max(1, total - warmup))
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=[multiplier, multiplier],
        )

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

    def _stable_training_loop(self):
        """Gradient-accumulated CM1-S loop; iteration means optimizer step."""
        accumulation = int(self.args.accumulation_steps)
        total_steps = int(self.args.max_iters)
        best_record = self.best_record
        best_score = self.best_score
        optimizer_step = int(self.start_iteration)
        valid_micro = 0
        running = {"loc_loss": 0.0, "clf_loss": 0.0}

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for batch in self.train_loader:
            if optimizer_step >= total_steps:
                break
            step_output = self.train_step(batch)
            if step_output is None:
                continue
            (step_output["loss"] / accumulation).backward()
            valid_micro += 1
            for key in running:
                running[key] += float(step_output["log_items"][key])
            if valid_micro % accumulation:
                continue

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.grad_clip_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.after_optimizer_step()
            optimizer_step += 1

            if optimizer_step % self.log_interval == 0:
                transition = torch.tanh(
                    self.model.building_transition.detach()
                ).item()
                log_items = {
                    key: value / float(self.log_interval * accumulation)
                    for key, value in running.items()
                }
                log_items.update(
                    {
                        "lr_shared": self.optimizer.param_groups[0]["lr"],
                        "lr_prior": self.optimizer.param_groups[1]["lr"],
                        "building_transition": transition,
                    }
                )
                self.emit_log(
                    self.format_train_log(optimizer_step, total_steps, log_items)
                )
                running = {"loc_loss": 0.0, "clf_loss": 0.0}

            if self.eval_loaders and optimizer_step % self.eval_interval == 0:
                self.model.eval()
                eval_results = {}
                for split_name, loader in self.eval_loaders.items():
                    eval_results[split_name] = self.evaluate_loader(split_name, loader)
                    self.emit_log(
                        self.format_eval_result(
                            split_name,
                            optimizer_step,
                            total_steps,
                            eval_results[split_name],
                        )
                    )
                score = self.selection_metric(eval_results)
                if best_score is None or score > best_score:
                    best_score = score
                    best_record = {
                        "iteration": optimizer_step,
                        "score": score,
                        "results": eval_results,
                        "checkpoint_path": os.path.join(
                            self.model_save_path, self.best_checkpoint_name
                        ),
                    }
                    self.save_checkpoint(
                        self.best_checkpoint_name,
                        optimizer_step,
                        best_score,
                        best_record,
                        extra_state={"last_eval_results": eval_results},
                    )
                self.save_checkpoint(
                    self.latest_checkpoint_name,
                    optimizer_step,
                    best_score,
                    best_record,
                    extra_state={"last_eval_results": eval_results},
                )
                self.model.train()

        if best_record is not None:
            self.emit_log(self.format_best_result(best_record))
        else:
            raise RuntimeError("CM1-S finished without an evaluated checkpoint")

    def training(self):
        os.makedirs(self.model_save_path, exist_ok=True)
        contract = {
            "protocol": "CHANGEMAMBA_BUILDING_PRIOR_ABLATION_V1",
            "variant": self.variant,
            "training_steps": self.args.max_iters,
            "step_unit": "optimizer_step",
            "micro_batch_size": self.args.train_batch_size,
            "accumulation_steps": (
                self.args.accumulation_steps if self.stable_training else 1
            ),
            "effective_batch_size": self.args.train_batch_size
            * (self.args.accumulation_steps if self.stable_training else 1),
            "backbone_learning_rate": (
                self.args.backbone_learning_rate if self.stable_training else self.args.learning_rate
            ),
            "new_module_learning_rate": (
                self.args.new_module_learning_rate if self.stable_training else None
            ),
            "warmup_steps": self.args.warmup_steps if self.stable_training else 0,
            "scheduler": "linear-warmup + cosine" if self.stable_training else "none",
            "grad_clip_norm": self.args.grad_clip_norm if self.stable_training else None,
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

        if self.stable_training:
            self._stable_training_loop()
        else:
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
