#!/usr/bin/env python3
"""Train CM0/CM1 for the isolated BRIGHT building-prior ablation."""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from changedetection.experiments.building_prior_ablation_v1.model import VALID_VARIANTS
from changedetection.experiments.building_prior_ablation_v1.trainer import BuildingPriorAblationTrainer
from changedetection.script.script_utils import populate_name_lists


def main():
    parser = argparse.ArgumentParser("ChangeMamba BRIGHT building-prior ablation")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--opts", nargs="+", default=None)
    parser.add_argument("--variant", choices=VALID_VARIANTS, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="BRIGHT")
    parser.add_argument("--model_type", default="ChangeMamba-BuildingPrior-Ablation-V1")
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--encoder_pretrained_path", default=None)
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--resume_training_path", default=None)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_num_workers", type=int, default=2)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--max_iters", type=int, default=40000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--roi_floor", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if bool(args.encoder_pretrained_path) == bool(args.init_checkpoint):
        parser.error(
            "provide exactly one of --encoder_pretrained_path or --init_checkpoint"
        )

    args.type = "train"
    args.shuffle = True
    args.model_checkpoint_path = None
    args.model_param_path = args.output_dir
    args.start_iter = 0
    populate_name_lists(
        args,
        {
            "train_data_list_path": "train_data_name_list",
            "val_data_list_path": "val_data_name_list",
        },
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    BuildingPriorAblationTrainer(args).training()


if __name__ == "__main__":
    main()
