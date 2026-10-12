#!/usr/bin/env python3
"""Train CM-P2/CM-P3 controlled structural transfers."""
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from changedetection.experiments.p3_transfer.model import VALID_P3_TRANSFER_VARIANTS
from changedetection.experiments.p3_transfer.trainer import P3TransferChangeMambaTrainer
from changedetection.script.script_utils import populate_name_lists


def main():
    parser = argparse.ArgumentParser("ChangeMamba-BRIGHT P3 structural transfer")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--opts", nargs="+", default=None)
    parser.add_argument("--variant", choices=VALID_P3_TRANSFER_VARIANTS, required=True)
    parser.add_argument("--dataset", default="BRIGHT")
    parser.add_argument("--model_type", default="ChangeMamba-P3-Transfer")
    parser.add_argument("--parent_checkpoint", required=True)
    parser.add_argument("--sar_stats_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_dataset_path", required=True)
    parser.add_argument("--train_data_list_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    parser.add_argument("--val_data_list_path", required=True)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--max_iters", type=int, default=3000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--roi_floor", type=float, default=0.10)
    parser.add_argument("--expected_parent_miou", type=float, default=64.2585)
    parser.add_argument("--baseline_tolerance_pp", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model_param_path", default="")
    parser.add_argument("--start_iter", type=int, default=0)
    args = parser.parse_args()
    args.type = "train"
    args.shuffle = True
    args.encoder_pretrained_path = None
    args.model_checkpoint_path = None
    args.resume_training_path = None
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
    P3TransferChangeMambaTrainer(args).training()


if __name__ == "__main__":
    main()
