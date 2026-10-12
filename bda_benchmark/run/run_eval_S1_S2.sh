#!/usr/bin/env bash
set -euo pipefail
cd /root/shared-nvme/BRIGHT-master/bda_benchmark

S2_CKPT="/root/shared-nvme/BRIGHT-checkpoints/S2/BRIGHT/S2_BGMP_K1_EBS16_50k_20260710_114832/best_model.pth"
S1_CKPT="/root/shared-nvme/BRIGHT-checkpoints/S1/BRIGHT/S1_BG_alpha01_EBS16_50k_20260710_114320/best_model.pth"
mkdir -p /root/shared-nvme/BRIGHT-eval-results

CUDA_VISIBLE_DEVICES=0 python script/standard_ML/eval_damageformer_checkpoint.py \
  --checkpoint "$S1_CKPT" \
  --model_variant s1 \
  --roi_alpha 0.1 \
  --eval_batch_size 1 \
  --val_dataset_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --val_data_list_path dataset/splitname/standard_ML/val_set.txt \
  --test_dataset_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --test_data_list_path dataset/splitname/standard_ML/test_set.txt \
  --num_workers 4 \
  --output_json /root/shared-nvme/BRIGHT-eval-results/S1_best_eval.json

CUDA_VISIBLE_DEVICES=0 python script/standard_ML/eval_damageformer_checkpoint.py \
  --checkpoint "$S2_CKPT" \
  --model_variant s2 \
  --roi_alpha 0.1 \
  --proto_tau 0.1 \
  --proto_warmup_iters 5000 \
  --proto_eval_mode both \
  --eval_batch_size 1 \
  --val_dataset_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --val_data_list_path dataset/splitname/standard_ML/val_set.txt \
  --test_dataset_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --test_data_list_path dataset/splitname/standard_ML/test_set.txt \
  --num_workers 4 \
  --output_json /root/shared-nvme/BRIGHT-eval-results/S2_best_eval.json
