#!/bin/bash

cd "$(dirname "$0")/../.."

python script/standard_ML/train_DamageFormerROIMask.py \
    --dataset 'BRIGHT' \
    --train_batch_size 8 \
    --eval_batch_size 4 \
    --num_workers 8 \
    --crop_size 640 \
    --max_iters 800000 \
    --model_type 'Baseline_ROI_Mask' \
    --learning_rate 1e-4 \
    --model_param_path '../saved_weights' \
    --train_dataset_path './dataset' \
    --train_data_list_path './dataset/splitname/standard_ML/train_set.txt' \
    --val_dataset_path './dataset' \
    --val_data_list_path './dataset/splitname/standard_ML/val_set.txt' \
    --test_dataset_path './dataset' \
    --test_data_list_path './dataset/splitname/standard_ML/test_set.txt'
