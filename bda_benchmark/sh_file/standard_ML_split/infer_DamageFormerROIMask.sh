#!/bin/bash

cd "$(dirname "$0")/../.."

python script/standard_ML/infer_DamageFormerROIMask.py \
    --model_path '../saved_weights/BRIGHT/Baseline_ROI_Mask_YYYYMMDD_HHMMSS/best_model.pth' \
    --test_dataset_path './dataset' \
    --test_data_list_path './dataset/splitname/standard_ML/test_set.txt' \
    --output_dir '../inference_results_DamageFormerROIMask'
