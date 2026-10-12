#!/usr/bin/env bash
source "$(dirname "$0")/common_env.sh"
CUDA_VISIBLE_DEVICES=0 python -u script/standard_ML/train_Priority_Modular.py   --stage B0C --exp_name B0C_EBS16_50k --optimizer_steps 50000 "${BASE_ARGS[@]}" "$@"
