#!/bin/bash
# Zero-shot anomaly-detection evaluation (VUS metrics) with the fine-tuned
# Chronos-2 checkpoint.

python evaluate_zero_shot.py \
    --split_ratio 0.1 \
    --horizon 64 \
    --context_length 512 \
    --gpu 0 \
    --score_method smape \
    --agg_method max \
    --smooth_window 5 \
    --sliding_window_VUS 100 \
    --vus_version opt \
    --vus_thre 250
