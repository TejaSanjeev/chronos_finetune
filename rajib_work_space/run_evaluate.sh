#!/bin/bash

python forward.py \
    --split_ratio 0.1 \
    --horizon 5 \
    --context_length 1024 \
    --normal_signal_length 256 \
    --gpu 0 \
    --score_method mse \
    --agg_method l2 \
    --smooth_window 10 \
    --sliding_window_VUS 100 \
    --vus_version opt \
    --vus_thre 250
