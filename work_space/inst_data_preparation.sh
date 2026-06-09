#!/usr/bin/env bash
# Run the simple sliding-window anomaly data preparation.
# Override any variable from the command line:
#   OUTPUT_DIR=./my_data bash inst_data_preparation.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-/home/rajib/mTSBench/Datasets/mTSBench}"
OUTPUT_DIR="${OUTPUT_DIR:-./prepared_data_simple}"
MIN_LENGTH="${MIN_LENGTH:-50}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-64}"
STRIDE="${STRIDE:-64}"

# Normal signal (256 steps) is always prepended as an instruction prefix.   
# Total model input = 256 (normal) + 512 (context) + 64 (future) = 832 steps.
# Pairs are split by future type into two datasets: normal/ and anomaly/.

python inst_data_prepare_simple.py \
    --data_dir          "$DATA_DIR"          \
    --output_dir        "$OUTPUT_DIR"        \
    --min_length        "$MIN_LENGTH"        \
    --val_fraction      "$VAL_FRACTION"      \
    --context_length    "$CONTEXT_LENGTH"    \
    --prediction_length "$PREDICTION_LENGTH" \
    --stride            "$STRIDE"

echo "Done. Outputs written to $OUTPUT_DIR/normal/ and $OUTPUT_DIR/anomaly/"
