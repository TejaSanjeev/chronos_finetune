#!/usr/bin/env bash
# Fine-tune Chronos-2 WHILE logging per-window training loss, then plot it.
#
#   Step 1: finetune_anomaly_loss_logging.py  -> <OUTPUT_DIR>/window_loss_history.csv
#   Step 2: plot_window_loss_history.py        -> window_loss_{anomaly,normal,all}.png
#                                                 + non_optimizing_windows.csv
#
# Usage:
#   bash run_finetune_logged.sh                       # all defaults (V4 config)
#   NUM_STEPS=5000 bash run_finetune_logged.sh        # override any var below
#   SKIP_TRAIN=1 bash run_finetune_logged.sh          # only re-make the plots
#   METRIC=train_contrib bash run_finetune_logged.sh  # plot the hinge contribution

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${REPO_ROOT}/rajib_work_space"

# ── Paths ────────────────────────────────────────────────────────────────────
PREPARED_DIR="${PREPARED_DIR:-${WS}/prepared_data_labeled}"
OUTPUT_DIR="${OUTPUT_DIR:-${WS}/chronos2-run-logged}"

# ── Model / mode ─────────────────────────────────────────────────────────────
MODEL_ID="${MODEL_ID:-amazon/chronos-2}"
DEVICE="${DEVICE:-cuda}"
FINETUNE_MODE="${FINETUNE_MODE:-lora}"

# ── Training hyperparameters (match run_finetune.sh / V4) ────────────────────
PREDICTION_LENGTH="${PREDICTION_LENGTH:-64}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-768}"        # = NORMAL_SIGNAL_LENGTH + actual context
NUM_STEPS="${NUM_STEPS:-3000}"
LR="${LR:-}"                                   # blank -> script default
BATCH_SIZE="${BATCH_SIZE:-160}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
FP16="${FP16:-1}"
NO_VALIDATION="${NO_VALIDATION:-1}"

# ── Anomaly / hinge ──────────────────────────────────────────────────────────
ANOMALY_THRESHOLD="${ANOMALY_THRESHOLD:-10}"
MARGIN_TAU="${MARGIN_TAU:-2.0}"
MARGIN_LAMBDA="${MARGIN_LAMBDA:-1.0}"

# ── LoRA ─────────────────────────────────────────────────────────────────────
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.01}"

# ── [SEP] token ──────────────────────────────────────────────────────────────
ENABLE_SEP_TOKEN="${ENABLE_SEP_TOKEN:-1}"
NORMAL_SIGNAL_LENGTH="${NORMAL_SIGNAL_LENGTH:-256}"
INPUT_PATCH_SIZE="${INPUT_PATCH_SIZE:-16}"

# ── Plotting ─────────────────────────────────────────────────────────────────
METRIC="${METRIC:-raw_loss}"                   # raw_loss | train_contrib
NBINS="${NBINS:-20}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"                   # 1 = only run the plotting step

echo "======================================================"
echo "  Chronos-2 fine-tune + per-window loss logging"
echo "    OUTPUT_DIR = $OUTPUT_DIR"
echo "    NUM_STEPS  = $NUM_STEPS   BATCH_SIZE = $BATCH_SIZE   GRAD_ACCUM = $GRAD_ACCUM"
echo "    MARGIN_TAU = $MARGIN_TAU  MARGIN_LAMBDA = $MARGIN_LAMBDA"
echo "    PLOT METRIC = $METRIC     NBINS = $NBINS     SKIP_TRAIN = $SKIP_TRAIN"
echo "======================================================"

# ── Step 1: training with logging ────────────────────────────────────────────
if [ "${SKIP_TRAIN}" != "1" ]; then
    ARGS=(
        --model_id                    "$MODEL_ID"
        --device                      "$DEVICE"
        --data_dir                    "$PREPARED_DIR"
        --output_dir                  "$OUTPUT_DIR"
        --finetune_mode               "$FINETUNE_MODE"
        --prediction_length           "$PREDICTION_LENGTH"
        --context_length              "$CONTEXT_LENGTH"
        --num_steps                   "$NUM_STEPS"
        --batch_size                  "$BATCH_SIZE"
        --gradient_accumulation_steps "$GRAD_ACCUM"
        --logging_steps               "$LOGGING_STEPS"
        --warmup_ratio                "$WARMUP_RATIO"
        --lr_scheduler_type           "$LR_SCHEDULER"
        --anomaly_threshold           "$ANOMALY_THRESHOLD"
        --margin_tau                  "$MARGIN_TAU"
        --margin_lambda               "$MARGIN_LAMBDA"
    )
    [ -n "${LR}" ] && ARGS+=(--lr "$LR")
    [ "${NO_VALIDATION}" = "1" ] && ARGS+=(--no_validation)
    [ "${FP16}" = "1" ] && ARGS+=(--fp16) || ARGS+=(--no_fp16)
    if [ "$FINETUNE_MODE" = "lora" ]; then
        ARGS+=(--lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT")
    fi
    if [ "${ENABLE_SEP_TOKEN}" = "1" ]; then
        ARGS+=(--enable_sep_token --normal_signal_length "$NORMAL_SIGNAL_LENGTH" --input_patch_size "$INPUT_PATCH_SIZE")
    fi

    echo ">>> Step 1/2: training (writes ${OUTPUT_DIR}/window_loss_history.csv)"
    python "${WS}/finetune_anomaly_loss_logging.py" "${ARGS[@]}"
else
    echo ">>> Step 1/2: SKIPPED (SKIP_TRAIN=1)"
fi

# ── Step 2: plotting ─────────────────────────────────────────────────────────
echo ">>> Step 2/2: plotting per-window loss"
python "${WS}/plot_window_loss_history.py" --run_dir "$OUTPUT_DIR" --metric "$METRIC" --nbins "$NBINS"

echo "======================================================"
echo "  Done. See in ${OUTPUT_DIR}:"
echo "    window_loss_history.csv      (raw per-window-per-step log)"
echo "    window_loss_anomaly.png      (anomaly windows)"
echo "    window_loss_normal.png       (non-anomaly windows)"
echo "    window_loss_all.png          (all windows)"
echo "    non_optimizing_windows.csv   (windows whose loss didn't go down)"
echo "======================================================"
