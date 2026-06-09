---
name: Two-stage anomaly fine-tuning workflow
description: Architecture, files, pair types, and key decisions for Chronos-2 anomaly detection via two-stage fine-tuning
type: project
originSessionId: 9e4c9af0-ff22-4fa5-b3ef-065feb2cb583
---
Two-stage anomaly-aware fine-tuning pipeline built on Chronos-2 for time-series anomaly detection.

**Why:** High prediction error at inference = high anomaly score. Stage 1 trains the model to always predict normal futures; Stage 2 degrades predictions on anomaly futures via gradient ascent.

**How to apply:** When working on data prep, fine-tuning, or inference for anomaly detection, this is the full context.

## Four pair types

| Type | Context | Future | Stage | Training direction |
|------|---------|--------|-------|--------------------|
| A    | normal  | normal | 1     | gradient descent   |
| B    | normal  | anomaly onset | 2 | gradient ascent |
| C    | anomaly | normal | 1     | gradient descent   |
| D    | anomaly | anomaly | 2   | gradient ascent    |

## Key files

- `inst_data_prepare.py` — builds Stage 1 and Stage 2 pkl datasets from mTSBench CSVs
- `finetune_anomaly.py` — runs both stages; Stage 2 uses `Chronos2AnomalyTrainer`
- `run_finetune.sh` — shell wrapper; `SKIP_STAGE1=1` hardcoded, loads `chronos2-stage1/finetuned-ckpt`
- `src/chronos/chronos2/anomaly_trainer.py` — custom trainer that negates loss for gradient ascent

## Output directories

- `chronos2-stage1/finetuned-ckpt` — LoRA adapter from Stage 1 (already trained)
- `chronos2-stage2/finetuned-ckpt` — final model for inference (Stage 2 output)
- `prepared_data/stage1/` and `prepared_data/stage2/` — pkl datasets consumed by `finetune_anomaly.py`

## Data source

mTSBench dataset at `/home/rajib/mTSBench/Datasets/mTSBench`. Only `*test.csv` files have `is_anomaly` labels; all pair generation uses these files.

## Important fixes applied

- `create_type_b_pairs`: clamp `effective_offset = min(offset, anom_len)` + dedup `seen` set
- `create_type_c_pairs`: enforce `normal_lead` guard + drift guard when anomaly early in series
- `create_type_d_pairs`: future must have `>= prediction_length // 4` anomaly steps (not just 1)
- `finetune_anomaly.py`: load Stage 1 LoRA ckpt with `Chronos2Pipeline.from_pretrained`, not `BaseChronosPipeline`
