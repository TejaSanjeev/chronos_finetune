# Simple Sliding-Window Data Preparation

Data prep for Chronos-2 anomaly fine-tuning on mTSBench. Builds `[CONTEXT][FUTURE]`
pairs where the future is either **fully normal** or **fully anomalous** (never mixed),
and splits them into two separate datasets.

## Files

| File | Purpose |
|------|---------|
| `inst_data_prepare_simple.py` | Build pairs from `*test.csv`, split by future type, save pkl files |
| `inst_data_preparation.sh`    | Wrapper script with overridable env vars |
| `visualize_pairs.py`          | Render sample pairs to a PNG |
| `visualize_pairs.ipynb`       | Same visualization, inline in a notebook |

## Pair design

For each series, slide a window over every timestamp:

```
context = data[t - context_length : t]      # always a full context_length real steps
future  = data[t : t + prediction_length]   # truncated at first label change, NaN-padded
```

Rules:

- **Start at `t = context_length`** (default 512), so the context is always full — no
  context padding.
- **Future is kept pure.** It is truncated at the first normal/anomaly transition and
  right-padded with NaN to `prediction_length`. Example: if only the first 10 steps are
  anomalous, keep those 10 and pad the remaining 54.
- **Future type** is decided by the label at `t`: `0 → normal`, `1 → anomaly`.
- **Tail windows** with no future steps left are excluded.
- A `normal_signal` reference of **256 steps** is prepended to each pair as an
  instruction prefix.

Final model-input layout:

```
[ normal_signal (256) | context (512) | future (64) ]   = 832 steps
```

### Context purity

Only the **future** is forced to be pure. The **context** is the raw `context_length`
steps before `t` with no content check, so a context may be all-normal, all-anomalous,
or **mixed** (contain both normal and anomalous steps). The two-case split (`normal/`
vs `anomaly/`) is based **only on the future type** (the label at `t`) — it says nothing
about what the context contains.

### Normal-signal prefix

The 256-step `normal_signal` is extracted **per series, from that same series' own
normal zones** (longest zone first; left-padded with NaN if <256 normal steps exist;
all-NaN if the series has no normal zone). The same prefix is attached to every pair
built from that series.

## Two-case output

Pairs are split by future type into two datasets:

```
prepared_data_simple/
├── normal/                       # future type = normal
│   ├── train_pairs.pkl
│   ├── val_pairs.pkl
│   ├── train_model_inputs.pkl
│   └── val_model_inputs.pkl
└── anomaly/                      # future type = anomaly
    ├── train_pairs.pkl
    ├── val_pairs.pkl
    ├── train_model_inputs.pkl
    └── val_model_inputs.pkl
```

Train/val split is at the **series** level (before the type split), so no series leaks
between train and val in either case.

## Stride behavior

`stride` is relative to `prediction_length` (64):

| stride | future windows | coverage |
|--------|----------------|----------|
| `< 64` (e.g. 8) | overlapping | every point covered multiple times |
| `= 64` | contiguous tiling | every point covered exactly once |
| `> 64` | gaps between windows | some points never in any future |

`stride=1` (default) gives maximum coverage but ~64× more pairs than `stride=64`.
Smaller stride also captures more anomaly onsets (more anomaly-future pairs).

## Usage

```bash
# Defaults: stride=1, context=512, prediction=64, output=./prepared_data_simple
bash inst_data_preparation.sh

# Override any variable
STRIDE=64 OUTPUT_DIR=./my_data bash inst_data_preparation.sh
```

Or directly:

```bash
python inst_data_prepare_simple.py \
    --data_dir /home/rajib/mTSBench/Datasets/mTSBench \
    --output_dir ./prepared_data_simple \
    --context_length 512 \
    --prediction_length 64 \
    --stride 1
```

### CLI arguments

| Arg | Default | Meaning |
|-----|---------|---------|
| `--data_dir` | `/home/rajib/mTSBench/Datasets/mTSBench` | Dataset root |
| `--output_dir` | `./prepared_data_simple` | Output root (`normal/`, `anomaly/` created inside) |
| `--min_length` | `50` | Minimum series length |
| `--val_fraction` | `0.1` | Fraction of series held out for validation |
| `--context_length` | `512` | Past steps used as context |
| `--prediction_length` | `64` | Future steps to predict |
| `--stride` | `1` | Sliding-window stride |

## Visualization

```bash
# Script → PNG
python visualize_pairs.py --pairs ./prepared_data_simple/anomaly/train_pairs.pkl

# Notebook → inline (set DATA_ROOT / SPLIT in the config cell)
jupyter notebook visualize_pairs.ipynb
```

Layout per plot: `normal prefix (gray) | context (blue) | future (green=normal, red=anomaly)`.
NaN-padded steps appear as gaps.

## Dependencies

`numpy`, `pandas`, `matplotlib`, `tqdm` (progress bars during prep).
