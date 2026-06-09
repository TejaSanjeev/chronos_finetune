import os
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from scipy.ndimage import uniform_filter1d
import sys
import torch
sys.path.insert(0, "/home/rajib/mTSBench")

from chronos import Chronos2Pipeline
from VUS_ROC_VUS_PR.metrics import get_metrics

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# -------------------- ARGUMENT PARSING --------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuned Chronos-2 SMD Anomaly Detection (VUS metrics)"
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.2,
        help="Train/Test split ratio (e.g., 0.2 means 20%% train, 80%% test)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=64,
        help="Chronos will predict this many timestamps (match fine-tuning prediction_length=64)"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=512,
        help="Number of recent past timestamps used as context (match fine-tuning context_length=512). "
             "The normal-signal prefix is added on top of this."
    )
    parser.add_argument(
        "--normal_signal_length",
        type=int,
        default=256,
        help="Length of the normal-signal prefix prepended before the context "
             "(must match fine-tuning normal_signal_length=256 so the SEP token lands correctly)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES"
    )
    parser.add_argument(
        "--score_method",
        type=str,
        default="interval",
        choices=["mse", "interval", "normalized_deviation", "smape"],
        help=(
            "Anomaly scoring method per feature:\n"
            "  mse                  - squared error vs median\n"
            "  interval             - violation beyond [0.1, 0.9] quantile band\n"
            "  normalized_deviation - |actual - median| / band_width\n"
            "  smape                - symmetric MAPE vs median"
        )
    )
    parser.add_argument(
        "--agg_method",
        type=str,
        default="topk_mean",
        choices=["l2", "max", "mean", "topk_mean"],
        help=(
            "How to aggregate per-feature scores into a single time-series score:\n"
            "  l2        - L2 norm\n"
            "  max       - maximum across features\n"
            "  mean      - mean across features\n"
            "  topk_mean - mean of top-k features"
        )
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=5,
        help="Uniform smoothing window for final anomaly score (1 = no smoothing)"
    )
    parser.add_argument(
        "--sliding_window_VUS",
        type=int,
        default=100,
        help="Sliding-window size used by VUS metrics"
    )
    parser.add_argument(
        "--vus_version",
        type=str,
        default="opt",
        choices=["opt", "opt_mem"],
        help="VUS computation backend"
    )
    parser.add_argument(
        "--vus_thre",
        type=int,
        default=250,
        help="Number of thresholds used in VUS curve generation"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results_finetuned.csv",
        help="Path to save per-file results CSV"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process the first N matched files (quick testing)"
    )
    return parser.parse_args()


# -------------------- GPU SETUP --------------------
args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# -------------------- LOAD FINE-TUNED CHRONOS-2 --------------------
# checkpoint-8000 stores LoRA adapters; from_pretrained detects adapter_config.json,
# loads via AutoPeftModel, and merges weights before returning the pipeline.
CKPT_PATH = "/home/rajib/chronos-forecasting/rajib_work_space/chronos2-single-stage_NS1000_V2/finetuned-ckpt"

print(f"Loading fine-tuned Chronos-2 from: {CKPT_PATH}")
pipeline: Chronos2Pipeline = Chronos2Pipeline.from_pretrained(
    CKPT_PATH,
    device_map="cuda",
)

# --- SEP position fix ------------------------------------------------------
# The adapter-load path in from_pretrained re-enables use_sep_token but never
# restores sep_patch_index (it's not persisted in adapter_config.json), so it
# silently falls back to the default 0. Training used patch index
# normal_signal_length // input_patch_size = 256 // 16 = 16, i.e. the SEP token
# sits at the normal-signal↔context boundary. Restore it here so inference uses
# the same input layout the weights were trained on.
if getattr(pipeline.model.chronos_config, "use_sep_token", False):
    INPUT_PATCH_SIZE = 16
    correct_sep_idx = args.normal_signal_length // INPUT_PATCH_SIZE
    old_sep_idx = pipeline.model.chronos_config.sep_patch_index
    pipeline.model.chronos_config.sep_patch_index = correct_sep_idx
    print(f"[SEP fix] sep_patch_index: {old_sep_idx} -> {correct_sep_idx}")

pipeline.model.eval()
print("Fine-tuned pipeline loaded.\n")


# -------------------- DATA PREPARATION --------------------
def split_dataset(df, split_ratio):
    split_idx = int(len(df) * split_ratio)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test  = df.iloc[split_idx:].reset_index(drop=True)
    return df_train, df_test


# Quantile indices: predict() returns the model's fixed quantile set, so map the
# nearest available level to the 0.1 / 0.5 / 0.9 we score against.
MODEL_QUANTILES = list(pipeline.quantiles)
def _q_idx(level):
    return int(np.argmin([abs(q - level) for q in MODEL_QUANTILES]))
Q10_IDX, Q50_IDX, Q90_IDX = _q_idx(0.1), _q_idx(0.5), _q_idx(0.9)


def build_normal_signal(df_train, feature_list, length):
    """Fixed normal-signal prefix from the (clean) train split: last `length` steps,
    NaN left-padded if train is shorter. Shape (n_features, length)."""
    arr = df_train[feature_list].values.T.astype(np.float32)   # (F, T_train)
    F = arr.shape[0]
    if arr.shape[1] >= length:
        return arr[:, -length:]
    pad = np.full((F, length - arr.shape[1]), np.nan, dtype=np.float32)
    return np.concatenate([pad, arr], axis=1)


# -------------------- PREDICTION --------------------
def generate_prediction(df_train, df_test, feature_list, prediction_length,
                        context_length, normal_signal_length):
    """
    Sliding-window forecast that reproduces the training input layout:

        [ normal_signal (256, fixed) | context (512, rolling) ]  →  predict `prediction_length`

    The normal-signal prefix is built once from the clean train split and reused for
    every window. The context is the `context_length` steps immediately before each
    window (drawn from train + already-seen test). Windows tile the test set with
    stride = prediction_length, so each test step is predicted exactly once.

    Returns three (T_test, n_features) arrays of the 0.1 / 0.5 / 0.9 quantiles,
    aligned row-for-row with df_test.
    """
    normal_sig = build_normal_signal(df_train, feature_list, normal_signal_length)  # (F, 256)
    full_arr   = pd.concat([df_train, df_test], ignore_index=True)[feature_list].values.T.astype(np.float32)
    train_len, test_len, F = len(df_train), len(df_test), len(feature_list)

    q10 = np.full((test_len, F), np.nan, dtype=np.float32)
    q50 = np.full((test_len, F), np.nan, dtype=np.float32)
    q90 = np.full((test_len, F), np.nan, dtype=np.float32)

    n_windows = (test_len + prediction_length - 1) // prediction_length
    for w in range(n_windows):
        test_start = w * prediction_length
        abs_start  = train_len + test_start                 # window start in full series
        horizon    = min(prediction_length, test_len - test_start)

        # rolling context: the `context_length` steps right before the window
        ctx = full_arr[:, max(0, abs_start - context_length):abs_start]
        if ctx.shape[1] < context_length:                   # left-pad short contexts
            pad = np.full((F, context_length - ctx.shape[1]), np.nan, dtype=np.float32)
            ctx = np.concatenate([pad, ctx], axis=1)

        model_input = np.concatenate([normal_sig, ctx], axis=1)   # (F, 256+512=768)

        preds = pipeline.predict(
            [model_input],
            prediction_length=horizon,
            context_length=normal_signal_length + context_length,
        )[0]                                                 # (F, n_quantiles, horizon)
        preds = preds.float().cpu().numpy()

        q10[test_start:test_start + horizon] = preds[:, Q10_IDX, :].T
        q50[test_start:test_start + horizon] = preds[:, Q50_IDX, :].T
        q90[test_start:test_start + horizon] = preds[:, Q90_IDX, :].T

    return q10, q50, q90


# -------------------- ANOMALY SCORING --------------------
def compute_feature_score(y_actual, y_lower, y_median, y_upper, method="mse"):
    if method == "mse":
        return (y_actual - y_median) ** 2

    elif method == "smape":
        eps = 1e-8
        return np.abs(y_actual - y_median) / (
            np.abs(y_actual) + np.abs(y_median) + eps
        )

    elif method == "interval":
        upper_violation = np.maximum(0.0, y_actual - y_upper)
        lower_violation = np.maximum(0.0, y_lower  - y_actual)
        return upper_violation + lower_violation

    else:  # normalized_deviation
        band_width = y_upper - y_lower + 1e-8
        deviation  = np.abs(y_actual - y_median)
        return deviation / band_width


def aggregate_scores(anomaly_df, method="l2"):
    if method == "l2":
        return np.sqrt((anomaly_df ** 2).sum(axis=1)).values

    elif method == "max":
        return anomaly_df.max(axis=1).values

    elif method == "mean":
        return anomaly_df.mean(axis=1).values

    else:  # topk_mean
        k = 4
        return anomaly_df.apply(
            lambda row: row.nlargest(k).mean(), axis=1
        ).values


def robust_normalize(series):
    p1  = np.percentile(series, 1)
    p99 = np.percentile(series, 99)
    clipped = np.clip(series, p1, p99)
    denom = p99 - p1
    if denom < 1e-8:
        return np.zeros_like(series, dtype=float)
    return (clipped - p1) / denom


# -------------------- PATHS --------------------
data_path = "/home/rajib/TSB-AD/Datasets/TSB-AD-M/*GHL*.csv"


# -------------------- MAIN LOOP --------------------
file_list         = sorted(glob.glob(data_path))
if args.limit > 0:
    file_list = file_list[:args.limit]
    print(f"[limit] processing only the first {len(file_list)} file(s)")
dic_for_each_file = defaultdict(list)
prediction_length = args.horizon
context_length    = args.context_length

write_header = not os.path.exists(args.output_csv)

for f in tqdm(file_list, desc="Processing TSB-AD files", unit="file"):
    file_name = os.path.basename(f).replace(".csv", "")
    print(f"\nProcessing: {file_name}")

    try:
        df_original = pd.read_csv(f)

        feature_list = [
            c for c in df_original.columns
            if c not in ["timestamp", "Label"]
        ]

        df_train, df_test = split_dataset(df_original, args.split_ratio)

        q10, q50, q90 = generate_prediction(
            df_train, df_test, feature_list,
            prediction_length, context_length, args.normal_signal_length
        )

        print("Prediction Done!")
        anomaly_scores = {}
        for j, feature_name in enumerate(feature_list):
            y_actual = df_test[feature_name].values.astype(float)
            anomaly_scores[feature_name] = compute_feature_score(
                y_actual, q10[:, j], q50[:, j], q90[:, j], method=args.score_method
            )

        anomaly_df = pd.DataFrame(anomaly_scores)

        if args.score_method != "smape":
            anomaly_df = anomaly_df.apply(
                lambda col: pd.Series(robust_normalize(col.values)), axis=0
            )
        anomaly_df = anomaly_df.fillna(0)

        y_score = aggregate_scores(anomaly_df, method=args.agg_method)

        if args.smooth_window > 1:
            y_score = uniform_filter1d(y_score, size=args.smooth_window)

        y_true = df_test["Label"].values.astype(int)

        if y_true.sum() == 0:
            print(f"Skipping {file_name}: no anomalies in ground truth")
            continue

        evaluation_result = get_metrics(
            y_score, y_true,
            slidingWindow=args.sliding_window_VUS,
            version=args.vus_version,
            thre=args.vus_thre,
        )

        vus_roc = evaluation_result["VUS-ROC"]
        vus_pr  = evaluation_result["VUS-PR"]
        auroc   = evaluation_result["AUC-ROC"]
        auprc   = evaluation_result["AUC-PR"]
        print(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | "
              f"VUS-ROC: {vus_roc:.4f} | VUS-PR: {vus_pr:.4f}")

        dic_for_each_file["file_name"].append(file_name)
        dic_for_each_file["AUROC"].append(auroc)
        dic_for_each_file["AUPRC"].append(auprc)
        dic_for_each_file["VUS-ROC"].append(vus_roc)
        dic_for_each_file["VUS-PR"].append(vus_pr)

        # write this file's row immediately so results survive a crash
        row_df = pd.DataFrame({
            "file_name": [file_name],
            "AUROC": [auroc], "AUPRC": [auprc],
            "VUS-ROC": [vus_roc], "VUS-PR": [vus_pr],
        })
        row_df.to_csv(args.output_csv, mode="a", header=write_header, index=False)
        write_header = False

    except Exception as e:
        print(f"ERROR on {file_name}: {e} — skipping")
    finally:
        torch.cuda.empty_cache()


# -------------------- SUMMARY --------------------
auroc_list   = dic_for_each_file["AUROC"]
auprc_list   = dic_for_each_file["AUPRC"]
vus_roc_list = dic_for_each_file["VUS-ROC"]
vus_pr_list  = dic_for_each_file["VUS-PR"]

mean_auroc   = float(np.mean(auroc_list))   if auroc_list   else float("nan")
mean_auprc   = float(np.mean(auprc_list))   if auprc_list   else float("nan")
mean_vus_roc = float(np.mean(vus_roc_list)) if vus_roc_list else float("nan")
mean_vus_pr  = float(np.mean(vus_pr_list))  if vus_pr_list  else float("nan")

print("\nFinished processing all SMD files")
print(f"Mean AUROC:   {mean_auroc:.4f}")
print(f"Mean AUPRC:   {mean_auprc:.4f}")
print(f"Mean VUS-ROC: {mean_vus_roc:.4f}")
print(f"Mean VUS-PR:  {mean_vus_pr:.4f}")

print(f"\nPer-file results saved (incrementally) to: {args.output_csv}")
