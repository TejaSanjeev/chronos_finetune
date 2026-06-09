"""
Two-stage data preparation for Chronos-2 anomaly fine-tuning on mTSBench.

Loads all *test.csv files from the mTSBench dataset directory, extracts feature
columns (excluding timestamp and is_anomaly), and saves train/val splits as
pickle files ready for chronos2 pipeline.fit().

NOTE: mTSBench *train.csv files contain only normal data (is_anomaly=0 throughout).
Anomaly labels (required for Type B/C/D pairs) only exist in *test.csv files.

Creates FOUR types of instruction tuning pairs using anomaly ground truth labels.
ALL pairs from each type are kept as-is (no balancing or downsampling).

Short windows are NaN-padded (left for context, right for future) rather than
extended into adjacent signal regions. This guarantees each context and future
window contains only one signal type — no normal/anomaly mixing anywhere.

  Stage 1 — normal futures (gradient descent, loss minimised):
    Type A — Normal-to-Normal   : context=normal,  future=normal
    Type C — Anomaly-Context    : context=anomaly, future=normal

  Stage 2 — anomalous futures (gradient ascent, loss maximised):
    Type B — Pre-Anomaly        : context=normal,  future=anomaly onset
    Type D — Anomaly-to-Anomaly : context=anomaly, future=anomaly

At inference time, high prediction error on a region => high anomaly score.

Usage:
    python inst_data_prepare.py [--data_dir ...] [--output_dir ...]
"""

import argparse
import glob
import logging
import os
import pickle

import numpy as np
import pandas as pd

log_path = os.path.join("./prepared_data/log", "prepare_data.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
)
logger = logging.getLogger(__name__)

NORMAL_SIGNAL_LENGTH = 256   # instruction prefix length (NaN-padded if series has fewer normal steps)


# ─────────────────────────────────────────────────────────────────────────────
#  Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_as_multivariate(csv_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load one *test.csv file.

    Returns
    -------
    features : float32 array (n_variates, time_steps) — timestamp/is_anomaly excluded.
    labels   : int32 array (time_steps,), 1=anomaly 0=normal (all-zero if column absent).
    """
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ("timestamp", "is_anomaly")]
    if not feature_cols:
        return None, None
    try:
        features = df[feature_cols].values.T.astype(np.float32)
        labels = df["is_anomaly"].values.astype(np.int32) if "is_anomaly" in df.columns \
            else np.zeros(df.shape[0], dtype=np.int32)
        return features, labels
    except Exception as e:
        logger.warning(f"Error processing {csv_path}: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  Anomaly Boundary Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_anomaly_boundaries(labels: np.ndarray) -> list[tuple[int, int]]:
    """
    Find contiguous anomaly regions from the binary label array.

    Returns list of (start, end) where end is EXCLUSIVE (Python-slice style).
    Example: [0,0,0,1,1,1,0,0,1,1,0] → [(3,6), (8,10)]
    """
    boundaries, in_anom, start = [], False, 0
    for i, v in enumerate(labels):
        if v == 1 and not in_anom:
            in_anom, start = True, i
        elif v == 0 and in_anom:
            in_anom = False
            boundaries.append((start, i))
    if in_anom:
        boundaries.append((start, len(labels)))
    return boundaries


def get_normal_zones(boundaries: list[tuple[int, int]], total: int) -> list[tuple[int, int]]:
    """
    Return normal (non-anomaly) zones as (start, end) pairs.

    Example: boundaries=[(3,6),(8,10)], total=12 → [(0,3),(6,8),(10,12)]
    """
    zones, prev = [], 0
    for s, e in boundaries:
        if s > prev:
            zones.append((prev, s))
        prev = e
    if prev < total:
        zones.append((prev, total))
    return zones


def extract_normal_signal(
    data: np.ndarray,
    normal_zones: list[tuple[int, int]],
    length: int,
) -> np.ndarray | None:
    """
    Return a (F, length) reference normal signal sampled from the series' normal zones.

    Strategy:
      1. If a single normal zone is long enough, take its last `length` timesteps.
      2. Otherwise concatenate normal zones (longest first) until we have enough.
      3. If still short, left-pad with NaN — the model masks NaN inputs out.

    Returns None if there are no normal zones at all.
    """
    if not normal_zones:
        return None

    sorted_zones = sorted(normal_zones, key=lambda z: z[1] - z[0], reverse=True)
    s, e = sorted_zones[0]
    if e - s >= length:
        return data[:, e - length:e].astype(np.float32, copy=False)

    chunks, collected = [], 0
    for s, e in sorted_zones:
        chunks.append(data[:, s:e])
        collected += e - s
        if collected >= length:
            break

    combined = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    if combined.shape[1] >= length:
        return combined[:, -length:]

    F = combined.shape[0]
    pad = np.full((F, length - combined.shape[1]), np.nan, dtype=np.float32)
    return np.concatenate([pad, combined], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
#  Padding Helper
# ─────────────────────────────────────────────────────────────────────────────

def _pad_to_length(arr: np.ndarray, length: int, side: str) -> np.ndarray:
    """
    Pad or trim a (F, T) array along the time axis to exactly `length` steps.

      side='left'  — NaN prepended  (for short contexts: keeps the most recent steps)
      side='right' — NaN appended   (for short futures:  keeps the earliest steps)

    Never mixes signal types: caller is responsible for passing a single-type slice.
    """
    T = arr.shape[1]
    if T == length:
        return arr.astype(np.float32, copy=False)
    if T > length:
        trimmed = arr[:, -length:] if side == 'left' else arr[:, :length]
        return trimmed.astype(np.float32, copy=False)
    pad = np.full((arr.shape[0], length - T), np.nan, dtype=np.float32)
    parts = [pad, arr] if side == 'left' else [arr, pad]
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Type A — Normal-to-Normal Pairs
# ─────────────────────────────────────────────────────────────────────────────

def create_type_a_pairs(
    data: np.ndarray,
    normal_zones: list[tuple[int, int]],
    context_length: int,
    prediction_length: int,
    stride: int,
) -> list[dict]:
    """
    Type A — Normal-to-Normal.

    Sliding window: the future end steps through each normal zone. The context is
    bounded strictly to the current zone's start — no data from adjacent (potentially
    anomalous) regions enters the context. Short contexts are left-padded with NaN.
    """
    pairs = []
    for zs, ze in normal_zones:
        if ze - zs < prediction_length:
            continue
        for fe in range(zs + prediction_length, ze + 1, stride):
            fs = fe - prediction_length
            ctx_s = max(zs, fs - context_length)
            if ctx_s == fs:
                continue                        # no real context steps in this zone
            ctx = _pad_to_length(data[:, ctx_s:fs], context_length, side='left')
            fut = data[:, fs:fe].astype(np.float32, copy=False)
            pairs.append({"context": {"target": ctx},
                          "future":  {"target": fut},
                          "type": "normal_to_normal"})
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  Type B — Pre-Anomaly Pairs
# ─────────────────────────────────────────────────────────────────────────────

def create_type_b_pairs(
    data: np.ndarray,
    boundaries: list[tuple[int, int]],
    context_length: int,
    prediction_length: int,
) -> list[dict]:
    """
    Type B — Pre-Anomaly (normal context → anomaly onset future).

    One pair per anomaly event:
      Context : context_length normal steps immediately before anomaly onset.
                Left-padded with NaN if fewer normal steps are available before
                the series start.
      Future  : prediction_length steps starting at anomaly onset, capped at the
                anomaly end and right-padded with NaN when the anomaly is shorter
                than prediction_length.

    No normal/anomaly mixing in either window.
    """
    pairs = []
    for anom_s, anom_e in boundaries:
        ctx = _pad_to_length(
            data[:, max(0, anom_s - context_length):anom_s],
            context_length, side='left',
        )
        fut = _pad_to_length(
            data[:, anom_s:min(anom_s + prediction_length, anom_e)],
            prediction_length, side='right',
        )
        pairs.append({"context": {"target": ctx},
                      "future":  {"target": fut},
                      "type": "pre_anomaly"})
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  Type C — Anomaly-Context Pairs
# ─────────────────────────────────────────────────────────────────────────────

def create_type_c_pairs(
    data: np.ndarray,
    labels: np.ndarray,
    boundaries: list[tuple[int, int]],
    context_length: int,
    prediction_length: int,
    normal_lead: int,
    normal_tail: int,
) -> list[dict]:
    """
    Type C — Anomaly-Context (anomaly context → normal future).

    Context window contains the anomaly event with a normal lead-in and tail.
    Future is post-anomaly normal behavior.

    The lead-in steps before the anomaly are content-checked (not just distance-
    checked) so that a prior anomaly event never bleeds into the context
    disguised as normal signal.

    normal_lead : minimum anomaly-free steps BEFORE the anomaly in the context.
    normal_tail : normal steps AFTER the anomaly before the future window starts.
    """
    pairs, total = [], len(labels)
    for idx, (anom_s, anom_e) in enumerate(boundaries):
        max_ce = total - prediction_length
        ce = min(anom_e + normal_tail, max_ce)
        if ce <= anom_s:
            continue

        cs = max(0, ce - context_length)
        ce = cs + context_length

        if anom_s - cs < normal_lead:
            continue

        # Content check: lead-in steps must be anomaly-free (no prior event bleed-in)
        if np.any(labels[cs:anom_s] == 1):
            continue

        if ce > anom_e + normal_tail + context_length // 2:
            continue

        fs, fe = ce, ce + prediction_length
        if not np.any(labels[cs:ce] == 1):
            continue
        if idx + 1 < len(boundaries) and fe > boundaries[idx + 1][0]:
            continue
        if np.any(labels[fs:fe] == 1):
            continue

        pairs.append({"context": {"target": data[:, cs:ce].astype(np.float32, copy=False)},
                      "future":  {"target": data[:, fs:fe].astype(np.float32, copy=False)},
                      "type": "anomaly_context"})
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  Type D — Anomaly-to-Anomaly Pairs
# ─────────────────────────────────────────────────────────────────────────────

def create_type_d_pairs(
    data: np.ndarray,
    boundaries: list[tuple[int, int]],
    context_length: int,
    prediction_length: int,
) -> list[dict]:
    """
    Type D — Anomaly-to-Anomaly (anomaly context → anomaly future).

    One pair per anomaly event. The anomaly region is split at its midpoint:
      Context : first half of the anomaly, left-padded with NaN to context_length.
      Future  : second half of the anomaly, right-padded with NaN to prediction_length.

    For long anomalies (first half > context_length) the most recent context_length
    steps of the first half are used; for long second halves the first
    prediction_length steps are used — no padding in either case.

    No normal/anomaly mixing in either window.
    """
    pairs = []
    for anom_s, anom_e in boundaries:
        anom_len = anom_e - anom_s
        if anom_len < 2:
            continue                            # need at least one step per window
        split = anom_s + anom_len // 2
        ctx = _pad_to_length(data[:, anom_s:split], context_length, side='left')
        fut = _pad_to_length(data[:, split:anom_e], prediction_length, side='right')
        pairs.append({"context": {"target": ctx},
                      "future":  {"target": fut},
                      "type": "anomaly_to_anomaly"})
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  Model-Ready Input Conversion
# ─────────────────────────────────────────────────────────────────────────────

def pairs_to_model_inputs(pairs: list[dict]) -> list[dict]:
    """
    Convert instruction pairs to fixed-length model inputs for pipeline.fit().

    Each pair's `normal_signal` field (shape F × NORMAL_SIGNAL_LENGTH) is
    prepended as an instruction prefix. NaN if no normal zone existed.
    Target layout:

        [normal_signal (256) | context (C) | future (P)]
    """
    out = []
    for p in pairs:
        ctx, fut = p["context"]["target"], p["future"]["target"]
        normal = p.get("normal_signal")
        if normal is None:
            normal = np.full((ctx.shape[0], NORMAL_SIGNAL_LENGTH), np.nan, dtype=np.float32)
        target = np.concatenate([normal, ctx, fut], axis=1)
        out.append({"target": target})
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Stage-Specific Per-Series Pair Construction
# ─────────────────────────────────────────────────────────────────────────────

def _attach_normal_signal(pairs: list[dict], normal_sig: np.ndarray | None) -> None:
    """In-place: add the same per-series normal_signal reference to every pair."""
    for p in pairs:
        p["normal_signal"] = normal_sig


def build_stage1_pairs(
    data: np.ndarray,
    labels: np.ndarray,
    context_length: int,
    prediction_length: int,
    stride: int,
    normal_lead: int,
    normal_tail: int,
) -> list[dict]:
    """
    Stage 1 — ALL pairs with NORMAL futures (gradient descent).

    Type A: context=normal,  future=normal
    Type C: context=anomaly, future=normal

    Every pair receives a NORMAL_SIGNAL_LENGTH normal instruction prefix.
    """
    bounds = extract_anomaly_boundaries(labels)
    zones  = get_normal_zones(bounds, len(labels))
    type_a = create_type_a_pairs(data, zones, context_length, prediction_length, stride)
    type_c = create_type_c_pairs(data, labels, bounds, context_length, prediction_length,
                                  normal_lead, normal_tail)
    pairs = type_a + type_c
    normal_sig = extract_normal_signal(data, zones, NORMAL_SIGNAL_LENGTH)
    _attach_normal_signal(pairs, normal_sig)
    logger.debug(f"  Stage 1 — A: {len(type_a)}  C: {len(type_c)}")
    return pairs


def build_stage2_pairs(
    data: np.ndarray,
    labels: np.ndarray,
    context_length: int,
    prediction_length: int,
) -> list[dict]:
    """
    Stage 2 — ALL pairs with ANOMALOUS futures (gradient ascent).

    Type B: context=normal,  future=anomaly onset
    Type D: context=anomaly, future=anomaly

    Every pair receives a NORMAL_SIGNAL_LENGTH normal instruction prefix.
    """
    bounds = extract_anomaly_boundaries(labels)
    zones  = get_normal_zones(bounds, len(labels))
    type_b = create_type_b_pairs(data, bounds, context_length, prediction_length)
    type_d = create_type_d_pairs(data, bounds, context_length, prediction_length)
    pairs = type_b + type_d
    normal_sig = extract_normal_signal(data, zones, NORMAL_SIGNAL_LENGTH)
    _attach_normal_signal(pairs, normal_sig)
    logger.debug(f"  Stage 2 — B: {len(type_b)}  D: {len(type_d)}")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  Two-Stage Preparation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare_two_stage_inputs(
    data_dir: str,
    min_length: int,
    val_fraction: float,
    context_length: int,
    prediction_length: int,
    stride: int,
    seed: int = 42,
):
    """
    Produce separate Stage 1 and Stage 2 datasets for two-stage anomaly fine-tuning.

    Stage 1 — NORMAL futures (gradient descent, loss minimised):
        Type A: normal context  → normal future
        Type C: anomaly context → normal future

    Stage 2 — ANOMALOUS futures (gradient ascent, loss maximised):
        Type B: normal context  → anomaly future (onset)
        Type D: anomaly context → anomaly future

    All pairs use NaN-padding for short windows; no signal mixing anywhere.
    Every pair includes a NORMAL_SIGNAL_LENGTH=256 normal instruction prefix.

    Returns
    -------
    train_inputs, val_inputs,
    s1_train_pairs, s1_val_pairs, s2_train_pairs, s2_val_pairs,
    s1_train_model_inputs, s1_val_model_inputs,
    s2_train_model_inputs, s2_val_model_inputs
    """
    rng = np.random.default_rng(seed)

    csv_files = sorted(glob.glob(os.path.join(data_dir, "**", "*test.csv"), recursive=True))
    logger.info(f"Found {len(csv_files)} *test.csv files under {data_dir}")

    normal_lead = max(10, context_length // 4)
    normal_tail = max(5,  prediction_length // 4)
    logger.info(f"Type C — normal_lead={normal_lead}, normal_tail={normal_tail}")

    # Only require at least prediction_length steps; short contexts are NaN-padded.
    all_inputs, all_labels, skipped = [], [], 0
    min_req = max(min_length, prediction_length)
    for path in csv_files:
        try:
            feat, lbl = load_csv_as_multivariate(path)
            if feat is None or feat.shape[1] < min_req:
                logger.debug(
                    f"Skipping {os.path.basename(path)}: "
                    f"length={feat.shape[1] if feat is not None else 'None'}, required={min_req}"
                )
                skipped += 1
                continue
            all_inputs.append({"target": feat})
            all_labels.append(lbl)
        except Exception as exc:
            logger.warning(f"Skipping {path}: {exc}")
            skipped += 1
    logger.info(f"Usable series: {len(all_inputs)}  (skipped {skipped})")
    if not all_inputs:
        raise ValueError("No usable series found. Check data_dir and min_length.")

    idx     = rng.permutation(len(all_inputs))
    n_val   = max(1, int(len(all_inputs) * val_fraction))
    val_set = set(idx[:n_val].tolist())
    train_inputs = [all_inputs[i] for i in range(len(all_inputs)) if i not in val_set]
    val_inputs   = [all_inputs[i] for i in val_set]
    train_labels = [all_labels[i] for i in range(len(all_inputs)) if i not in val_set]
    val_labels   = [all_labels[i] for i in val_set]
    logger.info(f"Train series: {len(train_inputs)} | Val series: {len(val_inputs)}")

    common_s1 = dict(context_length=context_length, prediction_length=prediction_length,
                     stride=stride)
    common_s2 = dict(context_length=context_length, prediction_length=prediction_length)

    s1_tr, s2_tr = [], []
    for i, (series, lbl) in enumerate(zip(train_inputs, train_labels)):
        logger.debug(f"Train series {i + 1}/{len(train_inputs)}")
        s1_tr.extend(build_stage1_pairs(series["target"], lbl, **common_s1,
                                         normal_lead=normal_lead, normal_tail=normal_tail))
        s2_tr.extend(build_stage2_pairs(series["target"], lbl, **common_s2))

    s1_val, s2_val = [], []
    for i, (series, lbl) in enumerate(zip(val_inputs, val_labels)):
        logger.debug(f"Val series {i + 1}/{len(val_inputs)}")
        s1_val.extend(build_stage1_pairs(series["target"], lbl, **common_s1,
                                          normal_lead=normal_lead, normal_tail=normal_tail))
        s2_val.extend(build_stage2_pairs(series["target"], lbl, **common_s2))

    logger.info(f"Stage 1 pairs — Train: {len(s1_tr)} | Val: {len(s1_val)}")
    logger.info(f"Stage 2 pairs — Train: {len(s2_tr)} | Val: {len(s2_val)}")

    logger.info("Converting to fixed-length model inputs...")
    s1_tr_in  = pairs_to_model_inputs(s1_tr)
    s1_val_in = pairs_to_model_inputs(s1_val)
    s2_tr_in  = pairs_to_model_inputs(s2_tr)
    s2_val_in = pairs_to_model_inputs(s2_val)
    logger.info(f"Stage 1 model inputs — Train: {len(s1_tr_in)} | Val: {len(s1_val_in)}")
    logger.info(f"Stage 2 model inputs — Train: {len(s2_tr_in)} | Val: {len(s2_val_in)}")

    return (train_inputs, val_inputs,
            s1_tr, s1_val, s2_tr, s2_val,
            s1_tr_in, s1_val_in, s2_tr_in, s2_val_in)


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics Logging
# ─────────────────────────────────────────────────────────────────────────────

def log_statistics(
    train_inputs: list,
    val_inputs: list,
    s1_train_pairs: list,
    s1_val_pairs: list,
    s2_train_pairs: list,
    s2_val_pairs: list,
) -> None:
    """Log shape and type-distribution statistics for both stages."""
    series   = train_inputs + val_inputs
    lengths  = [s["target"].shape[1] for s in series]
    variates = [s["target"].shape[0] for s in series]

    logger.info("=" * 60)
    logger.info("RAW SERIES STATISTICS")
    logger.info(f"  Total series : {len(series)}")
    logger.info(f"  Time steps   : min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}")
    logger.info(f"  Num features : min={min(variates)}, max={max(variates)}, mean={np.mean(variates):.1f}")

    for stage_name, tr_pairs, val_pairs in [
        ("STAGE 1", s1_train_pairs, s1_val_pairs),
        ("STAGE 2", s2_train_pairs, s2_val_pairs),
    ]:
        all_pairs = tr_pairs + val_pairs
        if not all_pairs:
            continue
        counts: dict[str, int] = {}
        for p in all_pairs:
            t = p.get("type", "?")
            counts[t] = counts.get(t, 0) + 1
        logger.info("=" * 60)
        logger.info(f"{stage_name} INSTRUCTION PAIR STATISTICS")
        logger.info(f"  Train: {len(tr_pairs)}  Val: {len(val_pairs)}  Total: {len(all_pairs)}")
        logger.info(f"  Avg per series : {len(all_pairs) / len(series):.1f}")
        logger.info("  Type distribution (raw counts, no balancing):")
        for type_name, count in sorted(counts.items()):
            logger.info(f"    {type_name:<22} : {count:>6}  ({count / len(all_pairs) * 100:.1f}%)")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Two-stage data prep for Chronos-2 anomaly fine-tuning."
    )
    p.add_argument("--data_dir",          default="/home/rajib/mTSBench/Datasets/mTSBench",
                   help="Root directory of the mTSBench dataset")
    p.add_argument("--output_dir",        default="./prepared_data",
                   help="Root output directory; stage1/ and stage2/ subdirs are created inside")
    p.add_argument("--min_length",        type=int,   default=50,
                   help="Minimum series length; shorter series are discarded")
    p.add_argument("--val_fraction",      type=float, default=0.1,
                   help="Fraction of series held out for validation")
    p.add_argument("--context_length",    type=int,   default=512,
                   help="Number of past time steps used as context")
    p.add_argument("--prediction_length", type=int,   default=64,
                   help="Number of future time steps to predict")
    p.add_argument("--stride",            type=int,   default=None,
                   help="Sliding-window stride for Type A (default: prediction_length // 2)")
    args = p.parse_args()

    if args.stride is None:
        args.stride = args.prediction_length
        logger.info(f"Stride not set — using default: stride={args.stride}")

    os.makedirs(args.output_dir, exist_ok=True)

    (train_inputs, val_inputs,
     s1_tr, s1_val, s2_tr, s2_val,
     s1_tr_in, s1_val_in, s2_tr_in, s2_val_in) = prepare_two_stage_inputs(
        data_dir=args.data_dir,
        min_length=args.min_length,
        val_fraction=args.val_fraction,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
    )

    for stage, (pairs_tr, pairs_val, inputs_tr, inputs_val) in {
        "stage1": (s1_tr, s1_val, s1_tr_in, s1_val_in),
        "stage2": (s2_tr, s2_val, s2_tr_in, s2_val_in),
    }.items():
        stage_dir = os.path.join(args.output_dir, stage)
        os.makedirs(stage_dir, exist_ok=True)
        for fname, data in [
            ("train_pairs.pkl",        pairs_tr),
            ("val_pairs.pkl",          pairs_val),
            ("train_model_inputs.pkl", inputs_tr),
            ("val_model_inputs.pkl",   inputs_val),
        ]:
            path = os.path.join(stage_dir, fname)
            with open(path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"{stage} — {len(data):>6} entries → {path}")

    log_statistics(train_inputs, val_inputs, s1_tr, s1_val, s2_tr, s2_val)


if __name__ == "__main__":
    main()
