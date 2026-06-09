"""
Visualize the per-window TRAINING loss recorded by finetune_anomaly_loss_logging.py.

Reads <run_dir>/window_loss_history.csv and produces THREE figures, so you can
identify windows whose loss is not optimizing:

    window_loss_anomaly.png      — anomaly windows only      (future_type == 1)
    window_loss_normal.png       — non-anomaly windows only  (future_type == 0)
    window_loss_all.png          — every window

Each figure has two panels:
    (left)  per-window loss trajectories over training steps (thin grey lines,
            step-binned) with the mean and median overlaid in bold. A flat/rising
            cloud = windows that are NOT optimizing.
    (right) a heatmap of windows × training-progress with windows sorted by how
            much their loss improved (top = improved most, bottom = not optimizing),
            so the stuck windows are the bright band at the bottom.

It also writes <run_dir>/non_optimizing_windows.csv listing, per group, the
windows whose loss did not go down (early-vs-late mean), for further analysis.

Usage
-----
    python plot_window_loss_history.py --run_dir ./chronos2-run-logged
    python plot_window_loss_history.py --run_dir ... --metric train_contrib
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot per-window training loss over steps")
    p.add_argument("--run_dir", default="./chronos2-single-stage_NS1000_V4",
                   help="Dir containing window_loss_history.csv")
    p.add_argument("--metric", default="raw_loss", choices=["raw_loss", "train_contrib"],
                   help="raw_loss = forecasting loss (default); train_contrib = hinge-objective contribution")
    p.add_argument("--nbins", type=int, default=40, help="Number of step bins for trajectories/heatmap")
    p.add_argument("--max_lines", type=int, default=300, help="Max per-window trajectories to draw (subsampled)")
    p.add_argument("--max_heatmap_rows", type=int, default=600, help="Max windows shown in the heatmap")
    return p.parse_args()


def bin_matrix(df, metric, nbins):
    """Return (window_ids, bin_centers, M) where M[w, b] = mean metric for window w in step-bin b (NaN if unseen)."""
    steps = df["step"].to_numpy()
    smin, smax = steps.min(), max(steps.max(), steps.min() + 1)
    edges = np.linspace(smin, smax + 1e-9, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    df = df.assign(_bin=np.clip(np.digitize(steps, edges) - 1, 0, nbins - 1))
    pivot = df.pivot_table(index="input_idx", columns="_bin", values=metric, aggfunc="mean")
    pivot = pivot.reindex(columns=range(nbins))
    return pivot.index.to_numpy(), centers, pivot.to_numpy()


def improvement(M):
    """Per-window (early mean - late mean); >0 means loss went down (optimizing)."""
    nb = M.shape[1]
    k = max(1, nb // 5)
    early = np.nanmean(M[:, :k], axis=1)
    late = np.nanmean(M[:, -k:], axis=1)
    return early, late, early - late


def make_figure(df, metric, title, out_path, args):
    if df.empty:
        print(f"  (no rows for {title}, skipping)")
        return None
    wids, centers, M = bin_matrix(df, metric, args.nbins)
    early, late, impr = improvement(M)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # ── left: trajectories + mean/median ────────────────────────────────────
    n = M.shape[0]
    sel = np.arange(n) if n <= args.max_lines else np.random.default_rng(0).choice(n, args.max_lines, replace=False)
    for w in sel:
        axL.plot(centers, M[w], color="0.6", lw=0.4, alpha=0.25)
    axL.plot(centers, np.nanmean(M, axis=0), color="tab:red", lw=2.5, label="mean")
    axL.plot(centers, np.nanmedian(M, axis=0), color="tab:blue", lw=2.0, ls="--", label="median")
    axL.set_xlabel("training step"); axL.set_ylabel(metric)
    axL.set_title(f"{title}\nper-window {metric} over training  (n={n} windows)")
    axL.legend()
    # clip y to p99 so a few exploding windows don't flatten the rest
    hi = np.nanpercentile(M, 99)
    if np.isfinite(hi) and hi > 0:
        axL.set_ylim(0, hi * 1.05)

    # ── right: heatmap sorted by improvement (stuck windows at the bottom) ───
    order = np.argsort(-impr)  # most-improved first
    Ms = M[order]
    if Ms.shape[0] > args.max_heatmap_rows:
        idx = np.linspace(0, Ms.shape[0] - 1, args.max_heatmap_rows).astype(int)
        Ms = Ms[idx]
    vmax = np.nanpercentile(M, 99)
    im = axR.imshow(Ms, aspect="auto", cmap="magma", vmin=0, vmax=vmax,
                    extent=[centers[0], centers[-1], Ms.shape[0], 0], interpolation="nearest")
    axR.set_xlabel("training step")
    axR.set_ylabel("windows (sorted: improved → stuck)")
    axR.set_title(f"{metric} heatmap\nbright band at bottom = NOT optimizing")
    fig.colorbar(im, ax=axR, label=metric)

    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
    print(f"wrote {out_path}")

    # return the per-window improvement table for the non-optimizing report
    return pd.DataFrame({"input_idx": wids, "early_loss": early, "late_loss": late,
                         "improvement": impr, "n_obs": (~np.isnan(M)).sum(axis=1)})


def main():
    args = parse_args()
    csv_path = os.path.join(args.run_dir, "window_loss_history.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run finetune_anomaly_loss_logging.py first "
            f"(it writes this file during training)."
        )
    df = pd.read_csv(csv_path)
    print(f"loaded {len(df)} records over steps {df['step'].min()}..{df['step'].max()}, "
          f"{df['input_idx'].nunique()} distinct windows")

    metric = args.metric
    reports = {}
    reports["anomaly"] = make_figure(df[df.future_type == 1], metric, "ANOMALY windows",
                                     os.path.join(args.run_dir, "window_loss_anomaly.png"), args)
    reports["normal"] = make_figure(df[df.future_type == 0], metric, "NON-anomaly windows",
                                    os.path.join(args.run_dir, "window_loss_normal.png"), args)
    reports["all"] = make_figure(df, metric, "ALL windows",
                                 os.path.join(args.run_dir, "window_loss_all.png"), args)

    # ── report windows that did not optimize (loss did not go down) ──────────
    rows = []
    for group, rep in reports.items():
        if rep is None:
            continue
        stuck = rep[rep["improvement"] <= 0].copy()
        stuck["group"] = group
        rows.append(stuck)
        print(f"  {group}: {len(stuck)}/{len(rep)} windows did NOT optimize ({metric} did not decrease)")
    if rows:
        out = pd.concat(rows).sort_values(["group", "improvement"])
        out_path = os.path.join(args.run_dir, "non_optimizing_windows.csv")
        out.to_csv(out_path, index=False)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
