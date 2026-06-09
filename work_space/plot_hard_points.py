"""
Visualize which data points the fine-tuned Chronos-2 model fits poorly.

Consumes per_sample_loss.npz produced by analyze_per_sample_loss.py and writes:

  1. loss_distribution.png  — three diagnostic panels:
       (a) histogram of per-window loss, split normal vs anomaly
       (b) sorted loss curve (rank vs loss) with p90/p95 "hard tail" lines
       (c) #anomalous-steps-in-window vs loss scatter
  2. worst_windows.png      — the K worst-fit windows: context tail + true
       future + model median forecast and 10–90% band, so you can SEE why
       they are hard. Use --only_normal to inspect the hardest *normal*
       windows (the surprising ones the model should fit but doesn't).

Usage
-----
    python plot_hard_points.py                       # defaults -> V4
    python plot_hard_points.py --only_normal         # worst normal windows
    python plot_hard_points.py --topk 12 --no_forecasts   # skip model reload
"""

import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot hard-to-fit windows from per_sample_loss.npz")
    p.add_argument("--run_dir", default="./chronos2-single-stage_NS1000_V4",
                   help="Dir containing per_sample_loss.npz (and where PNGs are written)")
    p.add_argument("--ckpt", default=None, help="Checkpoint for forecasts (default: <run_dir>/finetuned-ckpt)")
    p.add_argument("--data", default="./prepared_data_labeled/train_model_inputs.pkl")
    p.add_argument("--device", default="cuda")
    p.add_argument("--context_length", type=int, default=768)
    p.add_argument("--prediction_length", type=int, default=64)
    p.add_argument("--topk", type=int, default=8, help="How many worst windows to plot")
    p.add_argument("--ctx_tail", type=int, default=192, help="Context steps to show before the future")
    p.add_argument("--only_normal", action="store_true",
                   help="Plot worst NORMAL windows (future_type==0) instead of worst overall")
    p.add_argument("--no_forecasts", action="store_true",
                   help="Skip model reload; worst_windows.png shows truth only (no forecast overlay)")
    return p.parse_args()


def plot_distribution(d, out_path):
    loss = d["loss"]; ft = d["future_type"]; n_anom = d["n_anom_steps"]
    norm = loss[ft == 0]; anom = loss[ft == 1]

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # (a) histogram, clipped at p99 so the tail doesn't flatten the bulk
    clip = np.nanpercentile(loss, 99)
    bins = np.linspace(0, clip, 60)
    ax[0].hist(np.clip(norm, 0, clip), bins=bins, alpha=0.6, label=f"normal (n={len(norm)})", color="tab:blue")
    ax[0].hist(np.clip(anom, 0, clip), bins=bins, alpha=0.6, label=f"anomaly (n={len(anom)})", color="tab:red")
    ax[0].axvline(np.nanmedian(norm), color="tab:blue", ls="--", lw=1)
    if len(anom):
        ax[0].axvline(np.nanmedian(anom), color="tab:red", ls="--", lw=1)
    ax[0].set_xlabel(f"per-window loss (clipped at p99={clip:.1f})")
    ax[0].set_ylabel("count"); ax[0].set_title("(a) Loss distribution by window type"); ax[0].legend()

    # (b) sorted loss curve — the rising tail = poorly fit windows
    order = np.argsort(loss)
    rank = np.arange(len(loss))
    colors = np.where(ft[order] == 1, "tab:red", "tab:blue")
    ax[1].scatter(rank, loss[order], s=4, c=colors)
    for q, style in [(90, ":"), (95, "--")]:
        v = np.nanpercentile(loss, q)
        ax[1].axhline(v, color="k", ls=style, lw=1, label=f"p{q}={v:.1f}")
    ax[1].set_xlabel("window rank (sorted by loss)"); ax[1].set_ylabel("loss")
    ax[1].set_title("(b) Sorted loss — rising tail = hard windows"); ax[1].legend()

    # (c) how anomalous a window is vs how hard it is
    ax[2].scatter(n_anom, loss, s=5, alpha=0.25, c=np.where(ft == 1, "tab:red", "tab:blue"))
    ax[2].set_xlabel("# anomalous steps in 64-step future"); ax[2].set_ylabel("loss")
    ax[2].set_title("(c) Anomaly content vs loss")

    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
    print(f"wrote {out_path}")


def plot_worst_windows(args, d):
    loss = d["loss"]; ft = d["future_type"]; n_anom = d["n_anom_steps"]
    pool = np.where(ft == 0)[0] if args.only_normal else np.arange(len(loss))
    worst = pool[np.argsort(-loss[pool])[: args.topk]]

    with open(args.data, "rb") as f:
        data = pickle.load(f)

    C, H = args.context_length, args.prediction_length
    tail = args.ctx_tail

    forecasts = {}
    if not args.no_forecasts:
        import torch
        from chronos.chronos2.pipeline import Chronos2Pipeline
        from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
        ckpt = args.ckpt or os.path.join(args.run_dir, "finetuned-ckpt")
        print(f"loading {ckpt} for forecast overlays ...")
        pipe = Chronos2Pipeline.from_pretrained(ckpt, device_map=args.device)
        model = pipe.model; model.eval()
        cc = model.chronos_config
        sub = [{"target": data[i]["target"]} for i in worst]
        ds = Chronos2Dataset(sub, context_length=C, prediction_length=H, batch_size=10**9,
                             output_patch_size=cc.output_patch_size, min_past=C,
                             mode=DatasetMode.VALIDATION, convert_inputs=True)
        batch = ds._build_batch(list(range(len(sub))))
        gids = batch["group_ids"]; batch.pop("target_idx_ranges", None)
        feed = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**feed)
        qp = out.quantile_preds.float().cpu().numpy()  # (rows, n_q, horizon)
        quantiles = list(cc.quantiles)
        qlo, qmd, qhi = quantiles.index(0.1), quantiles.index(0.5), quantiles.index(0.9)
        for local_g, global_i in enumerate(worst):
            first_row = int(np.where(gids.numpy() == local_g)[0][0])  # first (target) variate
            forecasts[global_i] = (qp[first_row, qlo, :H], qp[first_row, qmd, :H], qp[first_row, qhi, :H])

    ncol = 2
    nrow = int(np.ceil(len(worst) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 2.6 * nrow), squeeze=False)
    for ax, i in zip(axes.ravel(), worst):
        target = data[i]["target"][0]                  # first variate, full 832
        ctx = target[C - tail : C]
        fut_true = target[C : C + H]
        x_ctx = np.arange(-tail, 0)
        x_fut = np.arange(0, H)
        ax.plot(x_ctx, ctx, color="0.4", lw=1, label="context")
        ax.plot(x_fut, fut_true, color="tab:green", lw=1.6, label="true future")
        if i in forecasts:
            lo, md, hi = forecasts[i]
            ax.plot(x_fut, md, color="tab:orange", lw=1.4, label="forecast (median)")
            ax.fill_between(x_fut, lo, hi, color="tab:orange", alpha=0.2, label="10–90%")
        ax.axvline(0, color="k", lw=0.6, ls=":")
        kind = "anomaly" if ft[i] == 1 else "normal"
        ax.set_title(f"idx {int(i)} | loss={loss[i]:.2f} | {kind} | anom_steps={int(n_anom[i])}", fontsize=9)
    for ax in axes.ravel()[len(worst):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(("Worst-fit NORMAL windows" if args.only_normal else "Worst-fit windows")
                 + " (true vs forecast)", y=1.0)
    fig.tight_layout()
    out_path = os.path.join(args.run_dir, "worst_windows.png")
    fig.savefig(out_path, dpi=120); plt.close(fig)
    print(f"wrote {out_path}")


def main():
    args = parse_args()
    npz = os.path.join(args.run_dir, "per_sample_loss.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(f"{npz} not found. Run analyze_per_sample_loss.py first.")
    d = dict(np.load(npz))
    plot_distribution(d, os.path.join(args.run_dir, "loss_distribution.png"))
    plot_worst_windows(args, d)


if __name__ == "__main__":
    main()
