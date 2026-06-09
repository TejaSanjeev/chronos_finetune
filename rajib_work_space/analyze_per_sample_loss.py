"""
Per-sample (per-window) loss analysis for a fine-tuned Chronos-2 anomaly model.

The training logs in trainer_state.json only store *aggregate* losses per step
(loss / normal_loss / anomaly_loss). That tells you the category trend but NOT
*which individual data points* the model fits poorly. This script fills that gap:
it runs the fine-tuned model over every prepared window once (deterministic last
window, exactly the slice training/validation uses) and records the Chronos-2
quantile loss for EACH window, so you can rank and visualize the hard samples.

How the per-window loss is obtained
-----------------------------------
Chronos2Model._compute_loss reduces to a scalar with `.mean(dim=-1).sum(dim=-1).mean()`
— the final `.mean()` averages over the batch rows. We monkeypatch _compute_loss to
also stash the per-ROW vector (everything before that final mean). Each input is a
multivariate group (2..55 variates here), so we average the rows of each group via
`group_ids` to get one loss per WINDOW.

Output
------
<output_dir>/per_sample_loss.npz  and  per_sample_loss.csv  with columns:
    index         position in the original train_model_inputs.pkl
    future_type   0 = normal window, 1 = anomaly window (>= threshold anomalous steps)
    n_anom_steps  number of anomalous timesteps in the 64-step future window
    n_variates    group size (number of series) for this window
    loss          per-window Chronos-2 quantile loss (higher = worse fit)

Usage
-----
    python analyze_per_sample_loss.py                       # defaults -> V4
    python analyze_per_sample_loss.py --limit 1000          # quick subset
    python analyze_per_sample_loss.py --ckpt <dir> --data <pkl>
"""

import argparse
import logging
import os
import pickle

import numpy as np
import torch
from einops import rearrange

from chronos.chronos2.pipeline import Chronos2Pipeline
from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
from chronos.chronos2.model import Chronos2Model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Per-window loss analysis for fine-tuned Chronos-2")
    p.add_argument("--ckpt", default="./chronos2-single-stage_NS1000_V4/finetuned-ckpt",
                   help="Fine-tuned adapter checkpoint dir")
    p.add_argument("--data", default="./prepared_data_labeled/train_model_inputs.pkl",
                   help="Prepared inputs pickle (list of {target, future_labels})")
    p.add_argument("--output_dir", default=None,
                   help="Where to write per_sample_loss.{npz,csv} (default: parent of --ckpt)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--anomaly_threshold", type=int, default=10,
                   help="A window is anomaly (future_type=1) iff it has >= this many anomalous steps")
    p.add_argument("--context_length", type=int, default=768)
    p.add_argument("--prediction_length", type=int, default=64)
    p.add_argument("--row_batch", type=int, default=512,
                   help="Accumulate inputs until this many series (rows), then run one forward")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N inputs (debug)")
    return p.parse_args()


# Buffer the most recent per-row loss vector produced inside the model.
_CAPTURE: dict[str, torch.Tensor] = {}
_ORIG_COMPUTE_LOSS = Chronos2Model._compute_loss


def _capturing_compute_loss(
    self,
    quantile_preds,
    future_target,
    future_target_mask,
    patched_future_covariates_mask,
    loc_scale,
    num_output_patches,
):
    """Replicates Chronos2Model._compute_loss but stashes the per-row loss before
    the final batch mean, then defers to the original for the returned scalar."""
    batch_size = future_target.shape[0]
    output_patch_size = self.chronos_config.output_patch_size

    ft, _ = self.instance_norm(future_target, loc_scale)
    ft = ft.unsqueeze(1).to(self.device)
    ftm = (
        future_target_mask.unsqueeze(1).to(self.device)
        if future_target_mask is not None
        else ~torch.isnan(ft)
    )
    ft = torch.where(ftm > 0.0, ft, 0.0)
    if quantile_preds.shape[-1] > ft.shape[-1]:
        pad = (*ft.shape[:-1], quantile_preds.shape[-1] - ft.shape[-1])
        ft = torch.cat([ft, torch.zeros(pad).to(ft)], dim=-1)
        ftm = torch.cat([ftm, torch.zeros(pad).to(ftm)], dim=-1)

    quantiles = rearrange(self.quantiles, "q -> 1 q 1")
    quantile_loss = 2 * torch.abs((ft - quantile_preds) * ((ft <= quantile_preds).float() - quantiles))
    inv_cov_mask = 1 - rearrange(
        patched_future_covariates_mask, "b n p -> b 1 (n p)",
        b=batch_size, n=num_output_patches, p=output_patch_size,
    )
    loss_mask = ftm.float() * inv_cov_mask
    per_row = (quantile_loss * loss_mask).mean(dim=-1).sum(dim=-1)  # (batch_rows,)
    _CAPTURE["per_row"] = per_row.detach().float().cpu()

    return _ORIG_COMPUTE_LOSS(
        self, quantile_preds, future_target, future_target_mask,
        patched_future_covariates_mask, loc_scale, num_output_patches,
    )


def main():
    args = parse_args()
    out_dir = args.output_dir or os.path.dirname(os.path.normpath(args.ckpt))
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Loading data from {args.data}")
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    if args.limit:
        data = data[: args.limit]
    logger.info(f"  {len(data)} inputs")

    # Window-level metadata, kept aligned with the original input order.
    n_anom_steps = np.array(
        [int(np.sum(d["future_labels"])) if d.get("future_labels") is not None else 0 for d in data]
    )
    future_type = (n_anom_steps >= args.anomaly_threshold).astype(int)
    # Strip future_labels so the stock dataset's key validation passes.
    cleaned = [{"target": d["target"]} for d in data]

    logger.info(f"Loading fine-tuned model from {args.ckpt} on {args.device}")
    pipe = Chronos2Pipeline.from_pretrained(args.ckpt, device_map=args.device)
    model = pipe.model
    model.eval()
    cc = model.chronos_config
    logger.info(f"  use_sep_token={cc.use_sep_token} sep_patch_index={cc.sep_patch_index} "
                f"output_patch_size={cc.output_patch_size}")

    ds = Chronos2Dataset(
        cleaned,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        batch_size=10**9,  # unused: we drive batching ourselves
        output_patch_size=cc.output_patch_size,
        min_past=args.context_length,
        mode=DatasetMode.VALIDATION,
        convert_inputs=True,
    )
    if len(ds.inputs) != len(cleaned):
        raise RuntimeError(
            f"Length filtering dropped inputs ({len(cleaned)} -> {len(ds.inputs)}); "
            "index alignment with future_type would break. Ensure every target is "
            ">= context_length + prediction_length steps."
        )

    Chronos2Model._compute_loss = _capturing_compute_loss

    n = len(ds.inputs)
    losses = np.full(n, np.nan, dtype=np.float64)
    n_variates = np.array([ds.inputs[i]["context"].shape[0] for i in range(n)])

    # Group inputs into forward passes by accumulating rows up to --row_batch.
    start = 0
    done = 0
    while start < n:
        rows = 0
        end = start
        while end < n and (rows == 0 or rows + n_variates[end] <= args.row_batch):
            rows += int(n_variates[end])
            end += 1
        idx = list(range(start, end))

        batch = ds._build_batch(idx)
        gids = batch["group_ids"]
        batch.pop("target_idx_ranges", None)
        feed = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            model(**feed)
        per_row = _CAPTURE["per_row"]
        # Local group g -> global input index idx[g]; window loss = mean over its rows.
        for local_g, global_i in enumerate(idx):
            losses[global_i] = float(per_row[gids == local_g].mean())

        done += len(idx)
        start = end
        if done % 2048 < len(idx):
            logger.info(f"  {done}/{n} windows")

    Chronos2Model._compute_loss = _ORIG_COMPUTE_LOSS

    index = np.arange(n)
    npz_path = os.path.join(out_dir, "per_sample_loss.npz")
    csv_path = os.path.join(out_dir, "per_sample_loss.csv")
    np.savez(npz_path, index=index, future_type=future_type[:n], n_anom_steps=n_anom_steps[:n],
             n_variates=n_variates, loss=losses)
    # Lightweight CSV without pandas.
    header = "index,future_type,n_anom_steps,n_variates,loss"
    rows_out = np.column_stack([index, future_type[:n], n_anom_steps[:n], n_variates, losses])
    np.savetxt(csv_path, rows_out, fmt=["%d", "%d", "%d", "%d", "%.6f"], delimiter=",", header=header, comments="")

    # ── Summary ───────────────────────────────────────────────────────────────
    norm = losses[future_type[:n] == 0]
    anom = losses[future_type[:n] == 1]
    logger.info("=" * 60)
    logger.info(f"Saved {npz_path}")
    logger.info(f"Saved {csv_path}")
    logger.info(f"Windows: {n}  (normal={len(norm)}, anomaly={len(anom)})")
    logger.info(f"  normal  loss: mean={np.nanmean(norm):.3f}  median={np.nanmedian(norm):.3f}  p95={np.nanpercentile(norm,95):.3f}")
    if len(anom):
        logger.info(f"  anomaly loss: mean={np.nanmean(anom):.3f}  median={np.nanmedian(anom):.3f}  p95={np.nanpercentile(anom,95):.3f}")
    worst = np.argsort(-losses)[:10]
    logger.info("Top-10 worst-fit windows (index : loss : future_type : n_anom_steps):")
    for i in worst:
        logger.info(f"  {int(i):6d} : {losses[i]:7.3f} : ftype={int(future_type[i])} : anom_steps={int(n_anom_steps[i])}")


if __name__ == "__main__":
    main()
