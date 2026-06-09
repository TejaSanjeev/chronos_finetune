"""
Drop-in variant of finetune_anomaly_simple.py that ALSO logs the per-window
TRAINING loss at every step, so you can see — window by window — whether the
loss is actually going down during fine-tuning (and find the ones that aren't).

It does NOT touch finetune_anomaly_simple.py: it imports and subclasses it.

What it adds
------------
1. The dataset attaches the GLOBAL input index of every row to each batch
   (training samples windows randomly, and the stock batch only keeps local
   group_ids, so otherwise there is no way to know *which* window a loss came
   from).
2. The trainer captures the per-ROW Chronos-2 quantile loss inside compute_loss
   (the same raw L_good / L_bad the objective is built from, captured BEFORE the
   batch-mean), averages the rows of each window, and appends one record per
   window per step to:

       <output_dir>/window_loss_history.csv
       columns: step, input_idx, future_type, raw_loss, train_contrib

   - raw_loss      : the model's forecasting loss on that window (what you watch
                     to see if a window is optimizing).
   - train_contrib : that window's contribution to the hinge objective
                     (= raw_loss for normal windows; = lambda*max(0, tau-raw_loss)
                     for anomaly windows).

Run it exactly like finetune_anomaly_simple.py (same CLI args). Easiest: point
run_finetune.sh at this file, or:

    python finetune_anomaly_loss_logging.py --output_dir ./chronos2-run-logged ...

Then visualize with:  python plot_window_loss_history.py --run_dir ./chronos2-run-logged
"""

import functools
import logging
import os

import numpy as np
import torch
from einops import rearrange

from chronos import BaseChronosPipeline, Chronos2Pipeline
from chronos.chronos2.model import Chronos2Model
import chronos.chronos2.pipeline as chronos2_pipeline

# Reuse everything from the real training script — no logic is duplicated.
from finetune_anomaly_simple import (
    AnomalyChronos2Dataset,
    Chronos2SingleStageTrainer,
    build_lora_config,
    derive_future_type,
    load_data,
    parse_args,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-row loss capture
#  Chronos2Model._compute_loss reduces to a scalar via per_row.mean(); we patch
#  it to stash the per-row vector (detached) and still return the identical
#  scalar, so training is byte-for-byte unchanged.
# ─────────────────────────────────────────────────────────────────────────────

_PERROW_BUFFER: dict = {}
_ORIG_COMPUTE_LOSS = Chronos2Model._compute_loss


def _capturing_compute_loss(
    self, quantile_preds, future_target, future_target_mask,
    patched_future_covariates_mask, loc_scale, num_output_patches,
):
    batch_size = future_target.shape[0]
    ops = self.chronos_config.output_patch_size
    ft, _ = self.instance_norm(future_target, loc_scale)
    ft = ft.unsqueeze(1).to(self.device)
    ftm = (
        future_target_mask.unsqueeze(1).to(self.device)
        if future_target_mask is not None else ~torch.isnan(ft)
    )
    ft = torch.where(ftm > 0.0, ft, 0.0)
    if quantile_preds.shape[-1] > ft.shape[-1]:
        pad = (*ft.shape[:-1], quantile_preds.shape[-1] - ft.shape[-1])
        ft = torch.cat([ft, torch.zeros(pad).to(ft)], dim=-1)
        ftm = torch.cat([ftm, torch.zeros(pad).to(ftm)], dim=-1)
    q = rearrange(self.quantiles, "q -> 1 q 1")
    ql = 2 * torch.abs((ft - quantile_preds) * ((ft <= quantile_preds).float() - q))
    inv = 1 - rearrange(
        patched_future_covariates_mask, "b n p -> b 1 (n p)",
        b=batch_size, n=num_output_patches, p=ops,
    )
    per_row = (ql * (ftm.float() * inv)).mean(dim=-1).sum(dim=-1)  # (rows,)
    _PERROW_BUFFER["per_row"] = per_row.detach().float().cpu()
    # Return the original scalar so the optimizer sees exactly the same loss.
    return _ORIG_COMPUTE_LOSS(
        self, quantile_preds, future_target, future_target_mask,
        patched_future_covariates_mask, loc_scale, num_output_patches,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset — carry the global input index of every row through the batch
# ─────────────────────────────────────────────────────────────────────────────

class LoggingAnomalyDataset(AnomalyChronos2Dataset):
    def _build_batch(self, input_indices):
        batch = super()._build_batch(input_indices)  # already adds future_type
        row_input_idx = []
        for input_idx in input_indices:
            group_size = self.inputs[input_idx]["context"].shape[0]
            row_input_idx.extend([input_idx] * group_size)
        batch["input_idx"] = torch.tensor(row_input_idx, dtype=torch.long)
        return batch


# ─────────────────────────────────────────────────────────────────────────────
#  Trainer — same objective, plus per-window training-loss logging
# ─────────────────────────────────────────────────────────────────────────────

class LoggingSingleStageTrainer(Chronos2SingleStageTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._win_rows: list[tuple] = []

    def _record(self, per_row, idx_rows, ftype):
        """Average per-row losses into per-window means and buffer one row each."""
        step = int(self.state.global_step)
        idx_np = idx_rows.detach().cpu().numpy()
        loss_np = per_row.numpy()
        for gi in np.unique(idx_np):
            m = idx_np == gi
            raw = float(loss_np[m].mean())
            contrib = raw if ftype == 0 else self.margin_lambda * max(0.0, self.margin_tau - raw)
            self._win_rows.append((step, int(gi), int(ftype), raw, contrib))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        future_type = inputs.pop("future_type")
        input_idx = inputs.pop("input_idx", None)
        normal_mask = future_type == 0
        anomaly_mask = future_type == 1
        n_normal = normal_mask.sum().item()
        n_anomaly = anomaly_mask.sum().item()

        outputs = None
        acc = self._acc["train" if model.training else "eval"]
        do_log = model.training and input_idx is not None

        # ── Normal sub-batch → L_good ────────────────────────────────────────
        L_good = torch.zeros((), device=future_type.device)
        if n_normal > 0:
            normal_out = model(**self._select(inputs, normal_mask))
            L_good = normal_out.loss
            outputs = normal_out
            acc["n_sum"] += L_good.detach().item() * n_normal
            acc["n_cnt"] += n_normal
            if do_log:
                self._record(_PERROW_BUFFER["per_row"], input_idx[normal_mask], ftype=0)

        # ── Anomaly sub-batch → L_bad ────────────────────────────────────────
        L_bad = None
        if n_anomaly > 0:
            anom_out = model(**self._select(inputs, anomaly_mask))
            L_bad = anom_out.loss
            acc["a_sum"] += L_bad.detach().item() * n_anomaly
            acc["a_cnt"] += n_anomaly
            if outputs is None:
                outputs = anom_out
            if do_log:
                self._record(_PERROW_BUFFER["per_row"], input_idx[anomaly_mask], ftype=1)

        # ── Combine: identical to the base trainer ───────────────────────────
        total_loss = L_good
        if L_bad is not None:
            hinge = torch.clamp(self.margin_tau - L_bad, min=0.0)
            total_loss = total_loss + self.margin_lambda * hinge

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs, *args, **kwargs):
        out = super().log(logs, *args, **kwargs)  # keeps trainer_state.json behaviour
        if self._win_rows:
            try:
                os.makedirs(self.args.output_dir, exist_ok=True)
                path = os.path.join(self.args.output_dir, "window_loss_history.csv")
                write_header = not os.path.exists(path)
                with open(path, "a", newline="") as fh:
                    if write_header:
                        fh.write("step,input_idx,future_type,raw_loss,train_contrib\n")
                    fh.writelines(
                        f"{s},{i},{ft},{r:.6f},{c:.6f}\n" for (s, i, ft, r, c) in self._win_rows
                    )
                self._win_rows = []
            except Exception as exc:
                logger.warning(f"Could not write window_loss_history.csv: {exc}")
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Main — mirrors finetune_anomaly_simple.main(), swapping in the logging classes
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.lr is None:
        args.lr = 5e-6 if args.finetune_mode == "lora" else 1e-6
    use_fp16 = args.fp16 and args.device != "cpu" and torch.cuda.is_available()

    train_path = os.path.join(args.data_dir, "train_model_inputs.pkl")
    val_path = os.path.join(args.data_dir, "val_model_inputs.pkl")
    train_data = load_data(train_path, "train")
    val_data = load_data(val_path, "val") if not args.no_validation else None

    derive_future_type(train_data, args.anomaly_threshold, "train")
    if val_data is not None:
        derive_future_type(val_data, args.anomaly_threshold, "val")

    logger.info(f"Loading {args.model_id} on {args.device}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id, device_map=args.device
    )
    lora_config = build_lora_config(args) if args.finetune_mode == "lora" else None

    fit_kwargs = dict(
        inputs=train_data,
        prediction_length=args.prediction_length,
        min_past=args.context_length,
        finetune_mode=args.finetune_mode,
        lora_config=lora_config,
        learning_rate=args.lr,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        context_length=args.context_length,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=use_fp16,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        trainer_cls=functools.partial(
            LoggingSingleStageTrainer,
            margin_tau=args.margin_tau,
            margin_lambda=args.margin_lambda,
        ),
    )
    if val_data is not None:
        fit_kwargs["validation_inputs"] = val_data
    if args.enable_sep_token:
        if args.normal_signal_length % args.input_patch_size != 0:
            raise ValueError(
                f"--normal_signal_length ({args.normal_signal_length}) must be a "
                f"multiple of --input_patch_size ({args.input_patch_size})"
            )
        fit_kwargs["enable_sep_token"] = True
        fit_kwargs["sep_patch_index"] = args.normal_signal_length // args.input_patch_size

    # Swap in the logging dataset + patch the per-row loss capture.
    chronos2_pipeline.Chronos2Dataset = LoggingAnomalyDataset
    Chronos2Model._compute_loss = _capturing_compute_loss

    logger.info(
        f"Single-stage training (per-window loss logging ON): lr={args.lr}, "
        f"steps={args.num_steps}, batch={args.batch_size}, margin_tau={args.margin_tau}, "
        f"margin_lambda={args.margin_lambda}, fp16={use_fp16}"
    )
    pipeline.fit(**fit_kwargs)

    ckpt_path = os.path.join(args.output_dir, "finetuned-ckpt")
    logger.info(f"Done. Checkpoint: {ckpt_path}")
    logger.info(f"Per-window loss history: {os.path.join(args.output_dir, 'window_loss_history.csv')}")


if __name__ == "__main__":
    main()
