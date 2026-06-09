# [SEP] Token Implementation for Chronos-2

This document summarises the changes made to introduce an optional `[SEP]`
token between a *normal reference signal* and the actual *context* during
Chronos-2 fine-tuning for anomaly detection.

When enabled, the encoder input sequence becomes:

```
[normal patches] [SEP] [context patches] [REG] [future patches]
```

The `[SEP]` token is a learned embedding from the shared token table and
acts as a boundary marker between the normal reference (everything before
SEP) and the actual historical context (everything between SEP and REG).

---

## How to enable / disable SEP

### Disable (default — original behaviour)

```bash
python inst_data_prepare.py
python finetune_anomaly.py --finetune_mode lora
```

Resulting encoder sequence:

```
[context patches] [REG] [future patches]
```

### Enable

Three flags must line up across the two scripts:

```bash
# Data prep — prepend a normal-zone signal of length 512 to every target
python inst_data_prepare.py --normal_signal_length 512

# Fine-tuning — switch on SEP and tell the script how long the normal part is
python finetune_anomaly.py \
    --finetune_mode lora \
    --enable_sep_token \
    --normal_signal_length 512 \
    --context_length 1024            # = normal_length (512) + actual_context (512)
```

Resulting encoder sequence:

```
[normal patches: 32] [SEP] [context patches: 32] [REG] [future patches]
```

The script computes `sep_patch_index = normal_signal_length / input_patch_size`
and forwards it to `pipeline.fit()`.

---

## Sequence layout, in detail

For the default values used above:

```
target tensor                : (F, 1024 + 64) = (F, 1088)
                                │← 1024 ─►│←64─►│
                                │  ctx   │ fut │   ← what dataset slices

context fed to model          : (batch_F, 1024)
                                │ normal(512) │ context(512) │
patched                       : 64 patches × 16 timesteps
embedded                      : (batch_F, 64, d_model)
SEP inserted at patch_index=32: (batch_F, 65, d_model)
REG appended                  : (batch_F, 66, d_model)
future slots appended         : (batch_F, 66 + num_output_patches, d_model)
```

`sep_patch_index = 32` is fixed by config — the model does not infer it from
the data. Every sample must respect the contract that its first
`sep_patch_index * input_patch_size` timesteps are the normal reference.

---

## Configuration knobs

| Knob | Where | Purpose |
|------|-------|---------|
| `--normal_signal_length` | `inst_data_prepare.py` | length of normal signal prepended to target |
| `--enable_sep_token` | `finetune_anomaly.py` | activates the SEP token in the model |
| `--normal_signal_length` | `finetune_anomaly.py` | used to compute `sep_patch_index` |
| `--context_length` | `finetune_anomaly.py` | must equal `normal_length + actual_context` |
| `--input_patch_size` | `finetune_anomaly.py` | divides `normal_signal_length` to give `sep_patch_index` (default 16) |
| `--warmup_ratio` | `finetune_anomaly.py` | fraction of steps for LR warmup (default 0.05) |
| `--lr_scheduler_type` | `finetune_anomaly.py` | LR schedule: `cosine` (default) or `linear` |
| `--stage2_loss_ceiling` | `finetune_anomaly.py` | cap raw loss before negation in stage 2 (default 15.0; set 0 to disable) |
| `use_sep_token`, `sep_patch_index` | `Chronos2ForecastingConfig` | persisted into the saved checkpoint |

---

## Files changed

| File | Change |
|------|--------|
| `src/chronos/chronos2/config.py` | added `use_sep_token: bool` and `sep_patch_index: int` |
| `src/chronos/chronos2/model.py` | vocab expansion in `__init__`, SEP insertion in `encode()`, updated shape assertion in `forward()` |
| `src/chronos/chronos2/pipeline.py` | new `enable_sep_token` / `sep_patch_index` params on `fit()`, vocab-mismatch handling when loading the pretrained `shared` embedding |
| `src/chronos/chronos2/anomaly_trainer.py` | added `loss_ceiling` param and pre-negation clamp to prevent stage-2 divergence |
| `inst_data_prepare.py` | `extract_normal_signal()`, `--normal_signal_length` CLI flag, normal-signal prepending in `pairs_to_model_inputs()` |
| `finetune_anomaly.py` | `--enable_sep_token`, `--normal_signal_length`, `--input_patch_size`, `--warmup_ratio`, `--lr_scheduler_type`, `--stage2_loss_ceiling` CLI flags; `modules_to_save=["shared"]` in LoRA config when SEP enabled; `functools.partial` wiring of `loss_ceiling` into stage-2 trainer |

The same model-side files (`config.py`, `model.py`, `pipeline.py`,
`anomaly_trainer.py`) must also be copied into the active conda env (see
*Conda env sync* below).

---

## Code changes — key snippets

### `config.py` — declare the flags

```python
@dataclass
class Chronos2ForecastingConfig:
    ...
    use_reg_token: bool = False
    use_sep_token: bool = False
    sep_patch_index: int = 0
```

### `model.py` — vocab expansion

```python
# PAD=0, then optional REG, then optional SEP
vocab_size = 1
if self.chronos_config.use_reg_token:
    config.reg_token_id = vocab_size; vocab_size += 1
if self.chronos_config.use_sep_token:
    config.sep_token_id = vocab_size; vocab_size += 1
config.vocab_size = vocab_size
self.shared = nn.Embedding(config.vocab_size, config.d_model)
```

### `model.py` — SEP insertion in `encode()`

```python
input_embeds = self.input_patch_embedding(patched_context)

# insert [SEP] between normal patches and context patches
if self.chronos_config.use_sep_token:
    sep_idx = self.chronos_config.sep_patch_index
    sep_input_ids = torch.full((batch_size, 1), self.config.sep_token_id, device=input_embeds.device)
    sep_embeds = self.shared(sep_input_ids)
    sep_mask = torch.ones(batch_size, 1, device=self.device)
    input_embeds = torch.cat(
        [input_embeds[:, :sep_idx], sep_embeds, input_embeds[:, sep_idx:]], dim=1
    )
    attention_mask = torch.cat(
        [attention_mask[:, :sep_idx].to(self.dtype),
         sep_mask.to(self.dtype),
         attention_mask[:, sep_idx:].to(self.dtype)],
        dim=1,
    )
```

### `model.py` — shape assertion in `forward()`

```python
num_special_tokens = int(self.chronos_config.use_reg_token) + int(self.chronos_config.use_sep_token)
assert hidden_states.shape == (
    batch_size, num_context_patches + num_special_tokens + num_output_patches, self.model_dim
)
```

### `pipeline.py` — vocab-mismatch handling

When SEP is enabled on a checkpoint that did not have it, the new
`self.shared` table is one row larger. The old rows are copied; the new
SEP row is left at the random initialisation that `Chronos2Model.__init__`
produced and is then trained:

```python
if enable_sep_token and not config.chronos_config.get("use_sep_token", False):
    config.chronos_config["use_sep_token"] = True
    config.chronos_config["sep_patch_index"] = sep_patch_index

model = Chronos2Model(config).to(self.model.device)

old_state = self.model.state_dict()
new_shared = model.shared.weight.data
old_shared = old_state.get("shared.weight")
if old_shared is not None and old_shared.shape[0] < new_shared.shape[0]:
    new_shared[: old_shared.shape[0]] = old_shared.to(new_shared.dtype).to(new_shared.device)
    old_state = {**old_state, "shared.weight": new_shared.clone()}
model.load_state_dict(old_state)
```

### `anomaly_trainer.py` — loss ceiling for stage-2 stability

```python
class Chronos2AnomalyTrainer(Chronos2Trainer):

    def __init__(self, *args, loss_ceiling=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_ceiling = loss_ceiling

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        # Clamp before negating so ascent stops once loss is "bad enough".
        # Without this, the model can diverge to arbitrarily large losses.
        if self.loss_ceiling is not None:
            loss = loss.clamp(max=self.loss_ceiling)

        loss = -loss   # gradient ascent
        return (loss, outputs) if return_outputs else loss
```

### `finetune_anomaly.py` — LoRA config with `modules_to_save`

```python
# modules_to_save ensures the SEP token embedding is trained, not frozen.
# Without this, PEFT freezes shared entirely and SEP stays at random init.
modules_to_save = ["shared"] if args.enable_sep_token else None

return LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=[
        "self_attention.q",
        "self_attention.v",
        "self_attention.k",
        "self_attention.o",
        "output_patch_embedding.output_layer",
    ],
    modules_to_save=modules_to_save,
)
```

### `finetune_anomaly.py` — warmup and scheduler in `build_fit_kwargs`

```python
fit_kwargs = dict(
    ...
    warmup_ratio=args.warmup_ratio,         # default 0.05
    lr_scheduler_type=args.lr_scheduler_type,  # default "cosine"
)
```

### `finetune_anomaly.py` — loss ceiling wired into stage-2 trainer

```python
loss_ceiling = args.stage2_loss_ceiling if args.stage2_loss_ceiling > 0 else None
stage2_fit_kwargs["trainer_cls"] = functools.partial(
    Chronos2AnomalyTrainer, loss_ceiling=loss_ceiling
)
```

### `inst_data_prepare.py` — normal signal extraction & prepending

```python
def extract_normal_signal(data, normal_zones, length):
    """Take 'length' timesteps from the series' normal zones; pad with NaN if short."""
    ...
    return data[:, e - length:e]
```

```python
# In pairs_to_model_inputs, when normal_signal_length > 0:
target = np.concatenate([normal, ctx, fut], axis=1)
out.append({"target": target})
```

---

## Validation

A smoke test was executed in the `chronos_clean` env after syncing the three
modified `src/chronos/chronos2/*.py` files into site-packages:

```python
pipeline.model.chronos_config.use_sep_token = True
pipeline.model.chronos_config.sep_patch_index = 32
# expand vocab (2 → 3) and run a forward pass
out = pipeline.model(context=randn(2, 1024), future_covariates=nan(2, 64),
                     future_target=randn(2, 64), num_output_patches=4)
# → loss: ~8.96, quantile_preds shape: (2, 21, 64)
```

The forward pass returns a finite loss and the expected quantile-prediction
shape, confirming SEP insertion, vocab expansion, and the loss path all
work together.

---

## Training issues found and fixed

### Stage 1 — loss not decreasing

Three root causes were identified from the `trainer_state.json` of the
first run (`checkpoint-3800`, ~1 epoch, LoRA r=8):

#### Bug: SEP token embedding frozen throughout training

`peft.get_peft_model()` freezes all parameters not listed in
`target_modules` or `modules_to_save`. The `shared` embedding — which
holds the newly added SEP token at index 2 — was not in either list, so
it remained at its random initialisation for all training steps:

```
shared: requires_grad=False, shape=torch.Size([2, 768])
```

Every batch received a meaningless random vector at the boundary position.
The attention layers fought this noise instead of learning the
normal-vs-context distinction.

**Fix:** added `modules_to_save=["shared"]` to the `LoraConfig` whenever
`--enable_sep_token` is set. PEFT then treats `shared` as a fully-trainable
module, saves it in the adapter checkpoint, and restores it on reload.

#### Issue: aggressive linear LR decay with zero warmup

`pipeline.py` hardcodes `lr_scheduler_type="linear"` and
`warmup_ratio=0.0`. The LR decays from `1e-5` to ~`0` across all steps,
so by the halfway point the effective learning rate is already halved.

**Fix:** added `--warmup_ratio` (default `0.05`) and
`--lr_scheduler_type` (default `cosine`) CLI args, forwarded as
extra kwargs to `TrainingArguments` via `pipeline.fit()`.

#### Issue: only 1 epoch of training

With 306k samples and effective batch size 80, one epoch ≈ 3,831 steps.
The first run set `max_steps=4000` (≈1 epoch). The eval loss was still
trending downward when training stopped — it never plateaued.

**Recommended fix:** default `STAGE1_STEPS` raised to `12000` (~3 epochs)
in `run_finetune.sh`.

#### Secondary: LoRA rank too small

`r=8` gives the model little capacity to learn a completely new input
paradigm. Default raised to `r=16, alpha=32` in `run_finetune.sh`.

---

### Stage 2 — gradient ascent divergence

#### Bug: same frozen SEP token

`modules_to_save: null` was also present in the stage-2 adapter config.
The same `LoraConfig` fix from stage 1 covers stage 2 automatically since
both stages call `build_lora_config(args)`.

#### Bug: no loss ceiling → gradient ascent diverges

Without a bound on how far loss can grow, the model finds the easy escape:
push its output distribution towards infinite variance. The non-SEP stage-2
run demonstrates this clearly:

| Step | grad_norm | train loss |
|------|-----------|------------|
| 100  | 3.96      | −6.1       |
| 300  | 19.7      | −7.8       |
| 400  | **68**    | −16.7      |
| 500  | **453**   | −48.9      |
| 700  | **852**   | −258       |
| 3000 | 362       | **−1197**  |

The SEP stage-2 run escaped only because it stopped at 500 steps by
coincidence; its own grad_norm trend (2.6 → 6.7 → 11.6 → 25.4) was on
the same trajectory.

**Fix:** added `loss_ceiling` parameter to `Chronos2AnomalyTrainer`.
The raw (positive) loss is clamped to this ceiling *before* negation.
Once the model predicts badly enough, gradients go to zero and ascent
stops naturally:

```python
if self.loss_ceiling is not None:
    loss = loss.clamp(max=self.loss_ceiling)
loss = -loss
```

Wired into `finetune_anomaly.py` via `--stage2_loss_ceiling` (default
`15.0`, ≈5× the stage-1 final eval loss of ~3.28) and passed to the
trainer with `functools.partial`.

#### Issue: stage 2 starts from a broken stage 1 base

The first stage-2 run loaded from the stage-1 checkpoint that had the
frozen SEP token throughout. The intended workflow is:

1. Re-run stage 1 with the SEP fix applied.
2. Re-run stage 2 from the corrected stage-1 checkpoint.

---

## Recommended training configuration (after all fixes)

```bash
# run_finetune.sh defaults after all fixes
STAGE1_STEPS=12000      # ~3 epochs
STAGE1_LR=2e-5
STAGE2_STEPS=1000       # safe with loss ceiling
STAGE2_LR=5e-6          # lower: ascent is less stable than descent
STAGE2_LOSS_CEILING=15.0
LORA_R=16
LORA_ALPHA=32
WARMUP_RATIO=0.05
LR_SCHEDULER=cosine
```

Or via CLI:

```bash
python finetune_anomaly.py \
    --finetune_mode lora \
    --enable_sep_token \
    --normal_signal_length 256 \
    --context_length 768 \
    --lora_r 16 --lora_alpha 32 \
    --stage1_lr 2e-5 --stage1_steps 12000 \
    --stage2_lr 5e-6 --stage2_steps 1000 \
    --stage2_loss_ceiling 15.0 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine
```

---

## Conda env sync

This repo's local `src/chronos` is **not** installed in editable mode, so
any change to the model-side files must be copied into the conda env
before it takes effect:

```bash
cp src/chronos/chronos2/{config,model,pipeline,anomaly_trainer}.py \
   ~/miniconda3/envs/chronos_clean/lib/python3.10/site-packages/chronos/chronos2/
```

(The data-prep and fine-tune scripts live at the repo root and run from
there, so they don't need to be copied.)

---

## Caveats

1. **The SEP token is random-initialised.** It carries no semantic meaning
   until fine-tuning teaches it one. In the anomaly setup the model sees
   normal-zone data before every SEP, so it learns to treat the pre-SEP
   region as a reference baseline.

2. **Time encoding still treats normal patches as "older history".** The
   patches before SEP receive negative time indices continuous with the
   context. If you want the model to *clearly* distinguish them from
   real history (as opposed to "long-ago" history), zero out the time
   encoding for the first `sep_patch_index` patches inside
   `_prepare_patched_context`. This is intentionally left for later — the
   architecture works without it.

3. **Fixed boundary.** `sep_patch_index` is a single config integer, so
   every sample must use the same normal-signal length. Variable-length
   normal signals would require passing a per-sample `sep_positions`
   tensor through the batch.

4. **Vocab change persists in the saved checkpoint.** Once you fine-tune
   with `enable_sep_token=True`, the saved checkpoint has
   `use_sep_token=True` in its config and `shared.weight` is `(3, d_model)`.
   Loading it back via `from_pretrained` keeps SEP enabled automatically;
   you don't need to re-pass `enable_sep_token` on the second fine-tuning
   stage.

5. **Stage-2 loss ceiling should scale with stage-1 final loss.** The
   default of `15.0` assumes a stage-1 final eval loss of ~3–5. If your
   stage-1 converges to a different value, adjust accordingly: a ceiling
   of 3–5× the stage-1 final eval loss is a reasonable starting point.
