---
name: Conda env sync required for local src changes
description: Changes to src/chronos/ must be manually copied to the chronos_clean conda env site-packages
type: feedback
originSessionId: 9e4c9af0-ff22-4fa5-b3ef-065feb2cb583
---
Local edits to `src/chronos/` are NOT automatically picked up at runtime — the package is not installed in editable mode.

**Why:** The `chronos_clean` conda env uses the installed copy at `site-packages/chronos/`. Running scripts with that env picks up the installed version, not local `src/`.

**How to apply:** After modifying any file under `src/chronos/`, copy it to the conda env:

```bash
cp src/chronos/chronos2/<file>.py \
   /home/rajib/miniconda3/envs/chronos_clean/lib/python3.10/site-packages/chronos/chronos2/<file>.py
```

Files modified so far and already synced:
- `src/chronos/chronos2/pipeline.py` — `trainer_cls` support in `fit()`, LoRA adapter loading in `from_pretrained`
- `src/chronos/chronos2/dataset.py` — `convert_inputs` parameter, `PreparedInput` TypedDict, renamed `validate_and_prepare_single_dict_input`
