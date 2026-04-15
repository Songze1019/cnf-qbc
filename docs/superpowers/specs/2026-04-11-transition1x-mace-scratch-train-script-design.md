# Transition1x MACE Scratch Train Script Design

## Goal

Add a shell script that trains a MACE model from scratch on the deduplicated Transition1x extxyz data without using a foundation model.

## Inputs

- `data/transition1x/train.xyz`
- `data/transition1x/val.xyz`
- `data/transition1x/test.xyz`

These files store energies under the ASE/extxyz key `energy` and forces under `forces`.

## Behavior

The script should:

- follow the style of `scripts/train.sh`
- call `mace_run_train` directly
- use separate train, validation, and test files
- default to training from scratch
- expose common hyperparameters and paths as environment-variable overrides
- write logs, checkpoints, and results under a dedicated work directory in `rets/`
- allow the user to append extra `mace_run_train` arguments at invocation time

## Defaults

Use conservative defaults close to the existing local training script:

- `E0s=average`
- `hidden_irreps='128x0e + 128x1o'`
- `r_max=6.0`
- `batch_size=16`
- `max_num_epochs=50`
- `default_dtype=float32`
- `device=cuda`
- `forces_weight=10.0`
- `ema`
- `amsgrad`
- `scheduler_patience=5`
- `enable_cueq=True`

## Output

Create one runnable script in `scripts/` with clear variable names and input validation. No training is run automatically.
