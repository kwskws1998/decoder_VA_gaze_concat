# Sigma diagnosis experiments

Use Python 3.11 or newer on the NVIDIA machine. Run the following from the
existing GitHub checkout's repository root, where `setup_environment.py` lives.
The entry points are ordinary Python files and execute subprocess argument lists
without invoking a shell.

## Environment setup

```text
git pull --ff-only origin main
python setup_environment.py
```

This creates a local `.venv`, installs the pinned training dependencies, and
records the detected environment in `environment_setup.json`. The experiment
runner automatically uses this Python. Setup reads `nvidia-smi` and chooses
`cu130` for a driver reporting CUDA 13.x or newer, or `cu126` for a CUDA 12.x
driver. It then tests actual CUDA forward/backward execution. The GPU name is
not used to select a wheel: RTX 4090, RTX 5090, or another NVIDIA GPU is checked
on the machine where the script runs.

The pinned PyTorch 2.12.1 CUDA builds are listed in the
[official PyTorch installation archive](https://pytorch.org/get-started/previous-versions/#v2121).
The CUDA-family fallback follows
[NVIDIA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
CUDA-family compatibility alone cannot guarantee every GPU architecture or
operation works; the CUDA test and training smoke run are the execution checks.
If a new GPU requires kernels unavailable in the selected wheel, a newer driver
and compatible wheel are needed. No installation success or GPU memory capacity
is inferred solely from a GPU model name.

To reuse an already prepared Python environment, install into it explicitly:

```text
python setup_environment.py --current-env
```

An explicit `--torch-index cu126` or `--torch-index cu130` overrides automatic
selection. The final execution test still runs. `--gpu 0` selects a physical GPU;
otherwise existing `CUDA_VISIBLE_DEVICES` settings are preserved.

## Experiment execution

```text
python run_sigma_experiments.py --mode dry-run
python run_sigma_experiments.py --mode smoke
python run_sigma_experiments.py --mode full
```

`dry-run` prints the planned commands without installation, data downloads,
model loading or training. `smoke` trains each condition for 20 optimizer steps
on fold 1 and runs its diagnostic probes and held-out prediction. It is a setup
check, not a performance comparison. Run `full` after the smoke check succeeds.
`full` runs 10 epochs on both folds for all three conditions, sequentially, and
packages each completed result automatically. Missing datasets are prepared
from the checksum-pinned source bundle automatically. Full GPU training has not
been executed by the local CPU verification.

All three conditions share full Qwen fine-tuning, TRT-only prefix, sentence-only
data, seed 42, BF16, batch 16, accumulation 1, max length 200, backbone LR 6e-6,
and identical projection/dropout settings. The conditions are:

| Name | Redistribution | Sigma optimization |
|---|---|---|
| raw | None | No sigma parameters |
| fixed | Symmetric Gaussian, initial widths 1/1 | Frozen, excluded from optimizer |
| learned | Asymmetric Gaussian, initial widths 1/1 | LR 1e-3, decay 0 |

The effective positive widths include the existing `min_sigma=1e-6`. Thus the
fixed control uses `1.000001 / 1.000001`, exactly matching the learned arm's
initial kernel. Kernel creation does not consume random numbers, preserving
paired initialization of the other model parameters.

An existing data directory can be supplied by name:

```text
python run_sigma_experiments.py --data-dir /workspace/decoder_VA_gaze_concat/va_model_code/data/paper7_seed42
```

Adjust the shared settings for a different machine using Python arguments:

```text
python run_sigma_experiments.py --batch-size 8 --gradient-accumulation-steps 2
```

The default remains batch 16 / accumulation 1. Batch 8 / accumulation 2 keeps
an effective batch of 16 but is not an assertion of identical training numerics.
All selected conditions share the requested settings. `--precision fp32`,
`--eval-batch-size`, `--seed`, `--epochs`, `--condition learned`, `--gpu`, and
`--probe-batch-size` are also available. `--python` selects an explicit Python
executable; otherwise the runner chooses project `.venv`, then the interpreter
running the script.

The runner reuses both existing fold files. A partial pair is rejected. Source
fold hashes are recorded in each training manifest. Run names include a timestamp
by default; a repeated explicit `--suite-name` fails before overwriting any run.
Changing `--seed` changes the training seed; automatic data preparation retains
the original fold-generation seed 42 for matched comparisons.

## Results and diagnostics

Outputs are below `results/<suite>_full_<raw|fixed|learned>_.../`.

- `oof_metrics.json`: completed two-fold VA metrics.
- `heldout_fold*/sigma_updates.jsonl`: first and each 50th optimizer step;
  accumulated sigma gradients before clipping, gradients after clipping,
  pre-update Adam moments/LR, and actual log-sigma displacement.
- `heldout_fold*/sigma_probes.jsonl`: initialization, each epoch, and the selected
  model after training. Probes use the same first two training examples, with
  dropout disabled and all original parameters temporarily frozen. They measure
  configured/raw/symmetric-1/left-0.5-right-2/left-2-right-0.5 interventions.
  Non-sigma weights remain fixed while gradients are allowed through the decoder.
- `heldout_fold*/checkpoints/trainer_state.json`: complete training history,
  including epochs after the selected best checkpoint.
- `*_results_only.zip`: automatic full-run package, including all three diagnostic
  files above. Model weights and optimizer checkpoint files are excluded.

Probe records include TRT distribution, valid gaze counts, token gaps,
projection/prediction changes, MSE, and individual sample/output/sigma gradients.
`gradient_coherence_left_right` is `abs(mean(g))/mean(abs(g))` over the small
probe batch and both output dimensions; null means all gradients were zero.
Raw and fixed arms have no trainable sigma gradients; their probe derivatives
are explicitly counterfactual local sensitivities, not optimizer gradients.

These fixed two-example probes reveal mechanisms; they are not full-dataset
statistics or validation metrics. Increase `--probe-batch-size`
in the Python runner if a larger probe fits memory. Diagnostics
support single-process BF16/FP32 only and restore modules, train/eval flags,
requires-grad flags and Python/NumPy/Torch RNG states even on probe failure.
Probing can warm ET2's raw-feature cache; all three conditions use the same
probe policy. A long training run requires the Python process to remain running.

## Read the result

1. TRT changes substantially but projector output barely changes: inspect
   projection input scale and LayerNorm sensitivity.
2. Projector changes but predictions/MSE barely change: inspect downstream
   dependence on gaze position.
3. Individual gradients are substantial but their mean is small: inspect
   conflicts between samples or between valence and arousal.
4. Before-clipping gradients are present but actual updates vanish: inspect
   clipping, optimizer moments and precision.
5. Compare fixed versus learned OOF results to isolate the extra effect of
   learning sigma, and raw versus fixed to isolate symmetric smoothing.
