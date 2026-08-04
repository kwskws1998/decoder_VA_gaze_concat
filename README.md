# Decoder-based VA prediction

The active training implementation is in `va_model_code`. The
`gaze_concat_code` directory is retained only as a read-only reference to the
upstream GazeConcat implementation.

## Install on a 24 GB NVIDIA machine

Run every installation command from this repository root, which is the
directory containing this README and `requirements.txt`.

```bash
conda create -n decoder-va python=3.11 pip -y
conda activate decoder-va

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
```

Do not install dependencies from `gaze_concat_code`; its original environment
is incompatible with the current Qwen training path.

## Verify and run

```bash
python - <<'PY'
from importlib.metadata import version

import torch
from transformers import Qwen3_5ForCausalLM

print("torch:", version("torch"))
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("transformers:", version("transformers"))
print("peft:", version("peft"))

assert torch.cuda.is_available()
PY

cd va_model_code
python prepare_english_data.py --download-default

cd ..
python -m pytest -q va_model_code/tests

cd va_model_code
python train_model.py --dry-run --no-iemocap
```

The dry run validates the installed Trainer API before downloading the tokenizer
or Qwen weights, so dependency mismatches fail early.

The generated `va_model_code/data` directory is intentionally excluded from
Git. Run the preparation command once on every fresh machine before training.

See `va_model_code/README.md` for the model design and full experiment
commands.
