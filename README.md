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
python -c 'import torch, transformers, peft; from transformers import Qwen3_5ForCausalLM; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), transformers.__version__, peft.__version__); assert torch.cuda.is_available()'

cd va_model_code
python -m pytest -q tests
python train_model.py --dry-run --no-iemocap
```

See `va_model_code/README.md` for the model design and full experiment
commands.
