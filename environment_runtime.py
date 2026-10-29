"""Standard-library helpers for the Python environment and experiment entry points."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
GPU_PROBE_CODE = """
import json
import torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable in this Python environment.')
device = torch.cuda.current_device()
bf16 = torch.cuda.is_bf16_supported()
if not bf16:
    raise RuntimeError('The experiments require CUDA BF16 support.')
dtype = torch.bfloat16
value = torch.randn(64, 64, device='cuda', dtype=dtype, requires_grad=True)
(value @ value.T).float().square().mean().backward()
torch.cuda.synchronize()
properties = torch.cuda.get_device_properties(device)
print(json.dumps({
    'gpu': properties.name,
    'compute_capability': list(torch.cuda.get_device_capability(device)),
    'memory_gib': properties.total_memory / 1024**3,
    'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
    'bf16_supported': bf16, 'cuda_forward_backward': 'passed',
}))
"""


def child_environment(gpu=None):
    """Respect existing device visibility unless a GPU is explicitly selected."""

    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return environment


def run(command, *, environment=None, capture=False):
    """Execute argument lists directly, without a shell or shell activation."""

    command = [str(part) for part in command]
    if not capture:
        print("Running:", " ".join(command), flush=True)
    return subprocess.run(
        command, cwd=PROJECT_ROOT, env=environment, check=True,
        text=True, capture_output=capture,
    )


def require_active_conda():
    """Require CONDA_PREFIX, conda metadata and this interpreter to identify one environment."""

    declared = os.environ.get("CONDA_PREFIX")
    prefix = Path(sys.prefix).resolve()
    if (not declared or Path(declared).resolve() != prefix or not (prefix / "conda-meta").is_dir()
            or (prefix / "pyvenv.cfg").exists()):
        raise RuntimeError("Activate a conda environment and use its python; CONDA_PREFIX must equal sys.prefix.")
    return {"prefix": str(prefix), "python": sys.executable,
            "name": os.environ.get("CONDA_DEFAULT_ENV"), "python_version": list(sys.version_info[:3])}


def check_gpu(python, *, gpu=None):
    """Validate CUDA execution and preserve child diagnostics on probe failure."""

    try:
        prefix = list(python) if isinstance(python, (list, tuple)) else [python]
        result = run([*prefix, "-c", GPU_PROBE_CODE], environment=child_environment(gpu), capture=True)
    except subprocess.CalledProcessError as exc:
        details = [f"CUDA forward/backward check failed (exit status {exc.returncode})."]
        if exc.stdout:
            details.append("GPU probe stdout:\n" + exc.stdout.strip())
        if exc.stderr:
            details.append("GPU probe stderr:\n" + exc.stderr.strip())
        details.append("Check nvidia-smi and GPU visibility before starting training.")
        raise RuntimeError("\n".join(details)) from exc
    return json.loads(result.stdout.strip().splitlines()[-1])
