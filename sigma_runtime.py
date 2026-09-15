"""Standard-library helpers for the Python environment and experiment entry points."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ENVIRONMENT = PROJECT_ROOT / ".venv"
GPU_PROBE_CODE = """
import json
import torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable in this Python environment.')
device = torch.cuda.current_device()
bf16 = torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if bf16 else torch.float32
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


def environment_python(directory):
    """Resolve the interpreter inside a standard Python virtual environment."""

    directory = Path(directory).expanduser().resolve()
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def select_python(explicit=None, environment_dir=DEFAULT_ENVIRONMENT):
    """Use an explicit interpreter, the project virtual environment, or current Python."""

    if explicit is not None:
        selected = Path(os.path.abspath(Path(explicit).expanduser()))
        if not selected.is_file():
            raise FileNotFoundError(f"Python executable not found: {selected}")
        return selected
    selected = environment_python(environment_dir)
    return selected if selected.is_file() else Path(sys.executable)


def detect_nvidia_driver():
    """Read driver CUDA capability and GPU names from NVIDIA's XML inventory."""

    result = run(["nvidia-smi", "-q", "-x"], capture=True)
    root = ET.fromstring(result.stdout)
    cuda = root.findtext("cuda_version", default="")
    match = re.fullmatch(r"(\d+)\.(\d+)", cuda.strip())
    if match is None:
        raise RuntimeError(f"Cannot read driver CUDA version from nvidia-smi: {cuda!r}")
    return {
        "driver_version": root.findtext("driver_version"),
        "driver_cuda": [int(match[1]), int(match[2])],
        "gpus": [gpu.findtext("product_name") for gpu in root.findall("gpu")],
    }


def choose_torch_index(driver_cuda):
    """Select official pinned-version wheels by driver CUDA family, not GPU name."""

    version = tuple(driver_cuda)
    if version >= (13, 0):
        return "cu130"
    if version >= (12, 0):
        return "cu126"
    raise RuntimeError("The pinned PyTorch environment requires a CUDA 12.x or newer NVIDIA driver.")


def check_gpu(python, *, gpu=None):
    """Validate actual CUDA forward/backward execution in the selected interpreter."""

    result = run([python, "-c", GPU_PROBE_CODE], environment=child_environment(gpu), capture=True)
    return json.loads(result.stdout.strip().splitlines()[-1])
