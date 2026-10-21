"""One frozen Qwen model with deterministic generation and audited response caching."""

from __future__ import annotations

import json
from pathlib import Path
import time

from .core import digest


MODEL_SPECS = {
    "0.5b": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "hidden_size": 896, "num_hidden_layers": 24,
    },
    "3b": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "hidden_size": 2048, "num_hidden_layers": 36,
    },
}
MODEL_ID = MODEL_SPECS["0.5b"]["model_id"]
MODEL_REVISION = MODEL_SPECS["0.5b"]["revision"]
GENERATION_POLICY = {
    "do_sample": False, "num_beams": 1, "repetition_penalty": 1.1,
    "temperature": 1.0, "top_p": 1.0, "top_k": 50,
}


def model_spec(model_size: str) -> dict:
    """Resolve a supported model to pinned weights and expected architecture."""

    if model_size not in MODEL_SPECS:
        raise ValueError(f"Unknown model size {model_size!r}; choose one of {tuple(MODEL_SPECS)}.")
    return dict(MODEL_SPECS[model_size])


def validate_model_config(config, model_size: str) -> None:
    """Reject mislabeled local model directories before loading large weight files."""

    spec = model_spec(model_size)
    if (
        config.model_type != "qwen2"
        or config.hidden_size != spec["hidden_size"]
        or config.num_hidden_layers != spec["num_hidden_layers"]
    ):
        raise ValueError(f"Selected model size {model_size} does not match the loaded architecture.")


def configure_generation(config) -> None:
    """Override model-specific decoding defaults for a matched size comparison."""

    for name, value in GENERATION_POLICY.items():
        setattr(config, name, value)


class QwenBackend:
    """Run prediction, retries, and insight extraction through the same weights."""

    def __init__(
        self, cache_dir: Path, *, model_path: str | None = None,
        device: str = "auto", context_tokens: int = 4096, threads: int = 4,
        model_size: str = "0.5b",
    ):
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if context_tokens < 512 or threads < 1:
            raise ValueError("context_tokens >= 512 and threads >= 1 are required.")
        spec = model_spec(model_size)
        torch.set_num_threads(threads)
        torch.manual_seed(42)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = device
        dtype = torch.float32 if device == "cpu" else torch.float16
        source = model_path or spec["model_id"]
        model_cache = Path(__file__).resolve().parents[1] / "artifacts/expel/hub"
        kwargs = {} if model_path else {"revision": spec["revision"], "cache_dir": str(model_cache)}
        config = AutoConfig.from_pretrained(source, trust_remote_code=False, **kwargs)
        validate_model_config(config, model_size)
        self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            source, config=config, torch_dtype=dtype, trust_remote_code=False,
            attn_implementation="sdpa", **kwargs,
        ).to(device).eval()
        self.model.requires_grad_(False)
        configure_generation(self.model.generation_config)
        self.context_tokens = min(context_tokens, self.model.config.max_position_embeddings)
        self.cache_dir = cache_dir / "responses"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.identity = {
            "model_id": spec["model_id"], "revision": spec["revision"],
            "resolved_source": str(Path(model_path).resolve()) if model_path else spec["model_id"],
            "device": device, "dtype": str(dtype),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "context_tokens": self.context_tokens, "threads": threads,
            "generation": "greedy", "weights_updated": False,
            "generation_parameters": {
                **GENERATION_POLICY,
                "eos_token_id": self.model.generation_config.eos_token_id,
                "pad_token_id": self.tokenizer.eos_token_id,
            },
        }
        self.generated_calls = 0
        self.cached_calls = 0

    def token_count(self, messages: list[dict]) -> int:
        """Count the actual templated input without truncating target text."""

        return len(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        ))

    def generate(self, messages: list[dict], max_new_tokens: int = 48) -> str:
        """Reject context overflow and persist complete request/response provenance."""

        import torch

        count = self.token_count(messages)
        if count + max_new_tokens > self.context_tokens:
            raise ValueError(
                f"Prompt needs {count}+{max_new_tokens} tokens, budget {self.context_tokens}; "
                "no target text was silently truncated."
            )
        request = {"backend": self.identity, "messages": messages, "max_new_tokens": max_new_tokens}
        path = self.cache_dir / f"{digest(request)}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            if cached["request"] != request:
                raise ValueError("Response cache identity mismatch.")
            self.cached_calls += 1
            return cached["output"]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").to(self.device)
        started = time.monotonic()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        value = {
            "request": request, "output": answer, "input_tokens": count,
            "output_tokens": int(generated.shape[1] - inputs["input_ids"].shape[1]),
            "elapsed_seconds": time.monotonic() - started,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)
        self.generated_calls += 1
        return answer
