"""One frozen Qwen model with deterministic generation and audited response caching."""

from __future__ import annotations

import json
from pathlib import Path
import time

from .core import digest


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


class QwenBackend:
    """Run prediction, retries, and insight extraction through the same weights."""

    def __init__(
        self, cache_dir: Path, *, model_path: str | None = None,
        device: str = "auto", context_tokens: int = 4096, threads: int = 4,
    ):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if context_tokens < 512 or threads < 1:
            raise ValueError("context_tokens >= 512 and threads >= 1 are required.")
        torch.set_num_threads(threads)
        torch.manual_seed(42)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = device
        dtype = torch.float32 if device == "cpu" else torch.float16
        source = model_path or MODEL_ID
        kwargs = {} if model_path else {"revision": MODEL_REVISION, "cache_dir": str(cache_dir / "hub")}
        self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            source, torch_dtype=dtype, trust_remote_code=False,
            attn_implementation="sdpa", **kwargs,
        ).to(device).eval()
        self.model.requires_grad_(False)
        self.context_tokens = min(context_tokens, self.model.config.max_position_embeddings)
        self.cache_dir = cache_dir / "responses"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.identity = {
            "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "resolved_source": str(Path(model_path).resolve()) if model_path else MODEL_ID,
            "device": device, "dtype": str(dtype),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "context_tokens": self.context_tokens, "threads": threads,
            "generation": "greedy", "weights_updated": False,
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
