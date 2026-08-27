"""OpenAI-compatible chat client. Point LLM_BASE_URL at Ollama (or any /v1 server)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI

from config.settings import CHAT_PRICE_PER_MILLION, LLM_BASE_URL, OPENAI_API_KEY


@dataclass
class CompletionResult:
    text: str
    model: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float

    def usage_only(self) -> dict:
        return {
            "model": self.model,
            "latency_s": round(self.latency_s, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


def empty_usage(model: str) -> dict:
    """Usage for a call that never happened, so summaries can stay arithmetic."""
    return {
        "model": model,
        "latency_s": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
    }


def token_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    # A local server bills nothing even though it still reports token counts.
    if LLM_BASE_URL:
        return 0.0
    prices = CHAT_PRICE_PER_MILLION.get(model)
    if not prices:
        return 0.0
    return (
        prompt_tokens * prices["input"] / 1_000_000
        + completion_tokens * prices["output"] / 1_000_000
    )


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (p / 100) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return round(ordered[low] * (1 - frac) + ordered[high] * frac, 3)


def usage_summary(usages: list[dict], wall_s: float | None = None) -> dict:
    """Per-call latencies sum to more than wall time; the pool runs them concurrently."""
    latencies = [u["latency_s"] for u in usages if u.get("latency_s") is not None]
    return {
        "model": usages[0]["model"] if usages else None,
        "base_url": LLM_BASE_URL,
        "calls": len(usages),
        "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usages),
        "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usages),
        "cost_usd": round(sum(u.get("cost_usd") or 0 for u in usages), 6),
        "latency_mean_s": (
            round(sum(latencies) / len(latencies), 3) if latencies else None
        ),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "eval_wall_s": round(wall_s, 3) if wall_s is not None else None,
    }


@lru_cache(maxsize=1)
def get_llm_client() -> OpenAI:
    """Reuse one client. Ollama ignores the key but the SDK still requires one."""
    if not LLM_BASE_URL and not OPENAI_API_KEY:
        raise SystemExit(
            "OPENAI_API_KEY missing — set it in .env, or set LLM_BASE_URL "
            "for a local server (e.g. http://localhost:11434/v1)"
        )
    kwargs: dict = {"api_key": OPENAI_API_KEY or "ollama"}
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    return OpenAI(**kwargs)


def complete(
    prompt: str,
    model: str,
    temperature: float | None = None,
    json_mode: bool = False,
    client: OpenAI | None = None,
) -> CompletionResult:
    """Chat Completions — the surface Ollama, vLLM, and OpenAI all share."""
    openai_client = client or get_llm_client()
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    # Time only the API call, so retrieval and embedding stay out of the number.
    started = time.perf_counter()
    response = openai_client.chat.completions.create(**kwargs)
    latency_s = time.perf_counter() - started

    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError(f"{model} returned empty content")

    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    return CompletionResult(
        text=text,
        model=model,
        latency_s=latency_s,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=token_cost_usd(model, prompt_tokens, completion_tokens),
    )
