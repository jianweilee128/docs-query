"""OpenAI-compatible chat client. Point LLM_BASE_URL at Ollama (or any /v1 server)."""

from functools import lru_cache

from openai import OpenAI

from config.settings import LLM_BASE_URL, OPENAI_API_KEY


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
) -> str:
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
    response = openai_client.chat.completions.create(**kwargs)
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError(f"{model} returned empty content")
    return text
