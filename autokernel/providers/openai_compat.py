"""OpenAI-compatible chat completions (OpenAI, local vLLM/Ollama endpoints, others).
Milestone 6 exercises this; it is here so the provider seam exists from the start.
Key from OPENAI_API_KEY; endpoint from model.base_url or OPENAI_BASE_URL."""

from __future__ import annotations

import os
from typing import Optional

from .base import Provider, ProviderError, Reply, Usage


class OpenAICompatProvider(Provider):
    name = "openai"

    def __init__(self, model: str, temperature: Optional[float] = None, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ProviderError("pip install openai to use the openai provider") from e
        key = os.environ.get("OPENAI_API_KEY") or ("local" if base_url or os.environ.get("OPENAI_BASE_URL") else None)
        if not key:
            raise ProviderError("OPENAI_API_KEY is not set (or give model.base_url for a local endpoint)")
        self.client = OpenAI(api_key=key, base_url=base_url or os.environ.get("OPENAI_BASE_URL"))
        self.model = model
        self.temperature = temperature

    def complete(self, system: str, user: str, max_tokens: int) -> Reply:
        kwargs: dict = {"model": self.model, "max_completion_tokens": max_tokens,
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:  # the SDK's error hierarchy varies by version
            raise ProviderError(f"openai-compatible API error: {e}") from e
        choice = resp.choices[0]
        u = resp.usage
        usage = Usage(input_tokens=getattr(u, "prompt_tokens", 0) or 0, output_tokens=getattr(u, "completion_tokens", 0) or 0)
        return Reply(text=choice.message.content or "", usage=usage, model=resp.model, stop_reason=choice.finish_reason)
