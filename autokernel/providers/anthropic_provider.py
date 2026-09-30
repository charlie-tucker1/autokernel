"""Anthropic Messages API provider. The key comes from ANTHROPIC_API_KEY (the CLI
loads .env first). The system prompt is marked cacheable since it repeats every step."""

from __future__ import annotations

import os
from typing import Optional

from .base import Provider, ProviderError, Reply, Usage


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str, temperature: Optional[float] = None,
                 thinking_budget: Optional[int] = None, base_url: Optional[str] = None):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderError("ANTHROPIC_API_KEY is not set. Put `ANTHROPIC_API_KEY=...` in .env at the repo root.")
        try:
            import anthropic
        except ImportError as e:
            raise ProviderError("the anthropic package is not installed in this environment") from e
        self.client = anthropic.Anthropic(base_url=base_url) if base_url else anthropic.Anthropic()
        self.model = model
        self.temperature = temperature
        self.thinking_budget = thinking_budget

    def complete(self, system: str, user: str, max_tokens: int) -> Reply:
        import anthropic

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
        }
        if self.thinking_budget:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
            kwargs["max_tokens"] = max(max_tokens, self.thinking_budget + 4000)
        elif self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            resp = self.client.messages.create(**kwargs)
        except anthropic.APIError as e:
            raise ProviderError(f"anthropic API error: {e}") from e
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        u = resp.usage
        usage = Usage(input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                      cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                      cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0)
        return Reply(text=text, usage=usage, model=resp.model, stop_reason=resp.stop_reason)
