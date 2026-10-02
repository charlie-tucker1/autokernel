"""Anthropic Messages API provider. The key comes from ANTHROPIC_API_KEY (the CLI
loads .env first). The system prompt is marked cacheable since it repeats every step."""

from __future__ import annotations

import os
from typing import Optional

from .base import Provider, ProviderError, Reply, Usage

# Worth waiting for: rate limit, overloaded, server trouble. 4xx otherwise means the request is wrong.
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


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
            # Streaming is required by the SDK for large max_tokens; we only need the final message.
            with self.client.messages.stream(**kwargs) as stream:
                resp = stream.get_final_message()
        except anthropic.APIStatusError as e:
            transient = e.status_code in RETRYABLE_STATUS
            raise ProviderError(f"anthropic API error {e.status_code}: {e.message}", retryable=transient) from e
        except anthropic.APIConnectionError as e:  # includes timeouts and dropped streams
            raise ProviderError(f"anthropic connection error: {e}", retryable=True) from e
        except anthropic.APIError as e:
            raise ProviderError(f"anthropic API error: {e}") from e
        text, thinking, kinds = [], [], []
        for b in resp.content:
            kind = getattr(b, "type", "")
            kinds.append(kind)
            if kind == "text":
                text.append(b.text)
            elif kind == "thinking":
                thinking.append(getattr(b, "thinking", "") or "")
            elif kind == "redacted_thinking":
                thinking.append("[redacted thinking block]")
        u = resp.usage
        usage = Usage(input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                      cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                      cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0)
        return Reply(text="".join(text), usage=usage, model=resp.model, stop_reason=resp.stop_reason,
                     thinking="\n\n".join(t for t in thinking if t), block_types=kinds)
