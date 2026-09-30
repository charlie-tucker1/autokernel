"""Model providers behind one interface: complete(system, user, max_tokens) -> Reply.

anthropic, openai (any OpenAI-compatible endpoint), mock (canned replies for tests),
human (you paste the kernel). API keys come from the environment only.
"""

from __future__ import annotations

from pathlib import Path

from ..spec.problem import Problem
from .base import Provider, ProviderError, Reply, Usage


def make_provider(problem: Problem, run_dir: Path) -> Provider:
    m = problem.model
    if m.provider == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(m.model, temperature=m.temperature, thinking_budget=m.thinking_budget, base_url=m.base_url)
    if m.provider == "openai":
        from .openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(m.model, temperature=m.temperature, base_url=m.base_url)
    if m.provider == "mock":
        from .mock import MockProvider
        if not m.mock_dir:
            raise ProviderError("model.provider=mock needs model.mock_dir pointing at a directory of reply files")
        base = problem.spec_path.parent if problem.spec_path else Path.cwd()
        return MockProvider.from_dir((base / m.mock_dir) if not Path(m.mock_dir).is_absolute() else Path(m.mock_dir))
    if m.provider == "human":
        from .human import HumanProvider
        return HumanProvider(run_dir)
    raise ValueError(f"unknown provider {m.provider!r}")


__all__ = ["Provider", "ProviderError", "Reply", "Usage", "make_provider"]
