"""Canned replies, in order. For tests and for dry runs that cost nothing."""

from __future__ import annotations

from pathlib import Path

from .base import Provider, ProviderError, Reply, Usage


class MockProvider(Provider):
    name = "mock"

    def __init__(self, replies: list[str], model: str = "mock"):
        self.replies = list(replies)
        self.model = model
        self.calls: list[tuple[str, str]] = []

    @classmethod
    def from_dir(cls, path: str | Path) -> "MockProvider":
        files = sorted(Path(path).glob("*.md")) + sorted(Path(path).glob("*.txt"))
        if not files:
            raise ProviderError(f"mock provider: no .md or .txt replies in {path}")
        return cls([f.read_text() for f in files], model=f"mock:{Path(path).name}")

    def complete(self, system: str, user: str, max_tokens: int) -> Reply:
        self.calls.append((system, user))
        if not self.replies:
            raise ProviderError("mock provider has no more replies")
        text = self.replies.pop(0)
        usage = Usage(input_tokens=(len(system) + len(user)) // 4, output_tokens=len(text) // 4)
        return Reply(text=text, usage=usage, model=self.model, stop_reason="end_turn")
