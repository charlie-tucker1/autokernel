"""One interface for every model provider: complete(system, user) -> Reply.

Each search step is a single fresh call. All state the model needs is in the prompt,
which keeps cost visible and lets any provider, including a human, take part."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Optional


class ProviderError(RuntimeError):
    """The provider cannot continue (no key, out of canned replies, API failure).
    `retryable` marks transient failures (rate limit, overload, connection) that the
    search loop may retry after a backoff; everything else stops the run."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total"] = self.total
        return d


@dataclass
class Reply:
    text: str
    usage: Usage
    model: str
    stop_reason: Optional[str] = None
    thinking: str = ""                      # the model's thinking blocks, when the API returns them
    block_types: list[str] = field(default_factory=list)


class Provider(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int) -> Reply: ...
