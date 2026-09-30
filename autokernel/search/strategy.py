"""Search strategies. Greedy keep-if-better is milestone 1; idea-branching and
population strategies come later behind the same interface."""

from __future__ import annotations

from typing import Optional, Protocol

from .ledger import Ledger
from .types import Attempt


class Strategy(Protocol):
    name: str

    def parent(self, attempts: list[Attempt]) -> Optional[Attempt]:
        """The attempt the next candidate should build on."""


class Greedy:
    name = "greedy"

    def parent(self, attempts: list[Attempt]) -> Optional[Attempt]:
        best = Ledger.best(attempts)
        if best is not None:
            return best
        return attempts[-1] if attempts else None


def make_strategy(name: str = "greedy") -> Strategy:
    if name == "greedy":
        return Greedy()
    raise ValueError(f"unknown strategy {name!r}")
