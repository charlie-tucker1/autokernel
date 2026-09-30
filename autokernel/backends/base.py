"""Backend interface: one implementation per (language, hardware family) pair.

A backend turns candidate source into something runnable, runs the shared evaluation
protocol on it in a subprocess, and reports. It never talks to a model provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import torch

from ..spec.problem import Problem
from .protocol import OutputCheck, TimingStats


class BackendError(RuntimeError):
    """The backend cannot start or describe its target (missing tool, bad device)."""


@dataclass
class BuildResult:
    ok: bool
    diagnostics: str
    artifact: Optional[Path]
    resources: dict = field(default_factory=dict)
    seconds: float = 0.0
    command: list[str] = field(default_factory=list)


@dataclass
class EvalCase:
    """One evaluation: a correctness input set and a separate timing input set, with
    the reference outputs for both."""
    check_inputs: dict[str, torch.Tensor]
    check_refs: dict[str, torch.Tensor]
    time_inputs: dict[str, torch.Tensor]
    time_refs: dict[str, torch.Tensor]


@dataclass
class EvalResult:
    ok: bool                      # the candidate ran to completion without runtime errors
    stage: str = ""
    error: str = ""
    checks: list[OutputCheck] = field(default_factory=list)          # correctness set
    timing_checks: list[OutputCheck] = field(default_factory=list)   # timing set
    trial_ms: list[float] = field(default_factory=list)
    timing: Optional[TimingStats] = None
    check_launch_ms: Optional[float] = None
    raw: dict = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def verified(self) -> bool:
        return self.ok and bool(self.checks) and all(c.passed for c in self.checks) \
            and all(c.passed for c in self.timing_checks)


class Backend(Protocol):
    name: str
    source_extension: str
    code_fence: str

    def describe_target(self) -> dict: ...
    def target_text(self) -> str: ...
    def contract_text(self) -> str: ...
    def measurement_text(self, problem: Problem) -> str: ...
    def build(self, source: str, workdir: Path) -> BuildResult: ...
    def run(self, build: BuildResult, case: EvalCase, workdir: Path) -> EvalResult: ...
