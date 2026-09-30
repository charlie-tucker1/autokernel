"""Candidate (what the model produced) and Attempt (what happened to it)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field


@dataclass
class Candidate:
    hypothesis: str
    source: Optional[str]
    scratchpad: Optional[str]
    raw_reply: str
    problems: list[str] = field(default_factory=list)


class Attempt(BaseModel):
    """One row of the ledger. Everything measured about one candidate."""
    id: int
    parent_id: Optional[int] = None
    run_id: str = ""
    timestamp: str = ""
    provider: str = ""
    model: str = ""
    hypothesis: str = ""
    source_path: Optional[str] = None
    source_sha256: Optional[str] = None
    error: str = ""  # orchestrator-level failure: no code in reply, provider trouble

    build_ok: bool = False
    build_diagnostics: str = ""
    build_seconds: float = 0.0
    resources: dict = Field(default_factory=dict)

    run_ok: bool = False
    run_stage: str = ""
    run_error: str = ""
    run_seconds: float = 0.0

    verified: bool = False
    checks: list[dict] = Field(default_factory=list)
    timing_checks: list[dict] = Field(default_factory=list)
    timing: Optional[dict] = None
    trial_ms: list[float] = Field(default_factory=list)
    check_launch_ms: Optional[float] = None

    baseline_median_ms: Optional[float] = None
    speedup_vs_baseline: Optional[float] = None
    best_before_median_ms: Optional[float] = None
    speedup_vs_best: Optional[float] = None
    kept: bool = False

    usage: dict = Field(default_factory=dict)
    stop_reason: Optional[str] = None
    seeds: dict = Field(default_factory=dict)
    protocol: dict = Field(default_factory=dict)
    target_arch: Optional[str] = None
    seconds: dict = Field(default_factory=dict)

    @property
    def median_ms(self) -> Optional[float]:
        return self.timing["median_ms"] if self.timing else None
