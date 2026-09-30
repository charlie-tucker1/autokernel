"""The Problem model: everything a run needs, validated once at load time.

Mirrors the `Frontend` section of docs/ARCHITECTURE.md. Field defaults are the
built-in preset; a spec file overrides them; CLI flags override the spec."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

DType = Literal["float32", "float64", "float16", "bfloat16", "int32", "int64", "uint8", "int8", "bool"]


class TensorSpec(BaseModel):
    name: str
    shape: list[int]
    dtype: DType = "float32"
    init: Literal["uniform", "normal", "zeros", "ones", "randint"] = "uniform"
    low: float = -1.0
    high: float = 1.0
    mean: float = 0.0
    std: float = 1.0

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n


class ComputationSpec(BaseModel):
    name: str
    description: str = ""
    reference: str = Field(description="`path/to/file.py:function` relative to the spec, or `package.module:function`")
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    flops: Optional[float] = Field(default=None, description="floating-point operations per call, for throughput reporting")
    min_bytes: Optional[float] = Field(default=None, description="compulsory bytes moved per call, for roofline reporting")


class ToleranceSpec(BaseModel):
    output: str = "*"
    atol: float = 1e-5
    rtol: float = 1e-5


class PrecisionSpec(BaseModel):
    tolerances: list[ToleranceSpec] = Field(default_factory=lambda: [ToleranceSpec()])
    allowed_reduced_precision: list[str] = Field(default_factory=list, description="e.g. ['tf32', 'bf16']")
    notes: str = ""

    def tolerance_for(self, output: str) -> ToleranceSpec:
        for t in self.tolerances:
            if t.output == output:
                return t
        for t in self.tolerances:
            if t.output == "*":
                return t
        return ToleranceSpec()


class TargetSpec(BaseModel):
    language: Literal["cuda_cpp", "triton"] = "cuda_cpp"
    device: int = 0
    arch: Optional[str] = Field(default=None, description="e.g. sm_120; detected from the device when unset")
    harness: Optional[str] = Field(default=None, description="path to the built harness executable; auto-located when unset")


class GoalsSpec(BaseModel):
    objective: Literal["latency", "throughput"] = "latency"
    max_registers: Optional[int] = None
    max_smem_bytes: Optional[int] = None
    notes: str = ""


class BudgetSpec(BaseModel):
    max_iterations: int = 10
    max_tokens: Optional[int] = Field(default=None, description="total input+output tokens across the run")
    max_wall_seconds: Optional[float] = None


class ProtocolSpec(BaseModel):
    warmup: int = 3
    trials: int = 20
    flush_l2: bool = True
    compile_timeout_s: float = 300.0
    run_timeout_s: float = 120.0
    seed: int = 0
    keep_tensors: bool = False


class FeedbackSpec(BaseModel):
    tier: int = 1
    max_diag_lines: int = 40
    recent_attempts: int = 5


class ModelSpec(BaseModel):
    provider: Literal["anthropic", "openai", "mock", "human"] = "anthropic"
    model: str = "claude-sonnet-5"
    max_output_tokens: int = 16000
    temperature: Optional[float] = None
    thinking_budget: Optional[int] = None
    base_url: Optional[str] = None
    mock_dir: Optional[str] = None


class Problem(BaseModel):
    computation: ComputationSpec
    precision: PrecisionSpec = Field(default_factory=PrecisionSpec)
    target: TargetSpec = Field(default_factory=TargetSpec)
    goals: GoalsSpec = Field(default_factory=GoalsSpec)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    protocol: ProtocolSpec = Field(default_factory=ProtocolSpec)
    feedback: FeedbackSpec = Field(default_factory=FeedbackSpec)
    model: ModelSpec = Field(default_factory=ModelSpec)
    spec_path: Optional[Path] = None

    @model_validator(mode="after")
    def _check_names(self) -> "Problem":
        names = [t.name for t in self.computation.inputs + self.computation.outputs]
        if len(set(names)) != len(names):
            raise ValueError(f"tensor names must be unique, got {names}")
        for tol in self.precision.tolerances:
            if tol.output != "*" and tol.output not in [t.name for t in self.computation.outputs]:
                raise ValueError(f"tolerance refers to unknown output {tol.output!r}")
        return self
