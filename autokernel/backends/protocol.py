"""The evaluation protocol pieces every backend shares: moving tensors through .npy
files, comparing outputs against the reference, timing statistics, and timing a torch
callable the same way the harness times a candidate. See docs/ARCHITECTURE.md,
"Evaluation protocol"."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..spec.problem import PrecisionSpec, TensorSpec
from ..spec.reference import TORCH_DTYPES


# ------------------------------------------------------------------ tensors <-> files

def save_npy(path: Path, t: torch.Tensor) -> None:
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        arr = t.view(torch.int16).numpy().view(np.uint16)
    else:
        arr = t.numpy()
    np.save(path, arr)


def load_npy(path: Path, dtype: str, device: torch.device) -> torch.Tensor:
    arr = np.load(path)
    if dtype == "bfloat16":
        return torch.from_numpy(arr.view(np.int16).copy()).view(torch.bfloat16).to(device)
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device)


def nan_filled(spec: TensorSpec) -> torch.Tensor:
    dt = TORCH_DTYPES[spec.dtype]
    if dt.is_floating_point:
        return torch.full(tuple(spec.shape), float("nan"), dtype=dt)
    return torch.zeros(tuple(spec.shape), dtype=dt)


# ------------------------------------------------------------------ correctness

@dataclass
class OutputCheck:
    name: str
    passed: bool
    atol: float
    rtol: float
    n_total: int = 0
    n_bad: int = 0
    n_nan: int = 0
    max_abs_err: float = 0.0
    max_tol_ratio: float = 0.0  # worst |err| / (atol + rtol*|ref|); < 1 passes
    worst_index: list[int] = field(default_factory=list)
    worst_got: float = 0.0
    worst_ref: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if self.note:
            return f"{self.name}: FAIL ({self.note})"
        if self.passed:
            return (f"{self.name}: PASS  max_abs_err {self.max_abs_err:.3g}; worst element used "
                    f"{100 * self.max_tol_ratio:.1f}% of its tolerance; {self.n_nan} NaN")
        frac = 100.0 * self.n_bad / max(self.n_total, 1)
        return (f"{self.name}: FAIL  {self.n_bad}/{self.n_total} elements ({frac:.2f}%) outside "
                f"atol={self.atol:g} + rtol={self.rtol:g}*|ref|; {self.n_nan} NaN; worst at "
                f"{self.worst_index}: got {self.worst_got:.6g}, ref {self.worst_ref:.6g}, "
                f"max_abs_err {self.max_abs_err:.3g}")


def compare_output(name: str, got: torch.Tensor, ref: torch.Tensor, precision: PrecisionSpec) -> OutputCheck:
    tol = precision.tolerance_for(name)
    chk = OutputCheck(name=name, passed=False, atol=tol.atol, rtol=tol.rtol, n_total=ref.numel())
    if tuple(got.shape) != tuple(ref.shape):
        chk.note = f"shape {tuple(got.shape)} != {tuple(ref.shape)}"
        return chk
    if got.dtype != ref.dtype:
        chk.note = f"dtype {got.dtype} != {ref.dtype}"
        return chk
    if not ref.is_floating_point():
        bad = got != ref
        chk.n_bad = int(bad.sum())
        chk.passed = chk.n_bad == 0
        if not chk.passed:
            idx = int(torch.argmax(bad.flatten().to(torch.int8)))
            chk.worst_index = list(np.unravel_index(idx, ref.shape))
            chk.worst_got = float(got.flatten()[idx])
            chk.worst_ref = float(ref.flatten()[idx])
        return chk

    g = got.to(torch.float64)
    r = ref.to(torch.float64)
    err = (g - r).abs()
    nan_mask = torch.isnan(g) | torch.isinf(g)
    chk.n_nan = int(nan_mask.sum())
    allowed = tol.atol + tol.rtol * r.abs()
    bad = ~(err <= allowed)  # NaN compares false, so NaN/Inf count as bad
    chk.n_bad = int(bad.sum())
    chk.passed = chk.n_bad == 0
    finite = ~nan_mask
    if bool(finite.any()):
        chk.max_abs_err = float(err[finite].max())
        chk.max_tol_ratio = float((err[finite] / allowed[finite].clamp_min(1e-30)).max())
    if chk.n_nan:
        chk.max_tol_ratio = math.inf
    score = torch.where(nan_mask, torch.full_like(err, math.inf), err)
    idx = int(torch.argmax(score.flatten()))
    chk.worst_index = [int(i) for i in np.unravel_index(idx, ref.shape)]
    chk.worst_got = float(g.flatten()[idx])
    chk.worst_ref = float(r.flatten()[idx])
    return chk


def compare_outputs(got: dict[str, torch.Tensor], refs: dict[str, torch.Tensor], precision: PrecisionSpec) -> list[OutputCheck]:
    return [compare_output(name, got[name], ref, precision) for name, ref in refs.items()]


# ------------------------------------------------------------------ timing

@dataclass
class TimingStats:
    n: int
    median_ms: float
    min_ms: float
    max_ms: float
    p90_ms: float
    mean_ms: float
    std_ms: float
    spread: float  # (p90 - min) / median

    @classmethod
    def from_trials(cls, trials: list[float]) -> "TimingStats":
        a = np.asarray(trials, dtype=np.float64)
        med = float(np.median(a))
        p90 = float(np.percentile(a, 90))
        return cls(n=int(a.size), median_ms=med, min_ms=float(a.min()), max_ms=float(a.max()),
                   p90_ms=p90, mean_ms=float(a.mean()), std_ms=float(a.std()),
                   spread=float((p90 - a.min()) / med) if med > 0 else 0.0)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (f"median {self.median_ms:.4f} ms, min {self.min_ms:.4f}, p90 {self.p90_ms:.4f}, "
                f"spread {100 * self.spread:.1f}% over {self.n} trials")


def time_torch(fn, args: list, warmup: int, trials: int, flush_bytes: int, device: torch.device) -> list[float]:
    """Time `fn(*args)` the way the harness times a candidate: warmup, then one call per
    trial between CUDA events, with an optional L2 flush before each trial."""
    flush = torch.empty(int(flush_bytes), dtype=torch.uint8, device=device) if flush_bytes else None
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize(device)
        times: list[float] = []
        for _ in range(trials):
            if flush is not None:
                flush.zero_()
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(*args)
            stop.record()
            stop.synchronize()
            times.append(start.elapsed_time(stop))
    return times
