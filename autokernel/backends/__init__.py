"""Backends: one per (language, hardware family) pair.

Each backend knows how to describe the target, state the candidate contract, build a
candidate, and run the shared evaluation protocol in a subprocess. cuda_cpp uses the
C++ harness in harness/; triton (milestone 2) runs a Python subprocess with torch.
"""

from __future__ import annotations

from ..spec.problem import Problem
from .base import Backend, BackendError, BuildResult, EvalCase, EvalResult


def make_backend(problem: Problem) -> Backend:
    lang = problem.target.language
    if lang == "cuda_cpp":
        from .cuda_cpp import CudaCppBackend
        return CudaCppBackend(problem)
    if lang == "triton":
        raise NotImplementedError("the triton backend is milestone 2; see docs/ARCHITECTURE.md")
    raise ValueError(f"unknown target language {lang!r}")


__all__ = ["Backend", "BackendError", "BuildResult", "EvalCase", "EvalResult", "make_backend"]
