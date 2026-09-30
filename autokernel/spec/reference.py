"""The reference implementation: loading it, generating inputs, computing outputs.

The reference is PyTorch code named in the spec. It defines "correct". It runs only in
the orchestrator process, so a candidate never has access to its outputs."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Callable

import torch

from .problem import Problem, TensorSpec

TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "bool": torch.bool,
}


def load_reference(problem: Problem) -> Callable[..., object]:
    target, _, func = problem.computation.reference.rpartition(":")
    if not func:
        raise ValueError("reference must be 'file.py:function' or 'module:function'")
    if target.endswith(".py"):
        base = problem.spec_path.parent if problem.spec_path else Path.cwd()
        file = (base / target).resolve()
        spec = importlib.util.spec_from_file_location(f"akref_{file.stem}", file)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    fn = getattr(module, func)
    if not callable(fn):
        raise TypeError(f"{problem.computation.reference} is not callable")
    return fn


def reference_source(fn: Callable[..., object]) -> str:
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return f"# source unavailable for {fn!r}"


def make_tensor(spec: TensorSpec, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    dt = TORCH_DTYPES[spec.dtype]
    shape = tuple(spec.shape)
    if spec.init == "zeros":
        return torch.zeros(shape, dtype=dt, device=device)
    if spec.init == "ones":
        return torch.ones(shape, dtype=dt, device=device)
    if spec.init == "randint" or not dt.is_floating_point:
        lo, hi = int(spec.low), int(spec.high)
        if dt == torch.bool:
            return torch.randint(0, 2, shape, generator=gen, device=device).bool()
        return torch.randint(lo, max(hi, lo + 1), shape, generator=gen, device=device, dtype=dt)
    if spec.init == "normal":
        x = torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
        return (x * spec.std + spec.mean).to(dt)
    x = torch.rand(shape, generator=gen, device=device, dtype=torch.float32)
    return (spec.low + (spec.high - spec.low) * x).to(dt)


def generate_inputs(problem: Problem, seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return {t.name: make_tensor(t, gen, device) for t in problem.computation.inputs}


def run_reference(problem: Problem, fn: Callable[..., object], inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    args = [inputs[t.name] for t in problem.computation.inputs]
    with torch.no_grad():
        result = fn(*args)
    outputs = problem.computation.outputs
    if isinstance(result, torch.Tensor):
        result = (result,)
    if len(result) != len(outputs):
        raise ValueError(f"reference returned {len(result)} tensors, spec lists {len(outputs)} outputs")
    refs: dict[str, torch.Tensor] = {}
    for spec, tensor in zip(outputs, result):
        if tuple(tensor.shape) != tuple(spec.shape):
            raise ValueError(f"reference output {spec.name} has shape {tuple(tensor.shape)}, spec says {spec.shape}")
        refs[spec.name] = tensor.contiguous()
    return refs
