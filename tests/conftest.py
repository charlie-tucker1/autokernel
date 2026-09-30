import pytest

from autokernel.env import REPO_ROOT
from autokernel.spec.problem import (BudgetSpec, ComputationSpec, ModelSpec, PrecisionSpec, Problem,
                                     ProtocolSpec, TensorSpec, ToleranceSpec)

FIXTURE_DIR = REPO_ROOT / "tests" / "kernels"


def _gpu_ready() -> bool:
    try:
        import torch
        from autokernel.backends.cuda_cpp import find_harness, find_nvcc
        return torch.cuda.is_available() and find_harness() is not None and find_nvcc() is not None
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _gpu_ready():
        return
    skip = pytest.mark.skip(reason="needs a CUDA device, nvcc and a built harness")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def tiny_problem() -> Problem:
    """A small GEMM that evaluates in well under a second."""
    p = Problem(
        computation=ComputationSpec(
            name="gemm_tiny", reference="reference/gemm.py:gemm",
            inputs=[TensorSpec(name="A", shape=[128, 64]), TensorSpec(name="B", shape=[64, 96])],
            outputs=[TensorSpec(name="C", shape=[128, 96])], flops=2 * 128 * 64 * 96),
        precision=PrecisionSpec(tolerances=[ToleranceSpec(output="C", atol=1e-3, rtol=1e-4)]),
        protocol=ProtocolSpec(warmup=1, trials=5, run_timeout_s=60),
        budget=BudgetSpec(max_iterations=2),
        model=ModelSpec(provider="mock"),
    )
    p.spec_path = REPO_ROOT / "specs" / "gemm_fp32.yaml"
    return p


@pytest.fixture
def naive_gemm_source() -> str:
    return (FIXTURE_DIR / "naive_gemm.cu").read_text()
