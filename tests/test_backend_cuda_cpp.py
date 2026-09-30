import pytest

from autokernel.backends import make_backend
from autokernel.search.run import evaluate_candidate
from autokernel.search.types import Attempt
from autokernel.spec.reference import load_reference

pytestmark = pytest.mark.gpu

NOOP = '''#include "ak_kernel.h"
extern "C" int ak_kernel(const ak_tensor* in, int32_t n_in, ak_tensor* out, int32_t n_out, cudaStream_t s) { return 0; }
'''
RETURNS_7 = NOOP.replace("return 0;", "return 7;")
BROKEN = '#include "ak_kernel.h"\nextern "C" int ak_kernel( {\n'


def _evaluate(problem, source, tmp_path, i=1):
    backend = make_backend(problem)
    ref_fn = load_reference(problem)
    return evaluate_candidate(problem, backend, ref_fn, source, Attempt(id=i, source_path="x.cu"), tmp_path / "w")


def test_naive_gemm_is_verified(tiny_problem, naive_gemm_source, tmp_path):
    a = _evaluate(tiny_problem, naive_gemm_source, tmp_path)
    assert a.build_ok, a.build_diagnostics
    assert a.run_ok, a.run_error
    assert a.verified
    assert a.timing["median_ms"] > 0 and len(a.trial_ms) == 5
    assert a.resources["kernels"][0]["registers"] > 0
    assert all(c["passed"] for c in a.checks + a.timing_checks)
    assert a.seeds["check"] != a.seeds["time"]


def test_broken_source_fails_build_with_diagnostics(tiny_problem, tmp_path):
    a = _evaluate(tiny_problem, BROKEN, tmp_path)
    assert not a.build_ok and "error" in a.build_diagnostics.lower()
    assert not a.verified


def test_kernel_that_writes_nothing_fails_verification(tiny_problem, tmp_path):
    a = _evaluate(tiny_problem, NOOP, tmp_path)
    assert a.build_ok and a.run_ok
    assert not a.verified
    assert a.checks[0]["n_nan"] == 128 * 96


def test_nonzero_return_is_a_runtime_failure(tiny_problem, tmp_path):
    a = _evaluate(tiny_problem, RETURNS_7, tmp_path)
    assert a.build_ok and not a.run_ok
    assert a.run_stage == "check" and "returned 7" in a.run_error


def test_target_description(tiny_problem):
    backend = make_backend(tiny_problem)
    t = backend.describe_target()
    assert t["arch"].startswith("sm_") and t["sm_count"] > 0
    assert "SMs" in backend.target_text()
    assert "ak_kernel.h" in backend.contract_text()
