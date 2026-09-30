import math

import torch

from autokernel.backends.protocol import TimingStats, compare_output
from autokernel.spec.problem import PrecisionSpec, ToleranceSpec

PREC = PrecisionSpec(tolerances=[ToleranceSpec(output="C", atol=1e-3, rtol=1e-4)])


def test_pass_within_tolerance():
    ref = torch.rand(8, 8)
    got = ref + 5e-4
    c = compare_output("C", got, ref, PREC)
    assert c.passed and c.n_bad == 0 and c.max_abs_err <= 5.1e-4


def test_fail_reports_worst_element():
    ref = torch.zeros(4, 4)
    got = ref.clone()
    got[2, 3] = 0.5
    c = compare_output("C", got, ref, PREC)
    assert not c.passed and c.n_bad == 1
    assert c.worst_index == [2, 3] and c.worst_got == 0.5
    assert "FAIL" in c.summary()


def test_nan_fails_and_is_counted():
    ref = torch.ones(3, 3)
    got = ref.clone()
    got[0, 0] = math.nan
    c = compare_output("C", got, ref, PREC)
    assert not c.passed and c.n_nan == 1 and c.n_bad == 1 and c.worst_index == [0, 0]


def test_shape_and_dtype_mismatch():
    ref = torch.ones(3, 3)
    assert "shape" in compare_output("C", torch.ones(3, 2), ref, PREC).note
    assert "dtype" in compare_output("C", torch.ones(3, 3, dtype=torch.float64), ref, PREC).note


def test_integer_exact():
    ref = torch.arange(6, dtype=torch.int32)
    assert compare_output("C", ref.clone(), ref, PREC).passed
    got = ref.clone()
    got[1] = 99
    assert not compare_output("C", got, ref, PREC).passed


def test_timing_stats():
    ts = TimingStats.from_trials([1.0, 1.2, 0.9, 1.1, 5.0])
    assert ts.n == 5 and ts.min_ms == 0.9 and ts.median_ms == 1.1 and ts.max_ms == 5.0
    assert "median 1.1000 ms" in ts.summary()
