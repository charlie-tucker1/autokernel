from autokernel.search.ledger import Ledger, outcome_line
from autokernel.search.types import Attempt


def _verified(i, med, kept=False, base=1.0):
    return Attempt(id=i, build_ok=True, run_ok=True, verified=True, hypothesis=f"h{i}", kept=kept,
                   timing={"n": 5, "median_ms": med, "min_ms": med, "max_ms": med, "p90_ms": med,
                           "mean_ms": med, "std_ms": 0.0, "spread": 0.0},
                   speedup_vs_baseline=base / med)


def test_append_load_best_digest(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    a1 = Attempt(id=1, build_ok=False, build_diagnostics="candidate.cu(3): error: bad\n", hypothesis="h1")
    a2 = _verified(2, 0.5, kept=True)
    a3 = Attempt(id=3, build_ok=True, run_ok=True, verified=False, hypothesis="h3",
                 checks=[{"name": "C", "passed": False, "n_bad": 10, "n_total": 100, "n_nan": 0}])
    a4 = _verified(4, 0.7)
    for a in (a1, a2, a3, a4):
        ledger.append(a)
    loaded = ledger.load()
    assert [a.id for a in loaded] == [1, 2, 3, 4]
    assert Ledger.best(loaded).id == 2
    assert outcome_line(a1).startswith("build failed: candidate.cu(3): error: bad")
    assert "10.0% of elements off" in outcome_line(a3)
    assert "new best" in outcome_line(a2)
    d = Ledger.digest(loaded, recent=2)
    assert "Best so far: attempt #2" in d
    assert "Earlier attempts: #1" in d and "#2" in d
    assert "- #3" in d and "- #4" in d


def test_empty_ledger(tmp_path):
    ledger = Ledger(tmp_path / "none.jsonl")
    assert ledger.load() == []
    assert Ledger.best([]) is None
    assert Ledger.digest([]) == ""
