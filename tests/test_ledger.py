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


def test_digest_is_bounded_and_best_respects_confirmation():
    attempts = []
    for i in range(1, 61):
        if i % 10 == 0:
            attempts.append(Attempt(id=i, build_ok=False, build_diagnostics="error: x", hypothesis=f"h{i}"))
        elif i % 7 == 0:
            attempts.append(Attempt(id=i, error="no code block found in the reply", hypothesis=f"h{i}"))
        else:
            attempts.append(_verified(i, 2.0 - i / 100, kept=True))
    d = Ledger.digest(attempts, recent=5, older_window=20)
    assert "55 earlier attempts (#1 to #55): " in d and "verified" in d and "build failures" in d
    assert "The last 20 of them: #36 " in d and "#3 verified" not in d
    assert "Improvements so far: ... -> " in d and d.count(" -> ") == 12
    assert len(d) < 6000
    assert "- #56" in d and "- #60" in d

    # Many more attempts barely grow the digest.
    more = attempts + [_verified(i, 1.0) for i in range(61, 1061)]
    assert len(Ledger.digest(more, recent=5, older_window=20)) < len(d) + 1500

    # A close win that failed its confirmation is not the best, and a duplicate never displaces its original.
    base = [_verified(1, 0.50, kept=True)]
    loser = _verified(2, 0.49)
    loser.confirmation = {"incumbent_median_ms": 0.48, "challenger_median_ms": 0.50, "kept": False}
    dup = _verified(3, 0.50)
    dup.duplicate_of = 1
    assert Ledger.best(base + [loser, dup]).id == 1
    assert "duplicate of #1, not re-run" in outcome_line(dup)
