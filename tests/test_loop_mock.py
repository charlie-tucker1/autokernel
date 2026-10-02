import json

import pytest

from autokernel.backends import make_backend
from autokernel.providers.mock import MockProvider
from autokernel.search.ledger import Ledger
from autokernel.search.prompt import feedback_text
from autokernel.search.run import run_search

pytestmark = pytest.mark.gpu


def wrap(source: str, hypothesis: str, notes: str = "- note") -> str:
    return f"<hypothesis>\n{hypothesis}\n</hypothesis>\n<scratchpad>\n# Notes\n{notes}\n</scratchpad>\n```cuda\n{source}\n```\n"


def test_greedy_loop_keeps_the_working_kernel(tiny_problem, naive_gemm_source, tmp_path):
    broken = naive_gemm_source.replace("acc += ", "acc +== ")
    provider = MockProvider([wrap(broken, "typo on purpose"), wrap(naive_gemm_source, "fixed", "- fixed the typo")])
    run_dir = tmp_path / "run"
    logs: list[str] = []
    summary = run_search(tiny_problem, run_dir, provider, make_backend(tiny_problem), log=logs.append)

    attempts = Ledger(run_dir / "ledger.jsonl").load()
    assert [a.id for a in attempts] == [1, 2]
    assert not attempts[0].build_ok and attempts[1].verified and attempts[1].kept
    assert attempts[1].parent_id == 1
    assert summary["best"]["id"] == 2 and summary["stopped_because"] == "max_iterations"
    assert (run_dir / "best.cu").read_text() == naive_gemm_source
    assert "fixed the typo" in (run_dir / "scratchpad.md").read_text()
    assert json.loads((run_dir / "baseline.json").read_text())["stats"]["median_ms"] > 0
    # The second prompt carried the compiler feedback from the first attempt.
    second_system, second_prompt = provider.calls[1]
    assert "Feedback on attempt #1" in second_prompt and "Build: FAILED" in second_prompt
    assert "# Candidate contract" in second_system and "# Candidate contract" not in second_prompt
    assert second_system == provider.calls[0][0]  # the system prompt is identical every step, so it caches
    assert (run_dir / "prompts" / "002_reply.md").is_file()
    assert not list(run_dir.rglob("*.so"))  # compiled libraries do not pile up


def test_resume_continues_numbering(tiny_problem, naive_gemm_source, tmp_path):
    run_dir = tmp_path / "run"
    tiny_problem.budget.max_iterations = 1
    run_search(tiny_problem, run_dir, MockProvider([wrap(naive_gemm_source, "first")]), make_backend(tiny_problem), log=lambda s: None)
    tiny_problem.budget.max_iterations = 2
    run_search(tiny_problem, run_dir, MockProvider([wrap(naive_gemm_source, "second")]), make_backend(tiny_problem), log=lambda s: None)
    attempts = Ledger(run_dir / "ledger.jsonl").load()
    assert [a.id for a in attempts] == [1, 2]
    assert attempts[1].verified and attempts[1].parent_id == 1
    # The second run resubmitted the same source: copied, not rebuilt or re-run.
    assert attempts[1].duplicate_of == 1 and not attempts[1].kept
    assert attempts[1].timing == attempts[0].timing and attempts[1].seeds == attempts[0].seeds
    assert not (run_dir / "attempts" / "002").exists()
    assert "identical to attempt #1" in feedback_text(attempts[1], tiny_problem)


def test_close_win_is_confirmed_back_to_back(tiny_problem, naive_gemm_source, tmp_path):
    """Two kernels that differ only by a comment time the same, so if the second one
    'wins' it wins by less than the margin and must be re-run against the first."""
    twin = "// same kernel, different bytes\n" + naive_gemm_source
    tiny_problem.protocol.confirm_margin = 0.5  # microsecond kernels are noisy; make any win a close win
    provider = MockProvider([wrap(naive_gemm_source, "first"), wrap(twin, "twin")])
    run_dir = tmp_path / "run"
    run_search(tiny_problem, run_dir, provider, make_backend(tiny_problem), log=lambda s: None)
    a1, a2 = Ledger(run_dir / "ledger.jsonl").load()
    assert a1.verified and a2.verified and a2.duplicate_of is None
    if a2.median_ms >= a1.median_ms:
        assert not a2.kept and a2.confirmation is None
    else:
        c = a2.confirmation
        assert c is not None and c["incumbent_verified"] and c["challenger_verified"]
        assert c["kept"] == (c["challenger_median_ms"] < c["incumbent_median_ms"]) == a2.kept
        assert (run_dir / "attempts" / "002" / "confirm" / "incumbent" / "build.log").is_file()
        assert (run_dir / "best.cu").read_text() == (twin if a2.kept else naive_gemm_source)
        assert "re-run back to back" in feedback_text(a2, tiny_problem)
        assert not list(run_dir.rglob("*.so"))
