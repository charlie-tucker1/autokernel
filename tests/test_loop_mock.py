import json

import pytest

from autokernel.backends import make_backend
from autokernel.providers.mock import MockProvider
from autokernel.search.ledger import Ledger
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
    second_prompt = provider.calls[1][1]
    assert "Feedback on attempt #1" in second_prompt and "Build: FAILED" in second_prompt
    assert (run_dir / "prompts" / "002_reply.md").is_file()


def test_resume_continues_numbering(tiny_problem, naive_gemm_source, tmp_path):
    run_dir = tmp_path / "run"
    tiny_problem.budget.max_iterations = 1
    run_search(tiny_problem, run_dir, MockProvider([wrap(naive_gemm_source, "first")]), make_backend(tiny_problem), log=lambda s: None)
    tiny_problem.budget.max_iterations = 2
    run_search(tiny_problem, run_dir, MockProvider([wrap(naive_gemm_source, "second")]), make_backend(tiny_problem), log=lambda s: None)
    attempts = Ledger(run_dir / "ledger.jsonl").load()
    assert [a.id for a in attempts] == [1, 2]
    assert attempts[1].verified and attempts[1].parent_id == 1
