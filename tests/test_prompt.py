from autokernel.search.ledger import outcome_line
from autokernel.search.prompt import PromptContext, build_system_prompt, build_user_prompt, diagnostics_excerpt, feedback_text, parse_reply
from autokernel.search.types import Attempt

GOOD = """<hypothesis>
Naive kernel first.
</hypothesis>

<scratchpad>
# Notes
- tried nothing yet
</scratchpad>

```cuda
#include "ak_kernel.h"
extern "C" int ak_kernel(const ak_tensor* in, int32_t n_in, ak_tensor* out, int32_t n_out, cudaStream_t s) { return 0; }
```
"""


def test_parse_good_reply():
    c = parse_reply(GOOD, "cuda")
    assert c.hypothesis == "Naive kernel first."
    assert c.scratchpad is not None and "tried nothing" in c.scratchpad
    assert c.source is not None and c.source.startswith('#include "ak_kernel.h"')
    assert c.problems == []


def test_scratchpad_code_blocks_are_not_the_candidate():
    text = GOOD.replace("- tried nothing yet", "- snippet:\n```cuda\nint x;\n```")
    c = parse_reply(text, "cuda")
    assert "ak_kernel" in c.source and "int x;" not in c.source


def test_missing_sections_and_untagged_block():
    c = parse_reply("Here you go:\n```\nint main() {}\n```", "cuda")
    assert c.source == "int main() {}\n"
    assert any("hypothesis" in p for p in c.problems)
    assert any("language tag" in p for p in c.problems)


def test_truncated_reply():
    c = parse_reply("<hypothesis>x</hypothesis>\n```cuda\nint a;", "cuda")
    assert c.source is None
    assert any("unterminated" in p for p in c.problems)


def test_no_code():
    c = parse_reply("I cannot do this.", "cuda")
    assert c.source is None and c.hypothesis == "I cannot do this."


def test_diagnostics_excerpt_drops_noise():
    diag = "ptxas info    : Used 3 registers\n/tmp/x/candidate.cu(3): error: expected a ';'\nptxas info    : Compile time = 1 ms\n"
    out = diagnostics_excerpt(diag, 10)
    assert out == "candidate.cu(3): error: expected a ';'"
    assert "more lines" in diagnostics_excerpt("\n".join(f"line {i}" for i in range(50)), 5)


def _ctx(tiny_problem, **kw):
    base = dict(problem=tiny_problem, fence="cuda", reference_source="def gemm(A, B): return A @ B",
                target_text="- GPU: Fake", contract_text="ABI goes here", measurement_text="- measured somehow",
                baseline=None, best=None, best_source=None, latest=None, latest_source=None,
                scratchpad="my notes", digest="", feedback="", next_id=1)
    base.update(kw)
    return PromptContext(**base)


def test_static_sections_live_in_the_system_prompt(tiny_problem):
    ctx = _ctx(tiny_problem)
    system, user = build_system_prompt(ctx), build_user_prompt(ctx)
    for fixed in ("# Computation: gemm_tiny", "ABI goes here", "- GPU: Fake", "# Precision requirement", "- measured somehow"):
        assert fixed in system and fixed not in user
    for moving in ("# Your notes", "my notes", "# Attempt history", "Produce attempt #1"):
        assert moving in user and moving not in system
    assert "```cuda" in system and "{fence}" not in system
    assert "80 lines or 8,000 characters" in system
    assert "re-run back to back" in system  # confirm_margin is on by default


def test_feedback_mentions_duplicates_truncation_and_confirmation(tiny_problem):
    timing = {"n": 5, "median_ms": 0.5, "min_ms": 0.5, "max_ms": 0.5, "p90_ms": 0.5, "mean_ms": 0.5, "std_ms": 0.0, "spread": 0.0}
    check = {"name": "C", "passed": True, "atol": 1e-3, "rtol": 1e-4, "n_total": 10, "n_bad": 0, "n_nan": 0,
             "max_abs_err": 0.0, "max_tol_ratio": 0.0, "worst_index": [0], "worst_got": 0.0, "worst_ref": 0.0, "note": ""}
    a = Attempt(id=3, source_path="candidates/003.cu", duplicate_of=1, build_ok=True, run_ok=True, verified=True,
                checks=[check], timing_checks=[check], timing=timing, scratchpad_truncated=True,
                best_before_median_ms=0.4, speedup_vs_best=0.8)
    text = feedback_text(a, tiny_problem)
    assert "identical to attempt #1" in text and "not run again" in text
    assert "went past the limit (80 lines, 8,000 characters)" in text
    assert outcome_line(a).startswith("duplicate of #1, not re-run: verified")

    b = Attempt(id=4, source_path="candidates/004.cu", build_ok=True, run_ok=True, verified=True, checks=[check],
                timing_checks=[check], timing=timing, best_before_median_ms=0.505, speedup_vs_best=1.01,
                confirmation={"incumbent_median_ms": 0.49, "challenger_median_ms": 0.5, "kept": False})
    text = feedback_text(b, tiny_problem)
    assert "re-run back to back" in text and "previous best stays" in text
    assert "close win not confirmed" in outcome_line(b)
