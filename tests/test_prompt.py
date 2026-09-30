from autokernel.search.prompt import diagnostics_excerpt, parse_reply

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
