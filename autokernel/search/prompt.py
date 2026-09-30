"""Prompt assembly, reply parsing and tier-1 feedback text.

The prompt is rebuilt from scratch every step from the problem, the target, the
current best, the model's notes, the ledger digest and the latest feedback."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..backends.protocol import OutputCheck, TimingStats
from ..metrics.ptxas import resources_summary
from ..spec.problem import Problem, TensorSpec
from .types import Attempt, Candidate

SYSTEM_PROMPT = """You are the kernel author inside autokernel, an automated search for fast, correct GPU kernels. Each turn you receive the computation to implement (reference PyTorch code and tensor shapes), the precision requirement, the target hardware, the candidate contract, how candidates are measured, your own notes from earlier turns, a summary of earlier attempts with their measured results, and detailed feedback on the most recent attempt. Each turn you produce exactly one new candidate.

Rules:
- You cannot run anything. The system compiles your candidate, runs it on fresh random inputs you never see, checks every output element against a hidden reference, and times it. Only measured results count. Never state numbers you were not given.
- Correctness first. A candidate outside tolerance on any element scores nothing, however fast it is.
- Do not try to detect or game the measurement. Outputs are filled with NaN before every call, inputs differ between the correctness and timing runs, and the timing run's outputs are verified as well.
- Prefer one deliberate change per attempt so the measured difference is attributable. When the current best is far from the baseline, larger structural changes are fine.
- The candidate must be complete and self-contained. Partial files or diffs cannot be compiled.

Respond in exactly this format and nothing else:

<hypothesis>
One short paragraph: what you believe the current bottleneck is, what you are changing, and the effect you expect, with a rough number.
</hypothesis>

<scratchpad>
The complete new contents of your notes file; it replaces the previous one. Keep what stays useful: ideas tried with their measured outcome, ideas not yet tried, facts learned about this target and this problem. Stay under about 60 lines.
</scratchpad>

```{fence}
The complete candidate source file.
```"""


@dataclass
class PromptContext:
    problem: Problem
    fence: str
    reference_source: str
    target_text: str
    contract_text: str
    measurement_text: str
    baseline: Optional[TimingStats]
    best: Optional[Attempt]
    best_source: Optional[str]
    latest: Optional[Attempt]
    latest_source: Optional[str]
    scratchpad: str
    digest: str
    feedback: str
    next_id: int


def _init_text(t: TensorSpec) -> str:
    if t.init == "uniform":
        return f"uniform[{t.low:g}, {t.high:g})"
    if t.init == "normal":
        return f"normal(mean {t.mean:g}, std {t.std:g})"
    if t.init == "randint":
        return f"integers in [{int(t.low)}, {int(t.high)})"
    return t.init


def tensor_table(tensors: list[TensorSpec], with_init: bool) -> str:
    head = "| name | shape | dtype |" + (" values |" if with_init else "")
    sep = "|---|---|---|" + ("---|" if with_init else "")
    rows = [head, sep]
    for t in tensors:
        row = f"| {t.name} | {'x'.join(map(str, t.shape))} | {t.dtype} |"
        if with_init:
            row += f" {_init_text(t)} |"
        rows.append(row)
    return "\n".join(rows)


def build_user_prompt(ctx: PromptContext) -> str:
    P = ctx.problem
    c = P.computation
    parts: list[str] = []

    head = f"# Computation: {c.name}"
    if c.description:
        head += "\n" + c.description.strip()
    parts.append(head)
    parts.append("Reference implementation (PyTorch; this defines correct):\n```python\n"
                 + ctx.reference_source.rstrip() + "\n```")
    parts.append("Inputs, in call order (row-major contiguous):\n" + tensor_table(c.inputs, with_init=True))
    parts.append("Outputs, in call order:\n" + tensor_table(c.outputs, with_init=False))
    if c.flops:
        parts.append(f"Work per call: {c.flops / 1e9:.3f} GFLOP.")

    prec = ["# Precision requirement",
            "For every element of each output: |out - ref| <= atol + rtol * |ref|. NaN or Inf in an output fails."]
    for o in c.outputs:
        t = P.precision.tolerance_for(o.name)
        prec.append(f"- {o.name}: atol = {t.atol:g}, rtol = {t.rtol:g}")
    prec.append("Reduced-precision paths allowed: " + (", ".join(P.precision.allowed_reduced_precision) or "none") + ".")
    if P.precision.notes:
        prec.append(P.precision.notes.strip())
    parts.append("\n".join(prec))

    parts.append("# Target hardware\n" + ctx.target_text)
    parts.append("# Candidate contract\n" + ctx.contract_text)

    meas = ["# How candidates are measured", ctx.measurement_text,
            f"- Objective: {P.goals.objective}. Lower median time is better."]
    if P.goals.max_registers:
        meas.append(f"- Constraint: at most {P.goals.max_registers} registers per thread.")
    if P.goals.max_smem_bytes:
        meas.append(f"- Constraint: at most {P.goals.max_smem_bytes} bytes of shared memory per block.")
    if P.goals.notes:
        meas.append(P.goals.notes.strip())
    if ctx.baseline is not None:
        meas.append(f"- Baseline: the PyTorch reference measured the same way: {ctx.baseline.summary()}.")
    parts.append("\n".join(meas))

    if ctx.best is not None and ctx.best_source:
        parts.append(f"# Current best candidate: attempt #{ctx.best.id}, median {ctx.best.median_ms:.4f} ms"
                     + (f", {ctx.best.speedup_vs_baseline:.2f}x baseline" if ctx.best.speedup_vs_baseline else "")
                     + f"\n```{ctx.fence}\n{ctx.best_source.rstrip()}\n```")
    else:
        parts.append("# Current best candidate\nNone yet: no attempt has passed verification.")
    if ctx.latest is not None and ctx.latest_source and (ctx.best is None or ctx.latest.id != ctx.best.id):
        parts.append(f"# Most recent attempt: #{ctx.latest.id} (not the best)\n```{ctx.fence}\n{ctx.latest_source.rstrip()}\n```")

    parts.append("# Your notes\n" + ctx.scratchpad.strip())
    parts.append("# Attempt history\n" + (ctx.digest.strip() or "No attempts yet."))
    if ctx.latest is not None:
        parts.append(f"# Feedback on attempt #{ctx.latest.id}\n" + ctx.feedback.strip())
    else:
        parts.append("# Feedback\nThis is the first attempt. Start with a kernel you are confident is correct and "
                     "reasonably efficient; you will iterate from measurements.")
    parts.append(f"Produce attempt #{ctx.next_id}.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- reply parsing

def _tag(name: str) -> re.Pattern:
    return re.compile(rf"<{name}>\s*(.*?)\s*</{name}>", re.S)


_FENCE = re.compile(r"```([\w+#.-]*)[ \t]*\r?\n(.*?)```", re.S)
_LANGS = {
    "cuda": {"cuda", "cu", "cpp", "c++", "cxx", "c"},
    "python": {"python", "py", "triton"},
}


def parse_reply(text: str, fence: str) -> Candidate:
    problems: list[str] = []
    m = _tag("hypothesis").search(text)
    hypothesis = m.group(1).strip() if m else ""
    m = _tag("scratchpad").search(text)
    scratchpad = m.group(1) if m else None
    if not hypothesis:
        problems.append("missing <hypothesis> section")
    if scratchpad is None:
        problems.append("missing <scratchpad> section")

    stripped = _tag("scratchpad").sub("", _tag("hypothesis").sub("", text))
    blocks = _FENCE.findall(stripped)
    wanted = _LANGS.get(fence, {fence})
    matching = [body for lang, body in blocks if lang.lower() in wanted]
    source: Optional[str] = None
    if matching:
        source = max(matching, key=len)
    elif blocks:
        source = max((body for _, body in blocks), key=len)
        problems.append("code block had no recognised language tag; used the longest block")
    elif stripped.count("```") % 2 == 1:
        problems.append("unterminated code block (was the reply cut off?)")
    else:
        problems.append("no code block found in the reply")
    if not hypothesis and source is None:
        hypothesis = text.strip()[:300]
    return Candidate(hypothesis=hypothesis, source=(source.rstrip() + "\n") if source else None,
                     scratchpad=scratchpad, raw_reply=text, problems=problems)


# ---------------------------------------------------------------- feedback (tier 1)

_NOISE = ("ptxas info", "bytes stack frame", "Compile time =", "bytes gmem")


def diagnostics_excerpt(diagnostics: str, max_lines: int) -> str:
    kept = [l for l in diagnostics.splitlines() if l.strip() and not any(n in l for n in _NOISE)]
    kept = [re.sub(r"^\S*/candidate\.cu", "candidate.cu", l) for l in kept]
    if len(kept) > max_lines:
        return "\n".join(kept[:max_lines]) + f"\n... ({len(kept) - max_lines} more lines)"
    return "\n".join(kept) or "(no output)"


def feedback_text(a: Attempt, problem: Problem, max_diag_lines: int = 40) -> str:
    lines: list[str] = []
    if a.error:
        lines.append(f"Attempt failed before evaluation: {a.error}.")
    if a.build_ok:
        lines.append(f"Build: OK in {a.build_seconds:.1f} s. {resources_summary(a.resources)}")
    elif a.source_path:
        lines.append(f"Build: FAILED. Compiler output:\n```\n{diagnostics_excerpt(a.build_diagnostics, max_diag_lines)}\n```")
    if a.build_ok and not a.run_ok:
        lines.append(f"Run: FAILED at stage '{a.run_stage}': {a.run_error}")
    elif a.build_ok:
        lines.append("Run: OK.")
        lines.append("Correctness on set A: " + "; ".join(OutputCheck(**c).summary() for c in a.checks))
        lines.append("Correctness on set B (timing inputs): " + "; ".join(OutputCheck(**c).summary() for c in a.timing_checks))
        if a.timing:
            ts = TimingStats(**a.timing)
            first = f"; first launch {a.check_launch_ms:.4f} ms" if a.check_launch_ms is not None else ""
            lines.append(f"Timing: {ts.summary()}{first}.")
            if problem.computation.flops:
                lines.append(f"Throughput at the median: {problem.computation.flops / ts.median_ms / 1e6:.0f} GFLOP/s.")
            if a.speedup_vs_baseline:
                lines.append(f"Versus baseline: {a.speedup_vs_baseline:.3f}x (baseline median {a.baseline_median_ms:.4f} ms; above 1.0 means faster than the baseline).")
            if a.speedup_vs_best is not None:
                lines.append(f"Versus previous best: {a.speedup_vs_best:.3f}x (best median {a.best_before_median_ms:.4f} ms).")
        if a.verified:
            if a.kept:
                lines.append("Result: verified. Kept as the new best.")
            elif a.best_before_median_ms is not None:
                lines.append("Result: verified, but not faster than the current best, so not kept.")
            else:
                lines.append("Result: verified.")
        else:
            lines.append("Result: NOT verified. Speed does not count until every element of every output passes on both input sets.")
    if a.stop_reason == "max_tokens":
        lines.append("Your previous reply hit the output token limit and was cut off. Be more concise: shorter notes and no text outside the required sections.")
    return "\n".join(lines) if lines else "No feedback recorded."
