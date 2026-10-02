"""Append-only JSONL of every attempt. Ground truth: the model reads a digest of it
and never edits it. Also the resume state."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

from .types import Attempt


def first_error_line(diagnostics: str, limit: int = 160) -> str:
    lines = [l.strip() for l in diagnostics.splitlines() if l.strip()]
    for l in lines:
        if "error" in l.lower() and "ptxas info" not in l:
            return l[:limit]
    return lines[0][:limit] if lines else "no compiler output"


def category(a: Attempt) -> str:
    if a.duplicate_of:
        return "duplicates of earlier sources"
    if a.error:
        return "not evaluated (no usable code in the reply)"
    if not a.build_ok:
        return "build failures"
    if not a.run_ok:
        return "runtime failures"
    if not a.verified:
        return "wrong results"
    return "verified"


def outcome_line(a: Attempt) -> str:
    prefix = f"duplicate of #{a.duplicate_of}, not re-run: " if a.duplicate_of else ""
    if a.error:
        return prefix + f"no evaluation: {a.error}"
    if not a.build_ok:
        return prefix + f"build failed: {first_error_line(a.build_diagnostics)}"
    if not a.run_ok:
        return prefix + f"runtime failure at '{a.run_stage}': {a.run_error[:160]}"
    if not a.verified:
        bad = []
        for c in a.checks + a.timing_checks:
            if not c.get("passed"):
                if c.get("note"):
                    bad.append(f"{c['name']} {c['note']}")
                else:
                    pct = 100.0 * c.get("n_bad", 0) / max(c.get("n_total", 1), 1)
                    bad.append(f"{c['name']} {pct:.1f}% of elements off, {c.get('n_nan', 0)} NaN")
        return prefix + ("wrong results: " + "; ".join(dict.fromkeys(bad)) if bad else "not verified")
    text = f"verified, median {a.median_ms:.4f} ms"
    if a.speedup_vs_baseline:
        text += f", {a.speedup_vs_baseline:.2f}x baseline"
    if a.kept:
        text += ", new best"
    elif a.confirmation and not a.confirmation.get("kept"):
        text += ", close win not confirmed on re-run (not kept)"
    elif a.speedup_vs_best is not None:
        text += f", {a.speedup_vs_best:.2f}x the best (not kept)"
    return prefix + text


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, attempt: Attempt) -> None:
        with self.path.open("a") as f:
            f.write(attempt.model_dump_json() + "\n")

    def load(self) -> list[Attempt]:
        if not self.path.is_file():
            return []
        return [Attempt.model_validate_json(line) for line in self.path.read_text().splitlines() if line.strip()]

    @staticmethod
    def best(attempts: list[Attempt]) -> Optional[Attempt]:
        """The fastest verified attempt, skipping close wins that failed their
        back-to-back confirmation. Ties go to the earliest (so a duplicate never
        displaces its original)."""
        verified = [a for a in attempts if a.verified and a.timing
                    and not (a.confirmation and not a.confirmation.get("kept"))]
        return min(verified, key=lambda a: a.median_ms) if verified else None

    @staticmethod
    def digest(attempts: list[Attempt], recent: int = 5, older_window: int = 20, chain: int = 12) -> str:
        """What the model sees of the history. Bounded: the best, the chain of
        improvements, counts over everything older than the window, one line each
        for the `older_window` attempts before the recent ones, and the `recent`
        most recent in detail."""
        if not attempts:
            return ""
        best = Ledger.best(attempts)
        lines: list[str] = []
        if best is not None:
            lines.append(f"Best so far: attempt #{best.id}, median {best.median_ms:.4f} ms"
                         + (f" ({best.speedup_vs_baseline:.2f}x baseline)" if best.speedup_vs_baseline else "")
                         + f". Hypothesis: {best.hypothesis[:300]}")
        else:
            lines.append("Best so far: none verified yet.")
        kept = [a for a in attempts if a.kept]
        if len(kept) > 1:
            shown = kept[-chain:]
            start = "... -> " if len(kept) > len(shown) else ""
            lines.append("Improvements so far: " + start + " -> ".join(f"#{a.id} {a.median_ms:.4f} ms" for a in shown))
        older, latest = attempts[:-recent], attempts[-recent:]
        if older:
            counts = Counter(category(a) for a in older)
            summary = ", ".join(f"{n} {cat}" for cat, n in counts.most_common())
            lines.append(f"{len(older)} earlier attempts (#{older[0].id} to #{older[-1].id}): {summary}.")
            window = older[-older_window:]
            label = "Earlier attempts" if len(window) == len(older) else f"The last {len(window)} of them"
            lines.append(f"{label}: " + "; ".join(f"#{a.id} {outcome_line(a)[:80]}" for a in window))
        lines.append("Most recent attempts, oldest first:")
        for a in latest:
            parent = f" (from #{a.parent_id})" if a.parent_id else ""
            lines.append(f"- #{a.id}{parent}: {outcome_line(a)}. Hypothesis: {a.hypothesis[:300]}")
        return "\n".join(lines)
