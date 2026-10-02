"""The search loop: prompt, call the provider, parse, build, run, verify, record.

Everything the next step needs is on disk before the next provider call, so a run can
be resumed by pointing at its directory."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from ..backends import Backend, EvalCase
from ..backends.protocol import TimingStats
from ..providers.base import Provider, ProviderError, Reply
from ..spec.problem import Problem
from ..spec.reference import generate_inputs, load_reference, reference_source, run_reference
from .ledger import Ledger, outcome_line
from .prompt import PromptContext, build_system_prompt, build_user_prompt, feedback_text, parse_reply
from .scratchpad import Scratchpad
from .strategy import make_strategy
from .types import Attempt

Log = Callable[[str], None]


def seeds_for(problem: Problem, attempt_id: int, round: int = 0) -> tuple[int, int]:
    """Correctness and timing seeds for an attempt. `round` > 0 gives a second,
    disjoint pair for re-runs of the same attempt."""
    base = problem.protocol.seed * 1_000_003 + round * 500_000
    return base + 2 * attempt_id, base + 2 * attempt_id + 1


BASELINE_PROCESSES = 3
BASELINE_ROUNDS = 2
RETRY_BASE_S = 30.0
RETRY_MAX_S = 600.0


def compute_baseline(problem: Problem, backend: Backend, ref_fn, run_dir: Path) -> dict:
    """Time the reference like a candidate in several fresh processes, several rounds
    each, and keep the fastest round. cuBLAS via torch settles into one of two speeds
    per process on the development laptop; the best is the honest bar to clear."""
    path = run_dir / "baseline.json"
    if path.is_file():
        return json.loads(path.read_text())
    tmp = run_dir / "baseline_tmp"
    tmp.mkdir(exist_ok=True)
    (tmp / "problem.json").write_text(problem.model_dump_json())
    warmup = max(problem.protocol.warmup, 10)
    results = []
    for i in range(BASELINE_PROCESSES):
        out = tmp / f"p{i}.json"
        cmd = [sys.executable, "-m", "autokernel.search.baseline", str(tmp / "problem.json"), str(out),
               "--flush-bytes", str(backend.flush_bytes()), "--rounds", str(BASELINE_ROUNDS), "--warmup", str(warmup)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0 or not out.is_file():
            raise RuntimeError(f"baseline measurement failed: {proc.stderr.strip()[-2000:]}")
        results.append(json.loads(out.read_text()))
    best = min(range(len(results)), key=lambda i: results[i]["stats"]["median_ms"])
    data = {"name": f"PyTorch reference, best of {BASELINE_PROCESSES} processes x {BASELINE_ROUNDS} rounds",
            "seed": results[best]["seed"], "trial_ms": results[best]["trial_ms"], "stats": results[best]["stats"],
            "processes": [[r["median_ms"] for r in res["rounds"]] for res in results]}
    path.write_text(json.dumps(data, indent=2))
    shutil.rmtree(tmp, ignore_errors=True)
    return data


def evaluate_candidate(problem: Problem, backend: Backend, ref_fn, source: str,
                       attempt: Attempt, workdir: Path, round: int = 0) -> Attempt:
    """Build, run and verify one source, recording everything on `attempt`. The
    compiled library is deleted afterwards unless `protocol.keep_build_artifacts`."""
    build = backend.build(source, workdir)
    attempt.build_ok = build.ok
    attempt.build_diagnostics = build.diagnostics
    attempt.build_seconds = build.seconds
    attempt.resources = build.resources
    if not build.ok:
        return attempt
    try:
        check_seed, time_seed = seeds_for(problem, attempt.id, round)
        attempt.seeds = {"check": check_seed, "time": time_seed}
        check_inputs = generate_inputs(problem, check_seed, backend.device)
        check_refs = run_reference(problem, ref_fn, check_inputs)
        time_inputs = generate_inputs(problem, time_seed, backend.device)
        time_refs = run_reference(problem, ref_fn, time_inputs)
        ev = backend.run(build, EvalCase(check_inputs, check_refs, time_inputs, time_refs), workdir)
    finally:
        if build.artifact and not problem.protocol.keep_build_artifacts:
            Path(build.artifact).unlink(missing_ok=True)
    attempt.run_ok = ev.ok
    attempt.run_stage = ev.stage
    attempt.run_error = ev.error
    attempt.run_seconds = ev.seconds
    attempt.checks = [c.to_dict() for c in ev.checks]
    attempt.timing_checks = [c.to_dict() for c in ev.timing_checks]
    attempt.trial_ms = ev.trial_ms
    attempt.timing = ev.timing.to_dict() if ev.timing else None
    attempt.check_launch_ms = ev.check_launch_ms
    attempt.verified = ev.verified
    return attempt


def copy_evaluation(src: Attempt, dst: Attempt) -> None:
    """A duplicate source gets the original's measurements instead of a re-run."""
    for name in ("build_ok", "build_diagnostics", "resources", "run_ok", "run_stage", "run_error", "verified",
                 "checks", "timing_checks", "timing", "trial_ms", "check_launch_ms", "seeds"):
        setattr(dst, name, getattr(src, name))
    dst.duplicate_of = src.id


def confirm_win(problem: Problem, backend: Backend, ref_fn, incumbent_source: str, challenger_source: str,
                attempt: Attempt, workdir: Path) -> dict:
    """Re-run the incumbent and the challenger back to back on fresh inputs. Guards
    against a 'win' that is really the GPU being cooler or faster-clocked than when
    the incumbent was measured."""
    inc = Attempt(id=attempt.id, provider="confirm", model="incumbent")
    evaluate_candidate(problem, backend, ref_fn, incumbent_source, inc, workdir / "incumbent", round=1)
    chal = Attempt(id=attempt.id, provider="confirm", model="challenger")
    evaluate_candidate(problem, backend, ref_fn, challenger_source, chal, workdir / "challenger", round=1)
    ok = inc.verified and chal.verified and inc.median_ms is not None and chal.median_ms is not None
    return {"incumbent_median_ms": inc.median_ms, "challenger_median_ms": chal.median_ms,
            "incumbent_verified": inc.verified, "challenger_verified": chal.verified,
            "kept": bool(ok and chal.median_ms < inc.median_ms)}


def call_with_retries(provider: Provider, system: str, user: str, max_tokens: int, retries: int,
                      log: Log = print, sleep: Callable[[float], None] = time.sleep) -> tuple[Reply, int]:
    """One provider call, repeated after transient failures with doubling waits
    (30 s, 60 s, ... up to 10 min). Returns the reply and how many retries it took."""
    delay = RETRY_BASE_S
    for n in range(retries + 1):
        try:
            return provider.complete(system, user, max_tokens), n
        except ProviderError as e:
            if not e.retryable or n == retries:
                raise
            log(f"provider error: {e}. Retry {n + 1}/{retries} in {delay:.0f} s")
            sleep(delay)
            delay = min(delay * 2, RETRY_MAX_S)
    raise AssertionError("unreachable")


def _source_of(run_dir: Path, a: Optional[Attempt]) -> Optional[str]:
    if a is None or not a.source_path:
        return None
    p = run_dir / a.source_path
    return p.read_text() if p.is_file() else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def progress_line(a: Attempt, tokens: int) -> str:
    return f"#{a.id:03d} {outcome_line(a)} | tokens {tokens:,} | {a.seconds.get('total', 0):.0f} s"


def run_search(problem: Problem, run_dir: Path, provider: Provider, backend: Backend,
               log: Log = print, strategy_name: str = "greedy") -> dict:
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    (run_dir / "problem.json").write_text(problem.model_dump_json(indent=2, exclude={"spec_path"}))
    target = backend.describe_target()
    (run_dir / "target.json").write_text(json.dumps(target, indent=2))
    for sub in ("candidates", "prompts", "attempts"):
        (run_dir / sub).mkdir(exist_ok=True)

    ref_fn = load_reference(problem)
    ref_src = reference_source(ref_fn)
    baseline = compute_baseline(problem, backend, ref_fn, run_dir)
    base_stats = TimingStats(**baseline["stats"])
    log(f"target: {target.get('name')} ({target.get('arch')}); baseline {baseline['name']}: {base_stats.summary()}")

    fb = problem.feedback
    ledger = Ledger(run_dir / "ledger.jsonl")
    scratch = Scratchpad(run_dir / "scratchpad.md", fb.max_scratchpad_lines, fb.max_scratchpad_chars)
    strategy = make_strategy(strategy_name)
    attempts = ledger.load()
    if attempts:
        log(f"resuming {run_id} with {len(attempts)} attempts on record")
    tokens = sum(a.usage.get("total", 0) for a in attempts)
    t_start = time.time()
    stop = "max_iterations"

    def context(next_id: int) -> PromptContext:
        best = Ledger.best(attempts)
        latest = attempts[-1] if attempts else None
        return PromptContext(
            problem=problem, fence=backend.code_fence, reference_source=ref_src,
            target_text=backend.target_text(), contract_text=backend.contract_text(),
            measurement_text=backend.measurement_text(problem), baseline=base_stats,
            best=best, best_source=_source_of(run_dir, best),
            latest=latest, latest_source=_source_of(run_dir, latest),
            scratchpad=scratch.read(), digest=Ledger.digest(attempts, fb.recent_attempts, fb.older_window),
            feedback=feedback_text(latest, problem, fb.max_diag_lines) if latest else "",
            next_id=next_id)

    system = build_system_prompt(context(1))  # fixed for the run; providers may cache it

    while True:
        if len(attempts) >= problem.budget.max_iterations:
            break
        if problem.budget.max_tokens and tokens >= problem.budget.max_tokens:
            stop = "max_tokens"
            break
        if problem.budget.max_wall_seconds and time.time() - t_start >= problem.budget.max_wall_seconds:
            stop = "max_wall_seconds"
            break

        next_id = attempts[-1].id + 1 if attempts else 1
        best = Ledger.best(attempts)
        parent = strategy.parent(attempts)
        user = build_user_prompt(context(next_id))
        (run_dir / "prompts" / f"{next_id:03d}_prompt.md").write_text(f"SYSTEM:\n{system}\n\nUSER:\n{user}\n")

        attempt = Attempt(id=next_id, parent_id=parent.id if parent else None, run_id=run_id, timestamp=_now(),
                          provider=provider.name, model=provider.model, protocol=problem.protocol.model_dump(),
                          target_arch=target.get("arch"), baseline_median_ms=base_stats.median_ms)
        t_step = time.time()
        try:
            reply, attempt.retries = call_with_retries(provider, system, user, problem.model.max_output_tokens,
                                                       problem.model.max_retries, log)
        except ProviderError as e:
            log(f"stopping: {e}")
            stop = f"provider: {e}"
            break
        t_reply = time.time() - t_step
        attempt.usage = reply.usage.to_dict()
        attempt.stop_reason = reply.stop_reason
        attempt.model = reply.model or provider.model
        tokens += reply.usage.total
        (run_dir / "prompts" / f"{next_id:03d}_reply.md").write_text(reply.text)
        if reply.thinking:
            (run_dir / "prompts" / f"{next_id:03d}_thinking.md").write_text(reply.thinking)
        if reply.block_types:
            attempt.usage["blocks"] = reply.block_types

        cand = parse_reply(reply.text, backend.code_fence)
        attempt.hypothesis = cand.hypothesis
        if cand.scratchpad is not None:
            attempt.scratchpad_truncated = scratch.write(cand.scratchpad)
        if cand.source is None:
            attempt.error = "; ".join(cand.problems) or "no code in reply"
        else:
            src_path = run_dir / "candidates" / f"{next_id:03d}{backend.source_extension}"
            src_path.write_text(cand.source)
            attempt.source_path = str(src_path.relative_to(run_dir))
            attempt.source_sha256 = hashlib.sha256(cand.source.encode()).hexdigest()
            original = next((a for a in attempts if a.source_sha256 == attempt.source_sha256 and not a.error), None)
            if original is not None:
                copy_evaluation(original, attempt)
            else:
                evaluate_candidate(problem, backend, ref_fn, cand.source, attempt, run_dir / "attempts" / f"{next_id:03d}")
            if attempt.verified and attempt.timing:
                med = attempt.median_ms
                attempt.speedup_vs_baseline = base_stats.median_ms / med if med else None
                if best is not None:
                    attempt.best_before_median_ms = best.median_ms
                    attempt.speedup_vs_best = best.median_ms / med if med else None
                attempt.kept = best is None or med < best.median_ms
                margin = problem.protocol.confirm_margin
                if attempt.kept and best is not None and margin > 0 and med > best.median_ms * (1 - margin):
                    incumbent_source = _source_of(run_dir, best)
                    if incumbent_source is not None:
                        log(f"#{next_id:03d} wins by less than {100 * margin:.0f}%; re-running both back to back")
                        attempt.confirmation = confirm_win(problem, backend, ref_fn, incumbent_source, cand.source,
                                                           attempt, run_dir / "attempts" / f"{next_id:03d}" / "confirm")
                        attempt.kept = attempt.confirmation["kept"]
                if attempt.kept:
                    shutil.copyfile(src_path, run_dir / f"best{backend.source_extension}")
                    (run_dir / "best.json").write_text(attempt.model_dump_json(indent=2))
        attempt.seconds = {"total": time.time() - t_step, "reply": t_reply,
                           "build": attempt.build_seconds, "run": attempt.run_seconds}
        ledger.append(attempt)
        attempts.append(attempt)
        log(progress_line(attempt, tokens))

    best = Ledger.best(attempts)
    summary = {
        "run_id": run_id, "stopped_because": stop, "attempts": len(attempts),
        "verified": sum(a.verified for a in attempts), "tokens": tokens,
        "wall_seconds": round(time.time() - t_start, 1),
        "baseline_median_ms": base_stats.median_ms,
        "best": None if best is None else {
            "id": best.id, "median_ms": best.median_ms, "speedup_vs_baseline": best.speedup_vs_baseline,
            "source": str(run_dir / f"best{backend.source_extension}")},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if best is None:
        log(f"done ({stop}): {len(attempts)} attempts, none verified. Ledger: {ledger.path}")
    else:
        log(f"done ({stop}): best is attempt #{best.id} at {best.median_ms:.4f} ms, "
            f"{best.speedup_vs_baseline:.2f}x baseline, saved to {summary['best']['source']}")
    return summary
