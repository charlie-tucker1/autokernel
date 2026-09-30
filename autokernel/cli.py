"""Command-line entry point.

  autokernel run SPEC        search for a kernel (needs a provider; see model.provider)
  autokernel eval SPEC --candidate FILE
                             build, run and verify one kernel file, no model involved
  autokernel prompt SPEC     print the prompt the model would get for attempt 1
  autokernel describe SPEC   print the target description
  autokernel doctor          what this machine can build, run and measure
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .env import load_dotenv


def _spec_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("spec", help="problem spec (YAML)")
    sp.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="override a spec field, e.g. --set budget.max_iterations=3")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autokernel", description="LLM-driven kernel search with an untrusting evaluator")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="search for a kernel")
    _spec_args(r)
    r.add_argument("--runs-dir", default="runs", help="where run directories go (default: ./runs)")
    r.add_argument("--run-id", help="name of the run directory; default is a timestamp plus the computation name")
    r.add_argument("--resume", metavar="RUN_ID", help="continue an existing run directory")
    r.add_argument("--provider", help="shorthand for --set model.provider=...")
    r.add_argument("--model", help="shorthand for --set model.model=...")
    r.add_argument("--iterations", type=int, help="shorthand for --set budget.max_iterations=...")

    e = sub.add_parser("eval", help="build, run and verify one candidate file without a model")
    _spec_args(e)
    e.add_argument("--candidate", required=True, help="kernel source file")
    e.add_argument("--workdir", help="where build products and the report go (default: runs/eval-<timestamp>)")

    pr = sub.add_parser("prompt", help="print the prompt a model would receive for the first attempt")
    _spec_args(pr)

    d = sub.add_parser("describe", help="print the target description")
    _spec_args(d)

    sub.add_parser("doctor", help="report what this machine can build, run and measure")
    return p


def cmd_run(args: argparse.Namespace) -> int:
    from .backends import make_backend
    from .providers import make_provider
    from .search.run import run_search
    from .spec.loader import load_problem

    overrides = list(args.overrides)
    if args.provider:
        overrides.append(f"model.provider={args.provider}")
    if args.model:
        overrides.append(f"model.model={args.model}")
    if args.iterations is not None:
        overrides.append(f"budget.max_iterations={args.iterations}")
    problem = load_problem(args.spec, overrides)
    if args.resume:
        run_dir = Path(args.runs_dir) / args.resume
        if not run_dir.is_dir():
            sys.exit(f"no run directory {run_dir}")
    else:
        run_id = args.run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{problem.computation.name}"
        run_dir = Path(args.runs_dir) / run_id
    backend = make_backend(problem)
    provider = make_provider(problem, run_dir)
    print(f"run directory: {run_dir}")
    summary = run_search(problem, run_dir, provider, backend)
    return 0 if summary["best"] else 2


def cmd_eval(args: argparse.Namespace) -> int:
    from .backends import make_backend
    from .backends.protocol import TimingStats
    from .search.prompt import feedback_text
    from .search.run import compute_baseline, evaluate_candidate
    from .search.types import Attempt
    from .spec.loader import load_problem
    from .spec.reference import load_reference

    problem = load_problem(args.spec, args.overrides)
    backend = make_backend(problem)
    workdir = Path(args.workdir) if args.workdir else Path("runs") / f"eval-{datetime.now():%Y%m%d-%H%M%S}"
    workdir.mkdir(parents=True, exist_ok=True)
    ref_fn = load_reference(problem)
    source = Path(args.candidate).read_text()
    (workdir / f"candidate{backend.source_extension}").write_text(source)
    attempt = Attempt(id=1, source_path=f"candidate{backend.source_extension}", provider="eval", model="none")
    baseline = compute_baseline(problem, backend, ref_fn, workdir)
    base = TimingStats(**baseline["stats"])
    attempt.baseline_median_ms = base.median_ms
    evaluate_candidate(problem, backend, ref_fn, source, attempt, workdir / "build")
    if attempt.verified and attempt.timing:
        attempt.speedup_vs_baseline = base.median_ms / attempt.median_ms
    target = backend.describe_target()
    print(f"target: {target.get('name')} ({target.get('arch')})")
    print(f"baseline ({baseline['name']}): {base.summary()}")
    print(feedback_text(attempt, problem, problem.feedback.max_diag_lines))
    (workdir / "attempt.json").write_text(attempt.model_dump_json(indent=2))
    print(f"details: {workdir}")
    return 0 if attempt.verified else 1


def cmd_prompt(args: argparse.Namespace) -> int:
    from .backends import make_backend
    from .search.prompt import SYSTEM_PROMPT, PromptContext, build_user_prompt
    from .search.scratchpad import DEFAULT
    from .spec.loader import load_problem
    from .spec.reference import load_reference, reference_source

    problem = load_problem(args.spec, args.overrides)
    backend = make_backend(problem)
    ref_fn = load_reference(problem)
    ctx = PromptContext(problem=problem, fence=backend.code_fence, reference_source=reference_source(ref_fn),
                        target_text=backend.target_text(), contract_text=backend.contract_text(),
                        measurement_text=backend.measurement_text(problem), baseline=None,
                        best=None, best_source=None, latest=None, latest_source=None,
                        scratchpad=DEFAULT, digest="", feedback="", next_id=1)
    print("SYSTEM:\n" + SYSTEM_PROMPT.replace("{fence}", backend.code_fence))
    print("\nUSER:\n" + build_user_prompt(ctx))
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    from .backends import make_backend
    from .spec.loader import load_problem

    problem = load_problem(args.spec, args.overrides)
    print(make_backend(problem).target_text())
    return 0


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.cmd == "doctor":
        from .metrics.doctor import report
        sys.exit(report())
    from .backends import BackendError
    from .providers import ProviderError

    handler = {"run": cmd_run, "eval": cmd_eval, "prompt": cmd_prompt, "describe": cmd_describe}[args.cmd]
    try:
        sys.exit(handler(args))
    except (BackendError, ProviderError, FileNotFoundError) as e:
        sys.exit(f"autokernel: {e}")
    except KeyboardInterrupt:
        sys.exit("autokernel: interrupted; the run directory can be resumed with --resume")


if __name__ == "__main__":
    main()
