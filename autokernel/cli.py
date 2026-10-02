"""Command-line entry point.

  autokernel run SPEC        search for a kernel (needs a provider; see model.provider)
  autokernel eval SPEC --candidate FILE
                             build, run and verify one kernel file, no model involved
  autokernel prompt SPEC     print the prompt the model would get for attempt 1
  autokernel describe SPEC   print the target description
  autokernel export RUN_DIR  copy a run's committable results to results/<run-id> with a REPORT.md
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

    ex = sub.add_parser("export", help="copy a run's committable results to results/<run-id> and write a REPORT.md")
    ex.add_argument("run_dir", help="the run directory, e.g. runs/20261001-182415-gemm_fp32")
    ex.add_argument("--out", help="destination directory (default: results/<run-id>)")
    ex.add_argument("--spec", help="the spec the run was started from; its reference file is copied so the "
                                   "exported problem.json is a complete spec on its own")
    ex.add_argument("--recheck", action="store_true",
                    help="first re-measure the best kernel against a fresh baseline on this machine (needs --spec)")
    ex.add_argument("--no-prompts", action="store_true", help="leave out prompts/ (every prompt, reply and thinking block)")
    ex.add_argument("--note", default="", help="free text appended to the report")
    ex.add_argument("--force", action="store_true", help="write into a destination that already has files")

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
    from .search.prompt import PromptContext, build_system_prompt, build_user_prompt
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
    print("SYSTEM:\n" + build_system_prompt(ctx))
    print("\nUSER:\n" + build_user_prompt(ctx))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    import json

    from .search.export import export_run, recheck_best
    from .spec.problem import Problem

    run_dir = Path(args.run_dir)
    if not (run_dir / "ledger.jsonl").is_file():
        sys.exit(f"{run_dir} has no ledger.jsonl; is it a run directory?")
    out = Path(args.out) if args.out else Path("results") / run_dir.resolve().name
    if out.is_dir() and any(out.iterdir()) and not args.force:
        sys.exit(f"{out} already has files; pass --force to write into it")
    spec = Path(args.spec).resolve() if args.spec else None
    recheck = None
    if args.recheck:
        if spec is None:
            sys.exit("--recheck needs --spec so the reference implementation can be found")
        best = next((p for p in run_dir.glob("best.*") if p.suffix in (".cu", ".py")), None)
        if best is None:
            sys.exit(f"{run_dir} has no best kernel to re-measure")
        problem = Problem.model_validate(json.loads((run_dir / "problem.json").read_text()))
        problem.spec_path = spec
        workdir = run_dir / f"recheck-{datetime.now():%Y%m%d-%H%M%S}"
        print(f"re-measuring {best.name} against a fresh baseline (work in {workdir})")
        recheck = recheck_best(problem, best, workdir)
        print(f"baseline {recheck['baseline']['summary']}")
        print(f"best kernel {recheck['candidate']['summary']}; verified {recheck['candidate']['verified']}; "
              f"speedup {recheck['speedup_vs_baseline'] or 0:.3f}x")
    report = export_run(run_dir, out, spec=spec, include_prompts=not args.no_prompts, recheck=recheck, notes=args.note)
    print(f"exported to {out}; report: {report}")
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

    handler = {"run": cmd_run, "eval": cmd_eval, "prompt": cmd_prompt, "describe": cmd_describe,
               "export": cmd_export}[args.cmd]
    try:
        sys.exit(handler(args))
    except (BackendError, ProviderError, FileNotFoundError) as e:
        sys.exit(f"autokernel: {e}")
    except KeyboardInterrupt:
        sys.exit("autokernel: interrupted; the run directory can be resumed with --resume")


if __name__ == "__main__":
    main()
