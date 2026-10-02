# autokernel

LLM-driven search for fast, correct GPU kernels, judged by an evaluator that trusts
nothing the model says. You give it a computation, a precision requirement, a hardware
target, a language, an objective and a budget; it iterates and hands back the best
kernel it found with measured numbers against a named baseline.

Design: `docs/ARCHITECTURE.md`. 

## Layout

- `autokernel/`  Python package: spec (frontend), search (middle), backends, providers, metrics
- `harness/`     C++/CUDA evaluator process used by the `cuda_cpp` backend
- `specs/`       example problem specs
- `runs/`        every search run as it happens: ledger, prompts, candidates, build logs (git-ignored)
- `results/`     exported runs worth keeping, each with a `REPORT.md` (committed)
- `tests/`       pytest, including the known-cheat suite

## Build and run

```
cmake --preset laptop && cmake --build --preset laptop      # the C++ harness
source .venv/bin/activate                                   # python side (venv is editable-installed)
autokernel doctor                                           # what this machine can build, run, measure
autokernel eval specs/gemm_fp32.yaml --candidate tests/kernels/naive_gemm.cu   # benchmark one kernel file
autokernel prompt specs/gemm_fp32.yaml                      # see what the model would be asked
autokernel run specs/gemm_fp32.yaml --iterations 5          # search (needs ANTHROPIC_API_KEY in .env)
autokernel run specs/gemm_fp32.yaml --provider human        # you play the model
autokernel export runs/<run-id> --spec specs/gemm_fp32.yaml --recheck   # results/<run-id>/REPORT.md
pytest -q                                                   # 38 tests, GPU ones included
```

Runs land in `runs/<run-id>/` with a JSONL ledger of every attempt, every prompt and
reply, the model's scratchpad and the best kernel found. `--resume <run-id>` continues one.

`export` copies the parts of a run worth keeping into `results/<run-id>/` and writes a
`REPORT.md` from the ledger: problem, target, protocol, every attempt with its measured
numbers, the best kernel and the model's final notes. With `--recheck` it first measures
the best kernel again against a fresh baseline on the current machine and puts that
number at the top, since the in-run baseline reflects whatever method was in force at
the time. The exported `problem.json` carries its reference next to it, so
`autokernel eval results/<run-id>/problem.json --candidate results/<run-id>/best.cu`
reproduces the measurement anywhere the toolchain exists.

First live run: `results/20261001-182415-gemm_fp32/REPORT.md`. Five attempts of Sonnet 5
on an fp32 2048x1024x1024 GEMM on an RTX 5070 Laptop went from 0.92x to 1.02x of
cuBLAS (0.3320 ms against 0.3398 ms on re-measurement).

The `.venv` is created with `--system-site-packages` on purpose: the system Python
already has torch 2.11 with CUDA 13.0 and Triton 3.6, which the Triton backend and the
reference implementations use.
