# autokernel

LLM-driven search for fast, correct GPU kernels, judged by an evaluator that trusts
nothing the model says. You give it a computation, a precision requirement, a hardware
target, a language, an objective and a budget; it iterates and hands back the best
kernel it found with measured numbers against a named baseline.

Design: `docs/ARCHITECTURE.md`. Read that first.

## Layout

- `autokernel/`  Python package: spec (frontend), search (middle), backends, providers, metrics
- `harness/`     C++/CUDA evaluator process used by the `cuda_cpp` backend
- `specs/`       example problem specs
- `runs/`        ledgers and scratchpads from search runs (git-ignored)
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
pytest -q                                                   # 24 tests, GPU ones included
```

Runs land in `runs/<run-id>/` with a JSONL ledger of every attempt, every prompt and
reply, the model's scratchpad and the best kernel found. `--resume <run-id>` continues one.

The `.venv` is created with `--system-site-packages` on purpose: the system Python
already has torch 2.11 with CUDA 13.0 and Triton 3.6, which the Triton backend and the
reference implementations use.
