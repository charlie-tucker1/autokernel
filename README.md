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
cmake --preset laptop && cmake --build --preset laptop   # harness
source .venv/bin/activate                                # python side
autokernel doctor                                        # what this machine can measure
```

The `.venv` is created with `--system-site-packages` on purpose: the system Python
already has torch 2.11 with CUDA 13.0 and Triton 3.6, which the Triton backend and the
reference implementations use.
