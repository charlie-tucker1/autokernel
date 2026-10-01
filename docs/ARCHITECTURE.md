# autokernel architecture

Status: In Progess

This document describes the intented structure of the
project.

## Goal

Given a computation, a precision requirement, a hardware target, a language, an
objective and a budget, search for the fastest correct kernel, using an LLM as the
proposal engine and a fixed, untrusting evaluator as the judge. The output is the best
kernel found, its measured numbers against a named baseline, and the full search
history.

Non-goals for now: training or fine-tuning models, multi-GPU, non-NVIDIA targets (kept
possible in the interfaces, not built), a custom compute DSL.

## Rule

The model produces text: kernel source and notes. It never runs commands, never sees a
shell, never decides what gets measured. Every executable step is fixed system code
whose capabilities the user chose at install time. Consequences:

- Nothing in this repo calls `sudo`. Profiler counters need a one-time admin setting
  (`NVreg_RestrictProfilingToAdminUsers=0`). `autokernel doctor` reports whether it is
  set and prints the commands for the human to run.
- Candidate code is untrusted. It runs in a subprocess with a timeout and, when
  bubblewrap is available, with no network and a read-only view of the repo. The model
  writes only the kernel plus a launch function with a fixed signature. The harness
  owns `main`.
- The evaluator trusts nothing the model reports. Timings, correctness and metrics come
  from the harness only. The threat model is reward hacking, as seen in Sakana's AI
  CUDA Engineer and CUDA-L1: kernels that read the reference from reused memory, skip
  work, or time a subset of the launch.

## Pipeline

The LLVM analogy, loosely: a frontend normalizes the request, the middle runs a search
over abstract Candidate and Attempt objects, and a backend knows one (language,
hardware) pair. The difference from LLVM is that there is no IR we transform; the model
does the transforming. What is shared across backends is the evaluation protocol and
the record format, so numbers from CUDA C++ and Triton are comparable.

### Frontend: `autokernel/spec`

One spec file (YAML) plus CLI flag overrides. Fields:

- `computation`: a Python module and function that is the reference implementation
  (PyTorch); inputs with shapes, dtypes and generators; outputs.
- `precision`: per output, tolerance mode (abs, rel, combined, ulp) and value; which
  reduced-precision paths the kernel may use (tf32, bf16), if any.
- `target`: hardware (auto-detected by default) and language (`cuda_cpp`, `triton`).
- `goals`: one objective (latency, throughput, utilization) and constraints (max
  registers, shared memory, device memory, precision).
- `budget`: tokens, wall time, iterations; whichever runs out first stops the search.
- `feedback`: tier (see Feedback tiers).
- `model`: provider and model id. API keys come from the environment.

The frontend produces a `Problem` object. Nothing downstream reads the YAML.

### Middle: `autokernel/search`

- `Candidate`: source text, the hypothesis the model wrote for it, parent id.
- `Attempt`: Candidate plus build result, eval result, metrics, token cost.
- `Ledger`: append-only JSONL of every Attempt, written before the next model call.
  Ground truth. The model reads a digest of it and never edits it.
- `Scratchpad`: a markdown file the model owns and rewrites each step: ideas tried,
  what it believes the bottleneck is, what to try next. Persisted next to the ledger.
- `Strategy`: `propose(state) -> list[Candidate]` and `observe(attempts)`. Greedy
  keep-if-better first; idea-branching (propose several natural-language ideas,
  implement each, keep the top k) second; population-based third.
- `Provider`: one interface, `complete(messages, tools) -> (reply, usage)`. Anthropic
  first, OpenAI-compatible second. Also `mock` (replays canned replies, for tests that
  cost nothing) and `human` (you paste the kernel), which doubles as a plain
  benchmarker for hand-written kernels.
- Prompt assembly: problem, target description from the backend, the candidate
  contract, the scratchpad, a ledger digest (best-ever in full, last N attempts in
  full, older ones as one line each), and the feedback for the latest attempt.
- Budget accounting from provider usage fields. Resumable: state is the ledger plus
  the scratchpad, so a restart continues where it stopped.

Each step is two-phase, following the Stanford fast-kernels result: the model first
states a hypothesis in plain language (bottleneck, change, expected effect), then
writes the code for it. Both are recorded, and the hypothesis is shown next to the
measured outcome in later digests.

### Backends: `autokernel/backends`

Interface:

- `describe_target() -> str`: arch, SM count, shared memory per SM, registers, L2 size,
  memory bandwidth, clocks. Goes into the prompt.
- `contract() -> str`: what the model must write: entry point signature and language
  rules. Goes into the prompt.
- `build(candidate) -> BuildResult`: diagnostics and resource usage (registers, spills,
  shared memory) from `ptxas -v` or Triton's compiled-kernel metadata.
- `run(build, protocol) -> EvalResult`: outputs, timings, checks.

`cuda_cpp`: the candidate is a `.cu` file exposing an `extern "C"` entry point that
takes tensor descriptors (pointer, dtype, ndim, shape, strides) and a stream. nvcc
compiles it to a shared library using the host compiler from the toolchain setup. The
C++ harness in `harness/` loads it with dlopen in a subprocess, reads inputs from
`.npy` files written by Python, runs the protocol, and writes outputs and timings back.

`triton`: the candidate is a Python module with a fixed function taking torch tensors.
It runs in a Python subprocess with torch; the same protocol implemented in Python.

Later: `ptx` (driver API load), `cute_dsl`, `hip`.

### Evaluation protocol (shared by every backend)

1. Inputs are generated fresh per attempt from a recorded seed. The model never sees
   values.
2. Output buffers are filled with NaN before every launch.
3. Correctness is measured on one input set, timing on a different one.
4. Reference outputs come from the PyTorch reference in the orchestrator process and
   are never available to the candidate process.
5. Timing: warmup launches, then N trials of one launch each between cudaEvent pairs
   on the stream, optional L2 flush between trials; report median, min, p90 and
   spread. Laptop power capping makes the median mandatory.
6. Baseline: the PyTorch reference timed the same way, plus cuBLAS where it applies.
7. A watchdog kills the subprocess on timeout, which is how hung kernels are recovered.

### Feedback tiers: `autokernel/metrics`

Collect everything; feed a digest. Raw data goes to the ledger on disk for humans and
pandas. Every token of feedback is paid again on every later step, so the model gets:

- Tier 1, always: compile diagnostics, ptxas resource usage, correctness detail (which
  output, max error, where), timing stats, speedup versus baseline and versus best.
- Tier 2, on by default: an NVML sidecar sampled during the run (SM and memory clocks,
  power, temperature, throttle reasons), achieved occupancy.
- Tier 3, on request or when the strategy asks: Nsight Compute sections (memory
  workload, warp stall reasons, roofline) via `ncu --csv`. Slow, needs the admin
  setting, run only on kept candidates.

## Record format

One JSON object per attempt in `runs/<run-id>/ledger.jsonl`: attempt id, parent,
timestamp, hypothesis, source hash (source itself in `runs/<run-id>/candidates/`),
build status and diagnostics, resource usage, correctness result, timing stats,
baseline stats, metrics by tier, token usage, provider and model, target description
hash, protocol parameters. Errored attempts are recorded with `verified: false`.

## Repo layout

- `autokernel/spec`       frontend
- `autokernel/search`     strategies, ledger, scratchpad, prompt assembly
- `autokernel/providers`  anthropic, openai_compat, mock, human
- `autokernel/backends`   cuda_cpp, triton
- `autokernel/metrics`    nvml sidecar, ncu, ptxas parsing, doctor checks
- `autokernel/cli.py`     `run`, `resume`, `doctor`
- `harness/`              C++/CUDA evaluator process (CMake, own preset)
- `specs/`                example specs, starting with fp32 GEMM
- `tests/`                pytest; includes a suite of known cheats that must be caught
- `runs/`                 output, git-ignored

## Milestones

1. Done 2026-09-29. Greedy loop, `cuda_cpp` backend, fp32 GEMM, Anthropic provider,
   tier-1 feedback, ledger and scratchpad. Runs end to end on a laptop (verified with
   the mock provider; the Anthropic provider is wired but not yet exercised).
2. `triton` backend on the same spec; the shared protocol lives in one place.
3. Anti-cheat hardening and the sandbox; the known-cheat test suite.
4. NVML sidecar and the ncu tier; `doctor`.
5. Idea-branching strategy; ledger digest and summarization.
6. Second provider; budget and resume; a pandas notebook over ledgers.
7. Population strategy; more ops (softmax, layernorm, attention pieces).

## Decisions made in milestone 1

Concrete choices the code now embodies. Change the doc first if you change them.

- **Candidate ABI** (`harness/include/ak_kernel.h`): descriptor-based. The candidate
  defines `extern "C" int ak_kernel(const ak_tensor* inputs, int32_t n_inputs,
  ak_tensor* outputs, int32_t n_outputs, cudaStream_t stream)`. `ak_tensor` carries a
  device pointer, dtype, ndim, shape and element strides. The harness never changes
  per problem; the prompt spells out the concrete shapes. Nonzero return fails the
  attempt. Optional `ak_abi_version()` is checked when present.
- **Harness CLI**: `harness describe` prints device properties as JSON (via
  `cudaDeviceGetAttribute`, since CUDA 13 dropped the clock fields from
  `cudaDeviceProp`). `harness eval` takes the library, `--check-in/--check-out` and
  `--time-in/--time-out` tensor files (.npy), warmup, trials and flush settings, and
  writes a result JSON. Distinct exit codes: 2 load, 3 runtime, 4 file I/O.
- **Protocol constants**: outputs are memset to 0xFF bytes (NaN for floats) before the
  correctness launch, before each warmup and before each trial. L2 flush is a memset
  of max(2 x L2, 32 MB) inside the trial loop, outside the timed region. Timing outputs
  are downloaded after the last trial and verified like the correctness outputs.
- **Seeds**: `base = protocol.seed * 1000003`; baseline inputs use `base`, attempt `i`
  uses `base + 2i` for correctness and `base + 2i + 1` for timing. Recorded per attempt.
- **Correctness metric**: per element `|out - ref| <= atol + rtol * |ref|`, computed in
  float64. Feedback reports the max absolute error and the worst element's error as a
  percentage of its tolerance (headroom), not a relative error, which is meaningless
  near zero.
- **Speedup convention**: `baseline_median / candidate_median`; above 1.0 is faster
  than the baseline. Same for "versus best".
- **Reply format**: `<hypothesis>` block, `<scratchpad>` block, then one fenced code
  block with the whole file. The parser strips the two tagged sections before looking
  for code, takes the longest block with a matching language tag, and records parsing
  problems on the attempt. Each step is one fresh API call; all state is in the prompt.
- **Run directory** (`runs/<run-id>/`): `problem.json`, `target.json`, `baseline.json`,
  `ledger.jsonl`, `scratchpad.md`, `candidates/NNN.cu`, `prompts/NNN_prompt.md` and
  `NNN_reply.md`, `attempts/NNN/` (build.log, run.log, result.json), `best.cu`,
  `best.json`, `summary.json`. Tensor files are deleted after each evaluation unless
  `protocol.keep_tensors` is set. A run resumes from its ledger and scratchpad.
- **Baseline**: the PyTorch reference timed with the same trials and L2 flush, at least
  10 warmup launches, 2 rounds in each of 3 fresh processes; the fastest round's median
  is the bar. Added after cuBLAS SGEMM was seen to run at either 0.34 or 0.37 ms
  depending on the process (every round within a process agreed), while the candidate
  held steady, which turned a real 3% win into an apparent 10%.
- **Thinking tokens**: Claude 5 models think adaptively on hard prompts and those tokens
  count against `max_output_tokens` (a 16k budget cut a reply off mid-code). The budget
  defaults to 32k; thinking blocks, when returned, are saved as `prompts/NNN_thinking.md`
  and the block types land in the attempt's usage record.
- **Providers**: `anthropic`, `openai` (any OpenAI-compatible endpoint, untested),
  `mock` (a directory of reply files), `human` (prompt to a file, reply from a file).
- **CLI**: `autokernel run|eval|prompt|describe|doctor`. `eval` runs one kernel file
  through the full protocol with no model, which is the plain benchmarker use.

## Open questions

- How much of the scratchpad the model may rewrite per step, and whether the system
  should append a locked "measured" line under each idea.
- Whether precision is only a constraint, or a goal the model may trade within bounds.
