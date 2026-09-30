"""cuda_cpp backend: nvcc compiles the candidate into a shared library, and the C++
harness in harness/ loads it in a subprocess, runs the protocol, and writes results
back. Tensors travel as .npy files. The orchestrator process never loads candidate
code."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import torch

from ..env import REPO_ROOT
from ..metrics.ptxas import parse_ptxas
from ..spec.problem import Problem
from .base import BackendError, BuildResult, EvalCase, EvalResult
from .protocol import TimingStats, compare_outputs, load_npy, nan_filled, save_npy

HOST_COMPILER_CANDIDATES = [
    Path.home() / "miniforge3/envs/cuda-gcc15/bin/x86_64-conda-linux-gnu-g++",
    Path("/usr/bin/g++-15"),
]


def find_nvcc() -> Optional[Path]:
    for base in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")):
        if base and (Path(base) / "bin/nvcc").is_file():
            return Path(base) / "bin/nvcc"
    found = shutil.which("nvcc")
    if found:
        return Path(found)
    default = Path("/usr/local/cuda/bin/nvcc")
    return default if default.is_file() else None


def find_host_compiler() -> Optional[Path]:
    env = os.environ.get("CUDAHOSTCXX")
    if env and Path(env).is_file():
        return Path(env)
    for cand in HOST_COMPILER_CANDIDATES:
        if cand.is_file():
            return cand
    return None


def find_harness(override: Optional[str] = None) -> Optional[Path]:
    for cand in (override, os.environ.get("AUTOKERNEL_HARNESS")):
        if cand and Path(cand).is_file():
            return Path(cand)
    found = [p for p in list(REPO_ROOT.glob("build/*/harness/harness")) +
             list(REPO_ROOT.glob("cmake-build-*/harness/harness"))
             if p.is_file() and os.access(p, os.X_OK)]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def nvcc_version(nvcc: Path) -> str:
    try:
        out = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    m = re.search(r"V(\d+\.\d+\.\d+)", out)
    return m.group(1) if m else "unknown"


class CudaCppBackend:
    name = "cuda_cpp"
    source_extension = ".cu"
    code_fence = "cuda"

    def __init__(self, problem: Problem):
        self.problem = problem
        self.device_index = problem.target.device
        self.device = torch.device("cuda", self.device_index)
        self.nvcc = find_nvcc()
        self.host_compiler = find_host_compiler()
        self.harness = find_harness(problem.target.harness)
        self.include_dir = REPO_ROOT / "harness" / "include"
        self._target: Optional[dict] = None
        missing = []
        if not self.nvcc:
            missing.append("nvcc (set CUDA_HOME or put nvcc on PATH)")
        if not self.harness:
            missing.append("harness executable (build it: cmake --preset laptop && cmake --build --preset laptop)")
        if missing:
            raise BackendError("cuda_cpp backend cannot start: " + "; ".join(missing))

    # ------------------------------------------------------------ description

    def describe_target(self) -> dict:
        if self._target is None:
            proc = subprocess.run([str(self.harness), "describe", "--device", str(self.device_index)],
                                  capture_output=True, text=True, timeout=60)
            if proc.returncode != 0:
                raise BackendError(f"harness describe failed: {proc.stderr.strip()}")
            t = json.loads(proc.stdout)
            if self.problem.target.arch:
                t["arch"] = self.problem.target.arch
            t["toolkit"] = nvcc_version(self.nvcc)
            t["host_compiler"] = str(self.host_compiler) if self.host_compiler else None
            self._target = t
        return self._target

    @property
    def arch(self) -> str:
        return self.describe_target()["arch"]

    def flush_bytes(self) -> int:
        if not self.problem.protocol.flush_l2:
            return 0
        return max(2 * int(self.describe_target()["l2_bytes"]), 32 << 20)

    def target_text(self) -> str:
        t = self.describe_target()
        return "\n".join([
            f"- GPU: {t['name']} ({t['arch']}, compute capability {t['cc_major']}.{t['cc_minor']})",
            f"- {t['sm_count']} SMs; up to {t['max_threads_per_sm']} threads and {t['max_blocks_per_sm']} "
            f"blocks per SM; warp size {t['warp_size']}; max {t['max_threads_per_block']} threads per block",
            f"- Registers: {t['regs_per_sm']} per SM, {t['regs_per_block']} max per block",
            f"- Shared memory: {t['smem_per_sm'] // 1024} KB per SM; {t['smem_per_block'] // 1024} KB per block "
            f"by default, {t['smem_per_block_optin'] // 1024} KB per block after cudaFuncSetAttribute opt-in",
            f"- L2 cache {t['l2_bytes'] // (1 << 20)} MB; device memory {t['total_mem_bytes'] / 2**30:.1f} GiB, "
            f"{t['mem_bus_width_bits']}-bit bus, nominal {t['nominal_mem_bw_gbs']:.0f} GB/s",
            f"- Nominal clocks: SM {t['sm_clock_khz'] / 1e6:.2f} GHz, memory {t['mem_clock_khz'] / 1e6:.2f} GHz "
            f"(a laptop part; sustained clocks are lower under power capping)",
            f"- CUDA toolkit {t['toolkit']}, driver API {t['driver_version']}",
        ])

    def compile_command(self, src: Path, out: Path) -> list[str]:
        cmd = [str(self.nvcc), "-O3", "-std=c++20", f"-arch={self.arch}", "-Xcompiler", "-fPIC", "-shared",
               "-lineinfo", "-Xptxas", "-v", "-Xlinker", "--no-undefined", "-I", str(self.include_dir)]
        if self.host_compiler:
            cmd += ["-ccbin", str(self.host_compiler)]
        return cmd + [str(src), "-o", str(out)]

    def contract_text(self) -> str:
        header = (self.include_dir / "ak_kernel.h").read_text()
        cmd = " ".join(self.compile_command(Path("candidate.cu"), Path("candidate.so")))
        cmd = cmd.replace(str(Path.home()), "~")
        return (
            "Your candidate is one CUDA C++ file. It must `#include \"ak_kernel.h\"` and define\n"
            "`extern \"C\" int ak_kernel(...)` exactly as declared below. The harness passes the inputs\n"
            "and outputs in the order listed under Computation. Only CUDA toolkit headers are\n"
            "available: no cuBLAS, CUTLASS, Thrust device algorithms that need linking, or other\n"
            "libraries. Unresolved symbols fail the build.\n\n"
            f"Compile command:\n    {cmd}\n\n"
            f"Contents of ak_kernel.h:\n```c\n{header}```"
        )

    def measurement_text(self, problem: Problem) -> str:
        p = problem.protocol
        flush = (f"an L2 flush (a {self.flush_bytes() // (1 << 20)} MB memset)" if p.flush_l2 else "no L2 flush")
        return (
            "- One correctness launch on input set A. Every element of every output is compared to the reference.\n"
            f"- Then {p.warmup} warmup launches and {p.trials} timed launches on a different input set B. Before\n"
            f"  each timed launch the outputs are filled with NaN bytes and {flush} runs. The time between\n"
            "  CUDA events recorded around your ak_kernel call on the harness's stream is what is measured;\n"
            "  the median over trials is the score. Set B outputs are verified too.\n"
            f"- A nonzero return, any CUDA error, or exceeding the {p.run_timeout_s:.0f} s run timeout fails the attempt."
        )

    # ------------------------------------------------------------ build / run

    def build(self, source: str, workdir: Path) -> BuildResult:
        workdir.mkdir(parents=True, exist_ok=True)
        src = workdir / "candidate.cu"
        so = workdir / "candidate.so"
        src.write_text(source)
        cmd = self.compile_command(src, so)
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir,
                                  timeout=self.problem.protocol.compile_timeout_s)
            out = proc.stdout + proc.stderr
            ok = proc.returncode == 0 and so.is_file()
        except subprocess.TimeoutExpired:
            out = f"compile timed out after {self.problem.protocol.compile_timeout_s:.0f} s"
            ok = False
        seconds = time.time() - t0
        (workdir / "build.log").write_text(out)
        return BuildResult(ok=ok, diagnostics=out, artifact=so if ok else None,
                           resources=parse_ptxas(out) if ok else {}, seconds=seconds, command=cmd)

    def run(self, build: BuildResult, case: EvalCase, workdir: Path) -> EvalResult:
        p = self.problem.protocol
        comp = self.problem.computation
        tdir = workdir / "tensors"
        tdir.mkdir(parents=True, exist_ok=True)
        args: list[str] = []
        for prefix, inputs in (("check", case.check_inputs), ("time", case.time_inputs)):
            for spec in comp.inputs:
                path = tdir / f"{prefix}_in_{spec.name}.npy"
                save_npy(path, inputs[spec.name])
                args += [f"--{prefix}-in", f"{spec.name}={path}"]
            for spec in comp.outputs:
                path = tdir / f"{prefix}_out_{spec.name}.npy"
                save_npy(path, nan_filled(spec))
                args += [f"--{prefix}-out", f"{spec.name}={path}"]
        result_path = workdir / "result.json"
        cmd = [str(self.harness), "eval", "--lib", str(build.artifact), "--result", str(result_path),
               "--device", str(self.device_index), "--warmup", str(p.warmup), "--trials", str(p.trials),
               "--flush-l2", "1" if p.flush_l2 else "0", *args]
        ev = EvalResult(ok=False)
        t0 = time.time()
        try:
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir, timeout=p.run_timeout_s)
                stderr, returncode, timed_out = proc.stderr, proc.returncode, False
            except subprocess.TimeoutExpired as e:
                err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
                stderr, returncode, timed_out = err, None, True
            ev.seconds = time.time() - t0
            (workdir / "run.log").write_text(stderr or "")
            ev.raw = json.loads(result_path.read_text()) if result_path.is_file() else {}
            if timed_out:
                ev.stage = "timeout"
                ev.error = f"the candidate exceeded the {p.run_timeout_s:.0f} s run timeout and was killed"
                return ev
            if not ev.raw:
                ev.stage = "harness"
                ev.error = f"harness exited {returncode} without a result: {(stderr or '').strip()[-1500:]}"
                return ev
            if ev.raw.get("status") != "ok":
                ev.stage = ev.raw.get("stage", "")
                ev.error = ev.raw.get("error", "")
                return ev
            ev.ok = True
            ev.stage = "done"
            ev.trial_ms = list(ev.raw.get("trial_ms", []))
            ev.check_launch_ms = ev.raw.get("check_launch_ms")
            got_check = {s.name: load_npy(tdir / f"check_out_{s.name}.npy", s.dtype, self.device) for s in comp.outputs}
            got_time = {s.name: load_npy(tdir / f"time_out_{s.name}.npy", s.dtype, self.device) for s in comp.outputs}
            ev.checks = compare_outputs(got_check, case.check_refs, self.problem.precision)
            ev.timing_checks = compare_outputs(got_time, case.time_refs, self.problem.precision)
            if ev.trial_ms:
                ev.timing = TimingStats.from_trials(ev.trial_ms)
            return ev
        finally:
            if not p.keep_tensors:
                shutil.rmtree(tdir, ignore_errors=True)
