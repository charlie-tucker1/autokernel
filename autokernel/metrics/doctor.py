"""`autokernel doctor`: what this machine can build, run and measure, and what the
human has to do about anything missing. Never escalates privileges."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..env import REPO_ROOT


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    hint: str = ""


def _run(cmd: list[str], timeout: float = 20) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return -1, str(e)


def run_checks() -> list[Check]:
    from ..backends.cuda_cpp import find_harness, find_host_compiler, find_nvcc, nvcc_version

    checks: list[Check] = []

    try:
        import torch
        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "no device"
        checks.append(Check("torch", cuda, f"torch {torch.__version__}, {name}",
                            "" if cuda else "install a CUDA build of torch and check the driver"))
    except ImportError:
        checks.append(Check("torch", False, "not importable", "pip install torch (CUDA build)"))
    try:
        import triton
        checks.append(Check("triton", True, f"triton {triton.__version__} (milestone 2 backend)"))
    except ImportError:
        checks.append(Check("triton", False, "not importable", "pip install triton; needed for the triton backend only"))

    nvcc = find_nvcc()
    checks.append(Check("nvcc", nvcc is not None, f"{nvcc} ({nvcc_version(nvcc)})" if nvcc else "not found",
                        "" if nvcc else "install a CUDA toolkit and set CUDA_HOME or PATH"))
    host = find_host_compiler()
    checks.append(Check("host compiler", host is not None, str(host) if host else "none found (nvcc will use the system c++)",
                        "" if host else "nvcc 13 needs GCC <= 15: set CUDAHOSTCXX or install gcc15-c++"))
    harness = find_harness()
    if harness:
        rc, out = _run([str(harness), "describe"])
        checks.append(Check("harness", rc == 0, str(harness) if rc == 0 else f"{harness}: {out[:200]}",
                            "" if rc == 0 else "rebuild: cmake --preset laptop && cmake --build --preset laptop"))
    else:
        checks.append(Check("harness", False, "not built", "cmake --preset laptop && cmake --build --preset laptop"))

    key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    checks.append(Check("ANTHROPIC_API_KEY", key, "set" if key else "not set",
                        "" if key else f"add ANTHROPIC_API_KEY=... to {REPO_ROOT / '.env'} (git-ignored)"))

    ncu = shutil.which("ncu") or (str(Path(nvcc).parent / "ncu") if nvcc and (Path(nvcc).parent / "ncu").is_file() else None)
    if ncu:
        params = Path("/proc/driver/nvidia/params")
        admin_only = None
        if params.is_file():
            m = re.search(r"RmProfilingAdminOnly:\s*(\d)", params.read_text())
            admin_only = m.group(1) == "1" if m else None
        ok = admin_only is False
        detail = f"{ncu}; GPU counters " + ("open to users" if ok else "restricted to root" if admin_only else "permission unknown")
        checks.append(Check("Nsight Compute (tier 3)", ok, detail, "" if ok else
                            "one-time admin step, then reboot: echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' "
                            "| sudo tee /etc/modprobe.d/nvidia-profiling.conf && sudo dracut -f"))
    else:
        checks.append(Check("Nsight Compute (tier 3)", False, "ncu not found", "comes with the CUDA toolkit; add its bin to PATH"))

    smi = shutil.which("nvidia-smi")
    checks.append(Check("nvidia-smi / NVML (tier 2)", smi is not None, smi or "not found"))
    bwrap = shutil.which("bwrap")
    checks.append(Check("bubblewrap sandbox", bwrap is not None, bwrap or "not found",
                        "" if bwrap else "dnf install bubblewrap; candidates then run without network"))
    return checks


def report() -> int:
    checks = run_checks()
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"{'OK     ' if c.ok else 'MISSING'} {c.name.ljust(width)}  {c.detail}")
        if not c.ok and c.hint:
            print(f"        {' ' * width}  -> {c.hint}")
    required = [c for c in checks if c.name in ("torch", "nvcc", "harness")]
    return 0 if all(c.ok for c in required) else 1
