"""Parse `nvcc -Xptxas -v` output into per-kernel resource usage."""

from __future__ import annotations

import re
import shutil
import subprocess

_ENTRY = re.compile(r"Compiling entry function '([^']+)' for '(sm_\d+[a-z]?)'")
_STACK = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
_USED = re.compile(r"Used (\d+) registers")
_SMEM = re.compile(r"(\d+) bytes smem")
_CMEM = re.compile(r"(\d+) bytes cmem\[(\d+)\]")
_BARRIERS = re.compile(r"used (\d+) barriers")


def demangle(names: list[str]) -> dict[str, str]:
    tool = shutil.which("cu++filt") or shutil.which("c++filt")
    if not tool or not names:
        return {n: n for n in names}
    try:
        out = subprocess.run([tool, *names], capture_output=True, text=True, timeout=10).stdout.splitlines()
        if len(out) == len(names):
            return dict(zip(names, [o.strip() for o in out]))
    except (OSError, subprocess.SubprocessError):
        pass
    return {n: n for n in names}


def parse_ptxas(text: str) -> dict:
    """Return {"kernels": [{name, arch, registers, smem_bytes, spill_stores, spill_loads,
    stack_bytes, cmem, barriers}], "warnings": [...]}."""
    kernels: list[dict] = []
    current: dict | None = None
    warnings: list[str] = []
    for line in text.splitlines():
        if "ptxas warning" in line:
            warnings.append(line.strip())
        m = _ENTRY.search(line)
        if m:
            current = {"name": m.group(1), "arch": m.group(2), "registers": None, "smem_bytes": 0,
                       "spill_stores": 0, "spill_loads": 0, "stack_bytes": 0, "cmem": {}, "barriers": None}
            kernels.append(current)
            continue
        if current is None:
            continue
        m = _STACK.search(line)
        if m:
            current["stack_bytes"], current["spill_stores"], current["spill_loads"] = map(int, m.groups())
        m = _USED.search(line)
        if m:
            current["registers"] = int(m.group(1))
        m = _SMEM.search(line)
        if m:
            current["smem_bytes"] = int(m.group(1))
        for m in _CMEM.finditer(line):
            current["cmem"][m.group(2)] = int(m.group(1))
        m = _BARRIERS.search(line)
        if m:
            current["barriers"] = int(m.group(1))
    names = demangle([k["name"] for k in kernels])
    for k in kernels:
        k["mangled"] = k["name"]
        k["name"] = names.get(k["name"], k["name"])
    return {"kernels": kernels, "warnings": warnings}


def resources_summary(res: dict) -> str:
    ks = res.get("kernels") or []
    if not ks:
        return "ptxas: no kernels found in compiler output"
    parts = []
    for k in ks:
        short = k["name"].split("(")[0]
        spill = k["spill_stores"] + k["spill_loads"]
        parts.append(f"`{short}`: {k['registers']} registers, {k['smem_bytes']} B static smem, "
                     f"{spill} B spills" + (f", {k['stack_bytes']} B stack" if k["stack_bytes"] else ""))
    text = "ptxas: " + "; ".join(parts)
    if res.get("warnings"):
        text += "\n" + "\n".join(res["warnings"][:5])
    return text
