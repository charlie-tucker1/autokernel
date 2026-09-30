"""Repo-root .env loading. Keys never live in specs; they come from the environment,
and the `.env` file (git-ignored) is the reliable way to get them there."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(paths: list[Path] | None = None) -> list[str]:
    """Set variables from the first `.env` found, without overriding existing ones.
    Returns the names that were set."""
    candidates = paths or [Path.cwd() / ".env", REPO_ROOT / ".env"]
    for path in candidates:
        if not path.is_file():
            continue
        names: list[str] = []
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                names.append(key)
        return names
    return []
