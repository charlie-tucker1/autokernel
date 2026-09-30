"""Load a spec file into a Problem and apply `key.path=value` overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .problem import Problem


def _set_path(tree: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = tree
    for k in keys[:-1]:
        if not isinstance(node.get(k), dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"override must look like section.key=value, got {text!r}")
    key, _, raw = text.partition("=")
    return key.strip(), yaml.safe_load(raw)


def load_problem(path: str | Path, overrides: list[str] | None = None) -> Problem:
    path = Path(path).resolve()
    tree = yaml.safe_load(path.read_text()) or {}
    if not isinstance(tree, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    for text in overrides or []:
        key, value = parse_override(text)
        _set_path(tree, key, value)
    problem = Problem.model_validate(tree)
    problem.spec_path = path
    return problem
