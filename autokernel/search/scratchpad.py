"""The model's own notes file. It rewrites the whole thing each step; the system only
stores it and shows it back."""

from __future__ import annotations

from pathlib import Path

DEFAULT = "No notes yet: this is the first attempt.\n"


class Scratchpad:
    def __init__(self, path: Path):
        self.path = Path(path)

    def read(self) -> str:
        return self.path.read_text() if self.path.is_file() else DEFAULT

    def write(self, text: str) -> None:
        self.path.write_text(text.rstrip() + "\n")
