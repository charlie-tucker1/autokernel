"""The model's own notes file. It rewrites the whole thing each step; the system only
stores it, caps its size, and shows it back."""

from __future__ import annotations

from pathlib import Path

DEFAULT = "No notes yet: this is the first attempt.\n"
TRUNCATION_MARK = "[notes cut here by the system: over the size limit]"


def cap(text: str, max_lines: int, max_chars: int) -> tuple[str, bool]:
    """Cut `text` to the limits. Returns the text and whether anything was cut."""
    text = text.rstrip()
    lines = text.splitlines()
    cut = False
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines]
        cut = True
    text = "\n".join(lines)
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip()
        cut = True
    if cut:
        text += "\n" + TRUNCATION_MARK
    return text + "\n", cut


class Scratchpad:
    def __init__(self, path: Path, max_lines: int = 0, max_chars: int = 0):
        self.path = Path(path)
        self.max_lines = max_lines
        self.max_chars = max_chars

    def read(self) -> str:
        return self.path.read_text() if self.path.is_file() else DEFAULT

    def write(self, text: str) -> bool:
        """Store the notes. Returns True if they were cut to fit the limits."""
        text, cut = cap(text, self.max_lines, self.max_chars)
        self.path.write_text(text)
        return cut
