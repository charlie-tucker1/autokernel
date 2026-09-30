"""You are the model. The prompt is written to a file; you write the reply to another
file in the same format a model would use. Doubles as a plain benchmarker for
hand-written kernels, with the same ledger and feedback as a model run."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .base import Provider, Reply, Usage


class HumanProvider(Provider):
    name = "human"
    model = "human"

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir) / "human"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def complete(self, system: str, user: str, max_tokens: int) -> Reply:
        self.n += 1
        prompt_path = self.dir / f"prompt_{self.n:03d}.md"
        reply_path = self.dir / f"reply_{self.n:03d}.md"
        prompt_path.write_text(f"{system}\n\n---\n\n{user}\n")
        print(f"\n[human] prompt written to {prompt_path}", flush=True)
        print(f"[human] write your reply, in the format the prompt asks for, to {reply_path}", flush=True)
        if sys.stdin.isatty():
            input("[human] press Enter once the reply file is saved... ")
        while not reply_path.is_file():
            time.sleep(1.0)
        return Reply(text=reply_path.read_text(), usage=Usage(), model="human", stop_reason="end_turn")
