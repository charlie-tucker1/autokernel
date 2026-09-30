"""Middle. Candidate, Attempt, Ledger, Scratchpad, Strategy, prompt assembly, budget.

Strategies produce parents and observe attempts. They never touch a backend or a
provider directly; `run.py` wires the three together.
"""

from .ledger import Ledger
from .run import evaluate_candidate, run_search
from .types import Attempt, Candidate

__all__ = ["Attempt", "Candidate", "Ledger", "evaluate_candidate", "run_search"]
