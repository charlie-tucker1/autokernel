"""Middle. Candidate, Attempt, Ledger, Scratchpad, Strategy, prompt assembly, budget.

Strategies produce Candidates and observe Attempts. They never touch a backend or a
provider directly; the loop in `run.py` (milestone 1) wires the three together.
"""
