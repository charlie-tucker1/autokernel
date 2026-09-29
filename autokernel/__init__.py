"""autokernel: LLM-driven kernel search with an untrusting evaluator.

Subpackages follow docs/ARCHITECTURE.md: spec (frontend), search (middle),
backends and providers (the two things that vary), metrics (what the machine can tell
us). Nothing in here runs model-written code directly; backends do that in subprocesses.
"""

__version__ = "0.0.1"
