"""Backends: one per (language, hardware family) pair.

Each backend knows how to describe the target, state the candidate contract, build a
candidate, and run the shared evaluation protocol in a subprocess. cuda_cpp uses the
C++ harness in harness/; triton runs a Python subprocess with torch.
"""
