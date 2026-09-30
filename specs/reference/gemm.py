import torch


def gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """C = A @ B in fp32. torch keeps fp32 matmul at full precision by default
    (no TF32), so this is the true fp32 answer and the baseline is cuBLAS SGEMM."""
    return torch.matmul(A, B)
