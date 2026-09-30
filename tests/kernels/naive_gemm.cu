// Test fixture: a correct, slow fp32 GEMM against the ak_kernel contract.
// C[M,N] = A[M,K] @ B[K,N], row-major. Used by the harness and backend tests.
#include "ak_kernel.h"

__global__ void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

extern "C" int ak_kernel(const ak_tensor* in, int32_t n_in, ak_tensor* out, int32_t n_out,
                         cudaStream_t stream) {
    if (n_in != 2 || n_out != 1 || in[0].ndim != 2 || in[1].ndim != 2) return 1;
    const int M = int(in[0].shape[0]), K = int(in[0].shape[1]), N = int(in[1].shape[1]);
    const dim3 block(16, 16);
    const dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_naive<<<grid, block, 0, stream>>>(static_cast<const float*>(in[0].data),
                                           static_cast<const float*>(in[1].data),
                                           static_cast<float*>(out[0].data), M, N, K);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
