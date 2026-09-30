// ak_kernel.h: the contract between autokernel's harness and a candidate kernel.
//
// A candidate is a single translation unit that includes this header and defines
// ak_kernel(). The harness loads the compiled shared library, fills the tensors, and
// calls ak_kernel() on a stream it owns. The candidate enqueues work on that stream
// and returns; the harness measures and verifies. See docs/ARCHITECTURE.md.
#ifndef AK_KERNEL_H
#define AK_KERNEL_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AK_ABI_VERSION 1
#define AK_MAX_DIMS 8

typedef enum ak_dtype {
    AK_F32 = 0,
    AK_F64 = 1,
    AK_F16 = 2,
    AK_BF16 = 3,
    AK_I32 = 4,
    AK_I64 = 5,
    AK_U8 = 6,
    AK_I8 = 7,
    AK_BOOL = 8
} ak_dtype;

typedef struct ak_tensor {
    void* data;                    // device pointer, 256-byte aligned
    int32_t dtype;                 // an ak_dtype value
    int32_t ndim;                  // 0..AK_MAX_DIMS
    int64_t shape[AK_MAX_DIMS];    // shape[0] is the outermost dimension
    int64_t strides[AK_MAX_DIMS];  // in elements; row-major contiguous
} ak_tensor;

// Defined by the candidate. Rules:
//  - Enqueue all work on `stream`. Do not synchronize the device or other streams.
//  - Write every element of every output on every call. Outputs are filled with NaN
//    bytes before each call and never carry values between calls.
//  - Read only the inputs. Do not read the outputs before writing them.
//  - Do not keep state between calls that changes the result. Scratch memory
//    allocated inside the call is allowed and is counted in the measured time.
//  - Return 0 on success. Any other value marks the attempt as failed.
int ak_kernel(const ak_tensor* inputs, int32_t n_inputs,
              ak_tensor* outputs, int32_t n_outputs,
              cudaStream_t stream);

static inline int64_t ak_numel(const ak_tensor* t) {
    int64_t n = 1;
    for (int32_t i = 0; i < t->ndim; ++i) n *= t->shape[i];
    return n;
}

#ifdef __cplusplus
}
#endif
#endif  // AK_KERNEL_H
