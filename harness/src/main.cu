// Evaluator process for the cuda_cpp backend. See docs/ARCHITECTURE.md, "Backends" and
// "Evaluation protocol". Milestone 1 replaces this stub with: dlopen the candidate
// library, read inputs, run the protocol, write outputs and timings.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

int main() {
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        std::fprintf(stderr, "harness: no usable CUDA device\n");
        return EXIT_FAILURE;
    }
    std::printf("harness stub: %s sm_%d%d, %d SMs\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
    return EXIT_SUCCESS;
}
