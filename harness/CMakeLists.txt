# Evaluator process for the cuda_cpp backend (docs/ARCHITECTURE.md, "Backends").
# Python compiles a candidate into a shared library and runs this executable in a
# subprocess; it loads the library, runs the evaluation protocol, and writes outputs
# and timings back. Stub until milestone 1.
add_executable(harness src/main.cu)
target_compile_options(harness PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(harness PRIVATE CUDA::cudart CUDA::nvml ${CMAKE_DL_LIBS})
