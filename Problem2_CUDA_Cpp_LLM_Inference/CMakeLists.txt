cmake_minimum_required(VERSION 3.18)
project(cuda_llm_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Toggle NVIDIA GPU acceleration. Requires the CUDA Toolkit (nvcc) to be
# installed and discoverable by CMake. When OFF (or when no CUDA toolkit is
# found), llama.cpp falls back to a plain CPU build so this project still
# compiles and runs -- just without GPU offload.
option(GGML_CUDA "Build llama.cpp with NVIDIA CUDA GPU acceleration" ON)

include(FetchContent)
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        b4079          # known-good tag; bump if you need newer model support
    GIT_SHALLOW    TRUE
)

# Skip llama.cpp's own examples/tests/server -- we only need the `llama`
# (+ ggml) library target for this minimal single-file demo.
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama_cpp)

add_executable(cuda_llm_infer main.cpp)
target_link_libraries(cuda_llm_infer PRIVATE llama)

if(GGML_CUDA)
    message(STATUS "cuda_llm_infer: building with NVIDIA CUDA GPU offload enabled (GGML_CUDA=ON)")
else()
    message(STATUS "cuda_llm_infer: building CPU-only (GGML_CUDA=OFF)")
endif()
