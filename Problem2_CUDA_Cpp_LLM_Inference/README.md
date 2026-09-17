# Problem 2 — C++ LLM Inference on NVIDIA GPUs via CUDA

A minimal, single-file C++ program ([main.cpp](main.cpp)) that loads a local
**GGUF** LLM (e.g. a quantized Llama 3.2 model) and runs text-generation
inference on an **NVIDIA GPU using CUDA**, built on top of
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `ggml-cuda` backend.

llama.cpp is not vendored in this repo -- [CMakeLists.txt](CMakeLists.txt)
uses CMake's `FetchContent` to clone and build it automatically (pinned to a
known-good tag) as part of the CMake configure step, with `GGML_CUDA=ON` so
its GPU (CUDA) kernels are compiled in.

> **Run it in Google Colab instead:** open [Problem2_Colab.ipynb](Problem2_Colab.ipynb)
> with a T4/GPU runtime (`Runtime > Change runtime type > GPU`) — it builds
> the CMake project with `GGML_CUDA=ON`, downloads a small GGUF model, and
> runs inference on the Colab GPU.

## How it works

1. `llama_backend_init()` initializes ggml/llama.cpp and registers the CUDA
   backend (present because llama.cpp was built with `GGML_CUDA=ON`).
2. `llama_model_load_from_file()` loads the `.gguf` model, offloading
   `n_gpu_layers` transformer layers onto the GPU (`999` = offload
   everything).
3. The prompt is tokenized and fed through the model in one batch
   (`llama_decode`) -- this "prefill" step runs as GPU kernels.
4. A greedy sampler (`llama_sampler_init_greedy`) picks the next token; the
   token is decoded back to text (`llama_token_to_piece`) and printed, then
   fed back into `llama_decode` for the next step, repeating up to
   `n_predict` tokens or until an end-of-generation token is produced.

## Prerequisites

- An NVIDIA GPU with a recent driver installed.
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (provides
  `nvcc`) — version 12.x recommended.
- CMake ≥ 3.18.
- A C++17 compiler: MSVC (Visual Studio 2022 Build Tools) on Windows, or
  gcc/clang on Linux.
- Git (needed by CMake's `FetchContent` to pull llama.cpp).
- A GGUF chat model file — see [models/README.md](models/README.md) for how
  to download one (e.g. `Llama-3.2-1B-Instruct-Q4_K_M.gguf`).

> No NVIDIA GPU available? Configure with `-DGGML_CUDA=OFF` to build and run
> the exact same program on CPU only, for functional testing.

## Build

### Windows (PowerShell, Visual Studio generator)

```powershell
cd Problem2_CUDA_Cpp_LLM_Inference
cmake -B build -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON
cmake --build build --config Release -j
```

### Linux / WSL (gcc/clang + Ninja or Makefiles)

```bash
cd Problem2_CUDA_Cpp_LLM_Inference
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The first configure step downloads llama.cpp source via `FetchContent` and
can take a few minutes; the CUDA kernels then take a while to compile the
first time.

## Run

```powershell
# Windows
.\build\Release\cuda_llm_infer.exe models\Llama-3.2-1B-Instruct-Q4_K_M.gguf "Explain CUDA in one sentence." 128 999

# Linux
./build/cuda_llm_infer models/Llama-3.2-1B-Instruct-Q4_K_M.gguf "Explain CUDA in one sentence." 128 999
```

Arguments: `<model.gguf> "<prompt>" [n_predict=128] [n_gpu_layers=999]`.

- `n_gpu_layers=999` offloads the whole model to VRAM (fastest). Lower it
  (e.g. `20`) if the model doesn't fit in your GPU's VRAM, or set it to `0`
  to force CPU-only execution with the same binary.
- `n_predict` caps how many tokens are generated.

Expected output looks like:

```
Prompt: Explain CUDA in one sentence.

Running on: NVIDIA GPU (CUDA)
Response:
CUDA is NVIDIA's parallel computing platform and API that lets developers
run general-purpose code directly on the GPU's many cores for massive
speedups over CPU-only execution.
```

## Troubleshooting

- **CMake can't find CUDA / `nvcc` not found**: ensure the CUDA Toolkit's
  `bin` directory is on `PATH` and that `nvcc --version` works in the same
  shell you run `cmake` from.
- **`FetchContent` clone fails / no internet**: clone
  `https://github.com/ggml-org/llama.cpp` manually and point CMake at it
  with `-DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=<path-to-clone>` instead of
  letting `FetchContent` download it.
- **Build errors after llama.cpp updates its API**: the `GIT_TAG` in
  [CMakeLists.txt](CMakeLists.txt) pins a known-good commit; if you bump it,
  the exact function names used in [main.cpp](main.cpp) (e.g.
  `llama_model_load_from_file`, `llama_init_from_model`) may need small
  updates to match that version's `llama.h`.
- **Out of VRAM**: lower `n_gpu_layers` so only part of the model is
  offloaded, or use a smaller/more aggressively quantized GGUF file.
