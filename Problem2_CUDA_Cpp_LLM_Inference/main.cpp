// cuda_llm_infer.cpp
//
// A minimal, single-file C++ program that loads a local GGUF LLM (e.g. a
// quantized Llama 3.2 model) and runs text-generation inference on an
// NVIDIA GPU via CUDA, using llama.cpp's `ggml-cuda` backend.
//
// Build & run instructions: see README.md in this folder.
//
// Usage:
//   cuda_llm_infer <model.gguf> "<prompt>" [n_predict=128] [n_gpu_layers=999]
//
//   n_gpu_layers = 999 offloads every transformer layer to the GPU (fastest,
//   requires enough VRAM). Use a smaller number (e.g. 20) to offload only
//   part of the model if it doesn't fit in VRAM, or 0 to force CPU-only.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string &msg) {
    std::fprintf(stderr, "Error: %s\n", msg.c_str());
    std::exit(1);
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(
            stderr,
            "Usage: %s <model.gguf> \"<prompt>\" [n_predict=128] [n_gpu_layers=999]\n",
            argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string prompt = argv[2];
    const int n_predict = argc > 3 ? std::atoi(argv[3]) : 128;
    const int n_gpu_layers = argc > 4 ? std::atoi(argv[4]) : 999;

    // 1. Initialize the ggml/llama backends. When llama.cpp was built with
    //    GGML_CUDA=ON and an NVIDIA GPU + driver is present, the CUDA
    //    backend is registered automatically and used for any layers we
    //    offload below.
    llama_backend_init();

    // 2. Load the GGUF model, offloading `n_gpu_layers` transformer layers
    //    to the GPU.
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    llama_model *model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fail("failed to load model from '" + model_path + "'");
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);

    // 3. Create an inference context (KV cache size, batch size, ...).
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;

    llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fail("failed to create llama context");
    }

    // 4. Tokenize the prompt. llama_tokenize returns a negative "required
    //    size" if the buffer we pass is too small, so we retry once.
    std::vector<llama_token> tokens(prompt.size() + 8);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                                   tokens.data(), (int32_t)tokens.size(),
                                   /*add_special=*/true, /*parse_special=*/true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                                   tokens.data(), (int32_t)tokens.size(), true, true);
    }
    tokens.resize(n_tokens);

    // 5. Prefill: feed every prompt token through the model in one batch.
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        fail("llama_decode failed while processing the prompt");
    }

    // 6. Greedy-decode one token at a time until n_predict tokens are
    //    generated or the model produces an end-of-generation token.
    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    std::printf("Prompt: %s\n\n", prompt.c_str());
    std::printf("Running on: %s\n", n_gpu_layers > 0 ? "NVIDIA GPU (CUDA)" : "CPU");
    std::printf("Response:\n");

    for (int i = 0; i < n_predict; ++i) {
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        char piece[256];
        int n = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, true);
        if (n > 0) {
            std::fwrite(piece, 1, n, stdout);
            std::fflush(stdout);
        }

        llama_batch next_batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx, next_batch) != 0) {
            fail("llama_decode failed during generation");
        }
    }
    std::printf("\n");

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
