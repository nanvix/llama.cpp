/*
 * nanvix_simple.cpp - Minimal llama.cpp inference for Nanvix
 *
 * Copyright(c) The Maintainers of Nanvix.
 * Licensed under the MIT License.
 *
 * Based on examples/simple/simple.cpp from llama.cpp.
 * Loads a GGUF model and generates text from a prompt.
 *
 * Usage:
 *   nanvixd.elf -- ./llama_simple.elf -m model.gguf [-n 32] [prompt]
 */

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/*
 * When NANVIX_MEMFS is defined, use the in-memory FAT32 filesystem to load
 * the model from RAMFS MMIO instead of the slow virtio-fs path. The model
 * file is packaged into a FAT32 image and passed to nanvixd via -ramfs.
 * The memfs layer intercepts fopen/fread/fstat calls and serves them
 * directly from memory, eliminating the IPC/VM-exit overhead.
 */
#ifdef NANVIX_MEMFS
extern "C" {
    int memfs_init_from_ramfs(const char *mount_path);
    long long memfs_file_size(const char *path);
}
#endif /* NANVIX_MEMFS */

/*
 * On i686 without SSE, all floating-point uses the x87 FPU which defaults
 * to 80-bit extended precision internally. This causes small rounding
 * differences compared to SSE2's strict 32/64-bit float operations.
 * Over 28+ transformer layers, these differences accumulate and can change
 * which token has the highest logit, causing generation to diverge
 * (e.g., getting stuck in a loop producing the same token).
 *
 * Setting the x87 precision control to double (64-bit) reduces the
 * divergence significantly. This is the same setting used by most
 * Linux distributions for compatibility.
 */
#if defined(__i386__) && !defined(__SSE2__)
static void set_x87_double_precision(void) {
    unsigned short cw;
    __asm__ volatile ("fnstcw %0" : "=m" (cw));
    cw = (cw & ~0x0300) | 0x0200;  // precision control: 10 = double (64-bit)
    __asm__ volatile ("fldcw %0" : : "m" (cw));
}
#else
static void set_x87_double_precision(void) { /* no-op on SSE2+ targets */ }
#endif

static void print_usage(const char * progname) {
    fprintf(stderr, "\nUsage: %s -m model.gguf [-n n_predict] [-t n_threads] [prompt]\n\n", progname);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m MODEL    Path to GGUF model file (required)\n");
    fprintf(stderr, "  -n N        Number of tokens to predict (default: 32)\n");
    fprintf(stderr, "  -t N        Number of threads (default: 1)\n");
    fprintf(stderr, "  -r          Send raw prompt without chat template\n");
    fprintf(stderr, "  prompt      Text prompt (default: \"Hello\")\n\n");
}

int main(int argc, char ** argv) {
    // Reduce x87 FPU extended precision to match SSE2 behavior
    set_x87_double_precision();

    std::string model_path;
    std::string prompt = "Hello";
    int n_predict = 32;
    int n_threads = 1;
    bool raw_prompt = false;

    // Parse arguments
    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
                model_path = argv[++i];
            } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n_predict = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
                n_threads = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-r") == 0) {
                raw_prompt = true;
            } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
                print_usage(argv[0]);
                return 0;
            } else {
                break;  // rest is prompt
            }
        }
        if (model_path.empty()) {
            print_usage(argv[0]);
            return 1;
        }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    // Wrap prompt in Qwen3 chat template unless -r (raw) flag is given.
    // Without the chat template, the base model does raw text completion
    // which produces low-quality output (e.g., repeated tokens).
    if (!raw_prompt) {
        prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    }

    fprintf(stderr, "llama.cpp on Nanvix\n");
    fprintf(stderr, "  model:   %s\n", model_path.c_str());
    fprintf(stderr, "  prompt:  \"%s\"\n", prompt.c_str());
    fprintf(stderr, "  predict: %d tokens\n", n_predict);
    fprintf(stderr, "  threads: %d\n\n", n_threads);

    // Initialize backend
    fprintf(stderr, "[PHASE] backend_init\n");
    llama_backend_init();

    // NOTE: ggml_backend_load_all() is NOT called because:
    // 1. CPU backend is statically linked (GGML_USE_CPU) and auto-registered
    // 2. The dynamic backend loader tries filesystem ops that fail on Nanvix

#ifdef NANVIX_MEMFS
    // Initialize in-memory filesystem from RAMFS MMIO region.
    // The model FAT32 image was passed to nanvixd via the -ramfs flag.
    // Mount it at /model so the model file is accessible at /model/<filename>.
    {
        fprintf(stderr, "[PHASE] memfs_init\n");
        if (memfs_init_from_ramfs("/model") != 0) {
            fprintf(stderr, "Warning: memfs init failed, falling back to virtio-fs\n");
        } else {
            // Rewrite model path to point to the memfs mount.
            // Extract just the filename from the model path.
            std::string filename = model_path;
            size_t last_slash = filename.find_last_of('/');
            if (last_slash != std::string::npos) {
                filename = filename.substr(last_slash + 1);
            }
            std::string memfs_path = "/model/" + filename;

            long long fsize = memfs_file_size(memfs_path.c_str());
            if (fsize > 0) {
                fprintf(stderr, "memfs: model at %s (%lld bytes)\n",
                        memfs_path.c_str(), fsize);
                model_path = memfs_path;
            } else {
                fprintf(stderr, "Warning: model not found in memfs at %s\n",
                        memfs_path.c_str());
            }
        }
    }
#endif /* NANVIX_MEMFS */

    // Verify CPU backend is available
    {
        size_t n = ggml_backend_reg_count();
        fprintf(stderr, "Registered backends: %d\n", (int)n);
        for (size_t i = 0; i < n; i++) {
            ggml_backend_reg_t reg = ggml_backend_reg_get(i);
            const char *name = ggml_backend_reg_name(reg);
            fprintf(stderr, "  backend %d: %s (%d devices)\n",
                    (int)i, name ? name : "(null)", (int)ggml_backend_reg_dev_count(reg));
        }
    }

    // Load model (no mmap, no GPU, CPU only)
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.use_mmap  = false;
    model_params.use_mlock = false;

    fprintf(stderr, "[PHASE] model_load\n");
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Error: failed to load model from '%s'\n", model_path.c_str());
        return 1;
    }
    fprintf(stderr, "[PHASE] model_loaded\n");

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Tokenize prompt
    fprintf(stderr, "[PHASE] tokenize\n");
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "Error: failed to tokenize prompt\n");
        llama_model_free(model);
        return 1;
    }
    fprintf(stderr, "Prompt tokenized: %d tokens\n", n_prompt);

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx    = n_prompt + n_predict;
    ctx_params.n_batch  = n_prompt;
    ctx_params.n_threads       = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.no_perf  = false;

    fprintf(stderr, "[PHASE] ctx_create\n");
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // Create sampler (greedy)
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // Print prompt (raw prompt text only, not chat template markup)
    fprintf(stderr, "\n--- output ---\n");

    // Decode prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    if (llama_model_has_encoder(model)) {
        llama_encode(ctx, batch);
        llama_token dec_start = llama_model_decoder_start_token(model);
        if (dec_start == LLAMA_TOKEN_NULL) {
            dec_start = llama_vocab_bos(vocab);
        }
        batch = llama_batch_get_one(&dec_start, 1);
    }

    // Generate tokens
    fprintf(stderr, "[PHASE] prompt_eval\n");
    const int64_t t_start = ggml_time_us();
    int n_decoded = 0;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "\nError: decode failed at pos %d\n", n_pos);
            break;
        }

        // Mark transition from prompt eval to generation after first decode
        if (n_decoded == 0) {
            fprintf(stderr, "[PHASE] generation\n");
        }

        n_pos += batch.n_tokens;

        // Sample next token
        llama_token new_id = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_id)) {
            break;
        }

        // Print token
        char buf[256];
        int n = llama_token_to_piece(vocab, new_id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            fwrite(buf, 1, n, stdout);
        }

        batch = llama_batch_get_one(&new_id, 1);
        n_decoded++;
    }

    printf("\n");
    fflush(stdout);
    const int64_t t_end = ggml_time_us();
    const double t_sec = (t_end - t_start) / 1000000.0;

    fprintf(stderr, "[PHASE] done\n");
    fprintf(stderr, "decoded %d tokens in %.2f s (%.2f t/s)\n", n_decoded, t_sec,
            t_sec > 0 ? n_decoded / t_sec : 0.0);

    llama_perf_context_print(ctx);

    // Cleanup
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
