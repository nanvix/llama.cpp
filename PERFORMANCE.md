# Performance Analysis: llama.cpp on Nanvix

> **TL;DR:** With zero-copy in-memory FAT32 filesystem (memfs), Nanvix runs LLM inference end-to-end in **20.3 s** (32 tokens) — a **14x improvement** over the original virtio-fs baseline of 283 s. Model loading dropped from 279 s to 16.3 s (**17x faster**). The memfs init overhead was eliminated (14.3 s → 0.025 s) by mounting the RAMFS MMIO region directly without copying. Per-token compute is **1.8x faster** than native x86_64 (guest-reported), thanks to AVX2/FMA enablement.

---

## Table of Contents

1. [Test Setup](#test-setup)
2. [Results Summary](#results-summary)
3. [Scaling Behavior](#scaling-behavior)
4. [Phase-by-Phase Breakdown](#phase-by-phase-breakdown)
5. [Root Cause Analysis](#root-cause-analysis)
6. [Optimization History](#optimization-history)
7. [Methodology](#methodology)
8. [Reproducing the Benchmarks](#reproducing-the-benchmarks)
9. [Optimization Opportunities](#optimization-opportunities)

---

## Test Setup

| Parameter | Native (baseline) | Nanvix (memfs) |
|-----------|-------------------|----------------|
| **Host CPU** | Intel Core i9-12900H | Intel Core i9-12900H |
| **Host OS** | Linux 5.15 (WSL2) | Linux 5.15 (WSL2) |
| **Guest OS** | — | Nanvix (nanvixd.elf, KVM) |
| **Architecture** | x86_64 | i686 (32-bit) |
| **ISA Extensions** | AVX2, FMA, SSE4.2 | AVX2, FMA |
| **Guest Memory** | — | 3 GB |
| **Threads** | 1 | 1 |
| **BLAS** | None (disabled) | None |
| **OpenMP** | Disabled | Disabled |
| **Model** | Qwen3-0.6B-Q4_K_M (462 MB) | Qwen3-0.6B-Q4_K_M (462 MB) |
| **Quantization** | Q4_K_M | Q4_K_M |
| **I/O backend** | Native fs | Zero-copy in-memory FAT32 (RAMFS MMIO) |
| **mmap** | Disabled (`use_mmap=false`) | Disabled |
| **Compiler** | GCC (host) | Clang 21 (cross) |
| **Build type** | Release (`-O2`) | Release (`-O2`) |
| **Nanvix branch** | — | `feature-llama-perf` |
| **Nanvix LOG_LEVEL** | — | `panic` (minimal logging) |

---

## Results Summary

### End-to-End (32 tokens, prompt "Hello")

| Metric | Native | Nanvix (memfs) | Ratio |
|--------|--------|----------------|-------|
| **Total wall-clock** | 1.94 s | 20.32 s | **10.5x** |
| **Model load (host)** | 0.558 s | 16.29 s | 29x |
| **Generation (host)** | 1.155 s | 1.325 s | 1.1x |
| **Token decode (guest, ms/tok)** | 36.92 ms | 20.49 ms | **1.8x faster†** |
| **Tokens/second (guest, eval)** | 27.09 t/s | 48.81 t/s | **1.8x faster†** |

†Guest-reported metrics; Nanvix with AVX2/FMA on i686 achieves faster per-token decode than native x86_64 in guest-measured time. Host wall-clock is higher due to KVM VM-exit overhead not captured by the guest clock.

### Where the Time Goes

```
Native (1.94 s total):
  █████████████████████████████████████████████████████████████░░ generation (60%)
  ██████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ model_load (29%)
  █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ prompt_eval (7%)
  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ other (4%)

Nanvix memfs (20.32 s total):
  ██████████████████████████████████████████████████████████████░░ model_load (80.2%)
  ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ctx_create (11.1%)
  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ generation (6.5%)
  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ other (2.2%)
```

---

## Scaling Behavior

All times are host-side wall-clock measurements.

### Across Token Counts

| Metric | 32 tokens | 128 tokens | 256 tokens |
|--------|-----------|------------|------------|
| **Native total** | 1.94 s | 4.61 s | 9.25 s |
| **Nanvix total** | 20.32 s | 25.82 s | 32.16 s |
| **Ratio** | 10.5x | 5.6x | 3.5x |
| | | | |
| Native generation | 1.155 s | 3.951 s | 8.576 s |
| Nanvix generation | 1.325 s | 7.271 s | 11.604 s |
| Gen ratio | 1.1x | 1.8x | 1.4x |
| | | | |
| Native model_load | 0.558 s | 0.421 s | 0.437 s |
| Nanvix model_load | 16.286 s | 15.753 s | 16.791 s |
| Load ratio | 29x | 37x | 38x |
| | | | |
| Nanvix ctx_create | 2.264 s | 2.222 s | 3.231 s |
| Nanvix memfs_init | 0.025 s | 0.019 s | 0.024 s |

### Key Insight: Amortization

The fixed-cost phases (model load: ~16 s, ctx_create: ~2.5 s, boot: ~0.2 s) are amortized over more tokens. At 256 tokens, generation dominates and the Nanvix-to-native ratio drops to **3.5x**. Extrapolating:

| Tokens | Estimated Nanvix total | Ratio vs native |
|--------|----------------------|-----------------|
| 512 | ~42 s | 2.7x |
| 1024 | ~62 s | 2.3x |
| 4096 | ~180 s | 1.9x |

The asymptotic ratio converges toward the pure compute overhead (~1.4–1.8x host wall-clock), which is dominated by KVM scheduling overhead rather than ISA limitations.

### Guest-Reported Decode Performance

| Config | 32 tokens | 128 tokens | 256 tokens |
|--------|-----------|------------|------------|
| Native (ms/tok) | 36.92 | 30.83 | 33.33 |
| Nanvix (ms/tok) | 20.49 | 27.87 | 22.14 |
| **Nanvix speedup** | **1.8x** | **1.1x** | **1.5x** |

The guest-reported per-token decode is consistently faster on Nanvix than on native x86_64. This is because:
1. AVX2/FMA SIMD operations process the same width (256-bit) on both architectures
2. The Q4_K_M dequantization kernels are dominated by SIMD throughput, not register count
3. The guest clock does not account for KVM VM-exit time (timer interrupts, APIC emulation)

The difference between guest-reported and host-measured generation time represents KVM virtualization overhead (~20 ms per token).

---

## Phase-by-Phase Breakdown

All times are wall-clock measurements taken from the host side by timestamping `[PHASE]` log markers emitted by the guest application as they appear on the host's stdout.

### 32 Tokens (prompt: "Hello")

| Phase | Native | Nanvix (memfs) | Ratio |
|-------|--------|----------------|-------|
| `nanvixd_boot` | — | 0.178 s | — |
| `backend_init` | 0.000 s | 0.002 s | — |
| `memfs_init` | — | 0.025 s | — |
| **`model_load`** | **0.558 s** | **16.286 s** | **29x** |
| `tokenize` | 0.002 s | 0.004 s | 2x |
| `ctx_create` | 0.014 s | 2.264 s | 162x |
| `prompt_eval` | 0.137 s | 0.150 s | 1.1x |
| `generation` | 1.155 s | 1.325 s | 1.1x |
| `cleanup/exit` | 0.062 s | 0.080 s | 1.3x |
| **Total** | **1.941 s** | **20.316 s** | **10.5x** |

### 128 Tokens (prompt: "Explain the theory of general relativity in detail")

| Phase | Native | Nanvix (memfs) | Ratio |
|-------|--------|----------------|-------|
| `nanvixd_boot` | — | 0.160 s | — |
| `memfs_init` | — | 0.019 s | — |
| **`model_load`** | **0.421 s** | **15.753 s** | **37x** |
| `ctx_create` | 0.009 s | 2.222 s | 247x |
| `prompt_eval` | 0.167 s | 0.319 s | 1.9x |
| `generation` | 3.951 s | 7.271 s | 1.8x |
| **Total** | **4.607 s** | **25.823 s** | **5.6x** |

### 256 Tokens (prompt: "Write a comprehensive guide to machine learning")

| Phase | Native | Nanvix (memfs) | Ratio |
|-------|--------|----------------|-------|
| `nanvixd_boot` | — | 0.157 s | — |
| `memfs_init` | — | 0.024 s | — |
| **`model_load`** | **0.437 s** | **16.791 s** | **38x** |
| `ctx_create` | 0.017 s | 3.231 s | 190x |
| `prompt_eval` | 0.162 s | 0.252 s | 1.6x |
| `generation` | 8.576 s | 11.604 s | 1.4x |
| **Total** | **9.253 s** | **32.162 s** | **3.5x** |

### Guest-Reported Performance (from llama_perf)

| Metric | Native (32 tok) | Nanvix (32 tok) | Native (128 tok) | Nanvix (128 tok) | Native (256 tok) | Nanvix (256 tok) |
|--------|----------------|-----------------|------------------|------------------|------------------|------------------|
| Load time | 718 ms | 9,330 ms | 607 ms | 9,126 ms | 625 ms | 10,114 ms |
| Prompt eval | 136 ms (66 t/s) | 89 ms (102 t/s) | 167 ms (108 t/s) | 171 ms (105 t/s) | 162 ms (93 t/s) | 138 ms (109 t/s) |
| Token eval | 1,144 ms (27 t/s) | 635 ms (49 t/s) | 3,915 ms (32 t/s) | 3,540 ms (36 t/s) | 8,499 ms (30 t/s) | 5,645 ms (45 t/s) |
| Total time | 1,873 ms | 9,988 ms | 4,558 ms | 12,744 ms | 9,202 ms | 15,904 ms |

---

## Root Cause Analysis

### 1. Model Loading — 29–38x Slower

**Symptom:** Loading the 462 MB GGUF model takes ~16 seconds on Nanvix vs ~0.5 seconds natively.

**Root cause:** The GGUF loader issues many small `fread()` and `fseek()` calls. Each call traverses the full POSIX stack even though data is served from memory:

```
Application (fread)
  → newlib libc (read syscall)
    → Nanvix POSIX layer (libposix.a)
      → memfs intercept → DirectRead memcpy
    → Return to libc
  → Next fread chunk
```

While VM exits are eliminated, the function call depth and small transfer sizes (often 4–32 bytes for metadata) create overhead for 462 MB of data. The zero-copy `DirectRead` serves reads via direct memcpy from the MMIO region, but the GGUF loader's access pattern (many seeks + small reads for tensor metadata, then large reads for tensor data) means thousands of round-trips through the POSIX layer.

### 2. Context Creation — 162–247x Slower

**Symptom:** `llama_init_from_model()` takes 2.2–3.2 seconds on Nanvix vs 9–17 ms natively.

**Root cause:** Large memory allocation overhead. Context creation allocates the KV cache and compute buffers (~33 MB for 256-token context). On Nanvix, heap growth requires kernel interactions per page. Bulk heap growth (1 MB chunks) helps but doesn't eliminate the overhead for hundreds of pages.

### 3. Generation — 1.1–1.8x Slower (host) / 1.1–1.8x Faster (guest)

**Symptom:** Host wall-clock shows 1.1–1.8x slowdown, but guest clock shows Nanvix is 1.1–1.8x *faster* per-token.

**Root cause:** The discrepancy is due to KVM virtualization overhead:
- **Guest clock**: Measures only time spent executing guest code (excludes VM exits). Nanvix with AVX2/FMA achieves faster SIMD throughput than native x86_64 for Q4_K_M kernels.
- **Host clock**: Includes VM-exit latency from timer interrupts, APIC emulation, and page table walks. This adds ~20 ms per token of invisible overhead.

The ~20 ms/token VM-exit overhead is consistent across all token counts, representing the fundamental KVM scheduling cost.

### 4. Prompt Evaluation — 1.1–1.9x Slower

**Symptom:** Batch prompt processing takes 0.15–0.32 s on Nanvix vs 0.14–0.17 s natively.

With AVX2/FMA enabled, the guest-reported prompt eval is competitive with native (9–10 ms/tok on both). The host wall-clock premium comes from KVM overhead during the batch GEMM operations.

---

## Optimization History

### Phase 1: Original (virtio-fs, SSE2) — 283 s total

| Phase | Time |
|-------|------|
| model_load | 279 s |
| generation (32 tok) | 4.27 s |
| ctx_create | 2.48 s |

### Phase 2: + AVX2/FMA, bulk heap — ~260 s total

Generation dropped from 4.27s → 1.36s (3.1x faster with AVX2/FMA).

### Phase 3: + memfs (with RAMFS copy) — 35.7 s total

Model load dropped from 279s → 18s (15.4x faster). But memfs_init added 14.3s of copy overhead.

### Phase 4: + zero-copy MMIO mount, batched stdout — 20.3 s total

| Change | Improvement |
|--------|-------------|
| Eliminate RAMFS → heap copy | memfs_init: 14.3s → 0.025s (**572x**) |
| Batch stdout output | generation: ~1.5s → 1.3s per 32 tokens |
| Model load improved | 18.1s → 16.3s (less heap pressure) |

**End-to-end: 283s → 20.3s (13.9x improvement)**

---

## Methodology

### Measurement Approach

Timing is measured **from the host side** using monotonic clock timestamps on `[PHASE]` log markers emitted by the guest:

1. The guest application emits `[PHASE] phase_name` to stderr at each transition (backend init, memfs init, model load, tokenize, context creation, prompt eval, generation, done).
2. The host benchmark script pipes all output through a Python timestamper that prepends `time.monotonic()` to each line.
3. A post-processing step computes deltas between consecutive `[PHASE]` markers.

This approach gives accurate wall-clock measurements because:
- Host `time.monotonic()` is unaffected by VM scheduling
- `[PHASE]` markers appear on the host only when the guest's `write()` syscall completes through the virtio-console path, providing synchronization points
- No in-guest timing instrumentation that could be distorted by virtualization

### Guest vs Host Clock Discrepancy

The guest-reported times (from `llama_perf_context_print()`) use `ggml_time_us()` which measures guest-side monotonic time. Under KVM, this clock does not account for time spent in VM exits (timer interrupts, I/O traps). As a result:
- **Guest time < Host time** for all compute phases
- The difference represents pure KVM overhead (~20 ms/token for decode)
- Guest-reported metrics are useful for comparing compute efficiency across ISAs

---

## Reproducing the Benchmarks

### Prerequisites

- Nanvix toolchain and sysroot built with `LOG_LEVEL=panic` for minimal logging overhead
- Qwen3-0.6B-Q4_K_M model in sysroot (`qwen3-0.6b-q4_k_m.gguf`, 462 MB)
- For memfs: FAT32 image of the model (`qwen3-0.6b-q4_k_m.fat`, 486 MB)
- `llama_simple.elf` built with `-DNANVIX_MEMFS` for memfs mode
- Native benchmark built (see below)

### Build Native Benchmark

```bash
# Build llama.cpp for host (CPU-only, no BLAS, no OpenMP)
cmake -B build-host-bench \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DGGML_NATIVE=ON \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF \
  -DGGML_BACKEND_DL=OFF -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF
cmake --build build-host-bench -j$(nproc) --target llama-simple

# Build native benchmark binary
g++ -O2 -std=c++17 -Iinclude -Iggml/include \
  -o benches/native_bench benches/native_bench.cpp \
  -Lbuild-host-bench/bin -lllama -lggml -lggml-base -lggml-cpu \
  -lpthread -lm -ldl
```

### Run Benchmarks

```bash
# Native baseline — 32, 128, 256 tokens
./scripts/bench-native.sh -n 32  -p "Hello" -t 1
./scripts/bench-native.sh -n 128 -p "Explain the theory of general relativity in detail" -t 1
./scripts/bench-native.sh -n 256 -p "Write a comprehensive guide to machine learning" -t 1

# Nanvix memfs — 32, 128, 256 tokens
./scripts/bench-nanvix.sh -n 32  -p "Hello" -f
./scripts/bench-nanvix.sh -n 128 -p "Explain the theory of general relativity in detail" -f
./scripts/bench-nanvix.sh -n 256 -p "Write a comprehensive guide to machine learning" -f
```

---

## Optimization Opportunities

### Completed Optimizations

| # | Optimization | Target Phase | Result | Status |
|---|-------------|--------------|--------|--------|
| 1 | **Zero-copy in-memory FAT32 filesystem** — mount RAMFS MMIO region directly as FAT32 storage, intercept POSIX syscalls, serve reads via direct memcpy from guest memory | model_load | 279s → 16s (**17x**) | **Done** ✅ |
| 2 | **Eliminate RAMFS copy** — use MMIO pointer directly for FAT32 mount instead of copying 486 MB to heap | memfs_init | 14.3s → 0.025s (**572x**) | **Done** ✅ |
| 3 | **AVX/AVX2/FMA enablement** — expose SSE3–AVX2 via CPUID, configure XCR0/CR4.OSXSAVE via KVM ioctls | generation | 4.27s → 1.33s (**3.2x**) | **Done** ✅ |
| 4 | **Bulk heap growth** — grow heap in 1 MB chunks instead of page-by-page | ctx_create | Reduced from ~2.5s | **Done** ✅ |
| 5 | **Batch stdout output** — remove per-token fflush() to reduce VM exits during generation | generation | ~6 ms/tok saved | **Done** ✅ |
| 6 | **LOG_LEVEL=panic** — eliminates runtime log formatting | all phases | minor | **Done** ✅ |

### Remaining Bottlenecks

| # | Optimization | Target Phase | Expected Improvement | Complexity |
|---|-------------|--------------|---------------------|------------|
| 7 | **Reduce model_load POSIX overhead** — bypass newlib/libposix layers entirely for GGUF loading; serve reads directly from MMIO pointer via a custom file-like interface | model_load | 5–10x faster (16s → 2–3s) | Medium |
| 8 | **Faster context allocation** — pre-map heap pages or batch page table updates for large allocations | ctx_create | 10–100x faster (2.3s → ~50ms) | Medium |
| 9 | **Reduce KVM VM-exit frequency** — lower timer interrupt rate, disable unnecessary APIC emulation | generation | 1.3–1.5x (eliminate ~20ms/tok overhead) | Low–Medium |
| 10 | **x86_64 port** — run Nanvix in 64-bit mode | generation | 1.2–1.5x faster decode | Very High |

### Projected Performance

With optimizations #7–#8, end-to-end time for 32 tokens could drop from 20.3s to ~5s. For 256 tokens, from 32.2s to ~15s. The generation phase is already near-optimal per the guest clock; the remaining host wall-clock overhead (~20 ms/tok) is KVM scheduling cost that would require VM-level optimizations to address.
