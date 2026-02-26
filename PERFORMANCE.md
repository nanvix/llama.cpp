# Performance Analysis: llama.cpp on Nanvix

> **TL;DR:** With in-memory FAT32 filesystem (memfs) and zero-copy reads, Nanvix runs end-to-end in **37.0 s** — a **7.1x improvement** over the previous 260.8 s. Model loading dropped from 257.0 s to 17.9 s (**14.4x faster**). Pure compute (generation) is only 1.4x slower than native, thanks to AVX2/FMA enablement.

---

## Table of Contents

1. [Test Setup](#test-setup)
2. [Results Summary](#results-summary)
3. [Phase-by-Phase Breakdown](#phase-by-phase-breakdown)
4. [Root Cause Analysis](#root-cause-analysis)
5. [Optimization History](#optimization-history)
6. [Methodology](#methodology)
7. [Reproducing the Benchmarks](#reproducing-the-benchmarks)
8. [Optimization Opportunities](#optimization-opportunities)

---

## Test Setup

| Parameter | Native (baseline) | Nanvix (virtio-fs) | Nanvix (memfs) |
|-----------|-------------------|---------------------|----------------|
| **Host CPU** | Intel Core i9-12900H | Intel Core i9-12900H | Intel Core i9-12900H |
| **Host OS** | Linux 5.15 (WSL2) | Linux 5.15 (WSL2) | Linux 5.15 (WSL2) |
| **Guest OS** | — | Nanvix (nanvixd.elf, KVM) | Nanvix (nanvixd.elf, KVM) |
| **Architecture** | x86_64 | i686 (32-bit) | i686 (32-bit) |
| **ISA Extensions** | AVX2, FMA, SSE4.2 | AVX2, FMA | AVX2, FMA |
| **Guest Memory** | — | 2 GB | 3 GB |
| **Threads** | 1 | 1 | 1 |
| **BLAS** | None (disabled) | None | None |
| **OpenMP** | Disabled | Disabled | Disabled |
| **Model** | Qwen3-0.6B-Q4_K_M (462 MB) | Qwen3-0.6B-Q4_K_M (462 MB) | Qwen3-0.6B-Q4_K_M (462 MB) |
| **Quantization** | Q4_K_M | Q4_K_M | Q4_K_M |
| **Tokens generated** | 32 | 32 | 32 |
| **Prompt** | "Hello" (chat template) | "Hello" (chat template) | "Hello" (chat template) |
| **I/O backend** | Native fs | virtio-fs (IPC) | In-memory FAT32 (zero-copy) |
| **mmap** | Disabled (`use_mmap=false`) | Disabled | Disabled |
| **Compiler** | GCC (host) | Clang 21 (cross) | Clang 21 (cross) |
| **Build type** | Release (`-O2`) | Release (`-O2`) | Release (`-O2`) |
| **Nanvix branch** | — | `feature-llama-perf` | `feature-llama-perf` |
| **Nanvix LOG_LEVEL** | — | `panic` (minimal logging) | `panic` (minimal logging) |

---

## Results Summary

### End-to-End

| Metric | Native | Nanvix (virtio-fs) | Nanvix (memfs) | Memfs vs Native | Memfs vs Virtio-fs |
|--------|--------|---------------------|----------------|-----------------|---------------------|
| **Total wall-clock** | 1.66 s | 260.83 s | 37.0 s | **22.3x** | **7.1x faster** |
| **Model load (host)** | 0.455 s | 257.0 s | 17.9 s | 39x | **14.4x faster** |
| **Tokens/second (eval)** | 30.53 t/s | 47.91 t/s | 45.08 t/s | 1.5x faster† | ~same |
| **Prompt eval (ms/tok)** | 12.58 ms | 10.75 ms | 10.96 ms | 1.1x faster† | ~same |
| **Token decode (ms/tok)** | 32.75 ms | 20.87 ms | 22.18 ms | 1.5x faster† | ~same |

†Guest-reported metrics; Nanvix now has AVX2/FMA enabled, making per-token compute competitive with native x86_64.

### Where the Time Goes

```
Native (1.66 s total):
  ██████████████████████████████████████████████████████████████ generation (62%)
  ████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ model_load (27%)
  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ prompt_eval (7%)
  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ other (4%)

Nanvix virtio-fs (260.83 s total):
  ██████████████████████████████████████████████████████████████ model_load (98.5%)
  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ generation (0.5%)
  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ctx_create (0.8%)
  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ other (0.2%)

Nanvix memfs (37.0 s total):
  ██████████████████████████████████████████████████████████████ model_load (48.3%)
  ████████████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░ memfs_init (39.1%)
  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ generation (3.9%)
  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ctx_create (7.4%)
  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ other (1.3%)
```

---

## Phase-by-Phase Breakdown

All times are wall-clock measurements taken from the host side by timestamping `[PHASE]` log markers emitted by the guest application as they appear on the host's stdout.

| Phase | Native | Nanvix (virtio-fs) | Nanvix (memfs) | Memfs vs Native | Memfs vs Virtio-fs |
|-------|--------|---------------------|----------------|-----------------|---------------------|
| `nanvixd_boot` | — | 0.170 s | 0.182 s | — | ~same |
| `backend_init` | 0.000 s | 0.002 s | 0.002 s | — | ~same |
| `memfs_init` | — | 0.009 s | **14.450 s** | — | RAMFS copy to heap |
| **`model_load`** | **0.455 s** | **256.963 s** | **17.880 s** | **39x** | **14.4x faster** |
| `tokenize` | 0.002 s | 0.004 s | 0.005 s | 2.5x | ~same |
| `ctx_create` | 0.010 s | 2.057 s | 2.727 s | 273x | ~same |
| `prompt_eval` | 0.114 s | 0.169 s | 0.169 s | 1.5x | ~same |
| `generation` | 1.024 s | 1.361 s | 1.458 s | 1.4x | ~same |
| `cleanup/exit` | 0.042 s | 0.093 s | 0.105 s | 2.5x | ~same |
| **Total** | **1.662 s** | **260.830 s** | **36.980 s** | **22.3x** | **7.1x faster** |

### Key Observations

1. **Model load: 257s → 18s (14.4x faster)** — Memfs eliminates the IPC/VM-exit overhead of virtio-fs. The remaining 18s is the FAT32 library reading metadata and llama.cpp processing the GGUF file through standard fread/fseek (now served from memory instead of virtio-fs).

2. **Memfs init: 14.5s** — This is new overhead: copying the 486 MB RAMFS image from MMIO into the user heap and mounting the FAT32 filesystem. This happens once at startup and is amortized over inference.

3. **Generation: 1.4x native** — With AVX2/FMA enabled, per-token compute on Nanvix (22 ms/tok guest-reported) is competitive with native x86_64 (33 ms/tok). The i686 penalty is offset by AVX2 enablement.

4. **ctx_create still slow (273x)** — KV cache and compute buffer allocation through Nanvix's heap allocator remains expensive. Bulk heap growth helps but doesn't eliminate per-page kernel interactions entirely.

### Guest-Reported Performance (from llama_perf)

| Metric | Native | Nanvix (virtio-fs) | Nanvix (memfs) |
|--------|--------|---------------------|----------------|
| Model load time | 587 ms | 129,265 ms | 10,365 ms |
| Prompt eval | 113 ms / 9 tok (79 t/s) | 97 ms / 9 tok (93 t/s) | 99 ms / 9 tok (91 t/s) |
| Token eval | 1,015 ms / 31 runs (31 t/s) | 647 ms / 31 runs (48 t/s) | 688 ms / 31 runs (45 t/s) |
| Total context time | 1,612 ms | 129,948 ms | 11,096 ms |

---

## Root Cause Analysis

### 1. Model Loading (virtio-fs) — 565x Slower

**Symptom:** Loading the 462 MB GGUF model takes 257 seconds via virtio-fs vs 0.5 seconds natively.

**Root cause:** File I/O through Nanvix's virtio/IPC path.

The model file is read via standard `fread()` calls from the guest. Each read request crosses multiple boundaries:

```
Application (fread)
  → newlib libc (read syscall)
    → Nanvix POSIX layer (libposix.a)
      → IPC message to kernel
        → Kernel virtio-fs handler
          → VM exit (KVM → host)
            → Host filesystem read
          → VM entry (host → KVM)
        → IPC response to userland
      → Copy data to user buffer
    → Return to libc
  → Next fread chunk
```

Each chunk involves at least one KVM VM-exit/VM-entry pair plus IPC overhead. For a 462 MB file read in small chunks, this adds up to hundreds of thousands of round trips.

### 2. Model Loading (memfs) — 39x Slower (vs native)

**Symptom:** With memfs, model loading takes 17.9 seconds — 14.4x faster than virtio-fs but still 39x slower than native.

**Root cause:** The remaining overhead comes from:

1. **FAT32 library overhead**: The `fatfs` crate still processes each `fread()` call through its cluster chain logic, even though the underlying storage is memory. Each read resolves clusters, walks the FAT table, and copies data in cluster-sized chunks (4-32 KB).

2. **POSIX layer round-trips**: Each `fread()` still goes through newlib → libposix → memfs handler → FAT32 read → memcpy. While there are no VM exits, the function call chain is deep.

3. **memfs_init overhead (14.5s)**: Copying 486 MB from MMIO to heap involves kernel page mapping and memcpy.

**Optimization path:** The zero-copy `DirectRead` struct (already implemented) resolves a file's contiguous byte range at open time, bypassing the FAT32 cluster walk for subsequent reads. This is active but the remaining 17.9s suggests the GGUF loader's many small fread/fseek calls still have overhead in the POSIX/newlib layers.

### 3. Context Creation — 273x Slower

**Symptom:** `llama_init_from_model()` takes 2.7 seconds on Nanvix vs 10 ms natively.

**Root cause:** Large memory allocation overhead. Context creation allocates the KV cache and compute buffers (~33 MB total). On Nanvix, each page of heap growth requires a kernel interaction.

### 4. Generation — 1.4x Slower

**Symptom:** Token decode takes 1.46 s for 32 tokens on Nanvix vs 1.02 s natively (host wall-clock).

**Root cause:** Minimal — primarily i686 32-bit overhead (half the GPRs, no native 64-bit ops). With AVX2/FMA now enabled, SIMD performance is comparable to native x86_64. The 1.4x ratio is close to the theoretical minimum for 32-bit vs 64-bit execution.

### 5. Prompt Evaluation — 1.5x Slower

**Symptom:** Prompt processing takes 0.169 s on Nanvix vs 0.114 s natively.

With AVX2/FMA enabled, the slowdown is now only 1.5x (down from 7.4x with SSE2-only). The remaining gap is due to i686 register and addressing limitations.

---

## Optimization History

### Before (virtio-fs only, SSE2): 252.8 s total

| Phase | Time |
|-------|------|
| model_load | 245.5 s |
| generation | 4.27 s |
| ctx_create | 1.94 s |
| Other | 1.1 s |

### After all optimizations (memfs, AVX2/FMA, bulk heap): 37.0 s total

| Phase | Time | Improvement |
|-------|------|-------------|
| model_load | 17.9 s | **13.7x faster** |
| memfs_init | 14.5 s | (new overhead) |
| generation | 1.46 s | **2.9x faster** (AVX2/FMA) |
| ctx_create | 2.73 s | ~same |
| Other | 0.46 s | ~same |

**End-to-end: 252.8s → 37.0s (6.8x improvement)**

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

### strace Analysis

Host-side `strace -c` on `nanvixd.elf` captures the hypervisor's system call profile. This shows what the host kernel sees (KVM ioctls, file I/O for the virtio backend).

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
# Native baseline (host-side phase timing)
./scripts/bench-native.sh -n 32 -p "Hello" -t 1

# Nanvix with virtio-fs (host-side phase timing)
./scripts/bench-nanvix.sh -n 32 -p "Hello"

# Nanvix with memfs (host-side phase timing)
./scripts/bench-nanvix.sh -n 32 -p "Hello" -f

# Nanvix with strace syscall profiling
./scripts/bench-nanvix.sh -n 32 -p "Hello" -s
```

---

## Optimization Opportunities

Ranked by expected impact:

### Completed Optimizations

| # | Optimization | Target Phase | Result | Status |
|---|-------------|--------------|--------|--------|
| 1 | **In-memory FAT32 filesystem + zero-copy reads** — load model from RAMFS MMIO; POSIX syscall bindings intercept `open`/`read`/`fstat`/`lseek` and serve directly from memory, bypassing all IPC/VM-exit overhead. Zero-copy `DirectRead` resolves file byte range at open time. | model_load | 257s → 18s (**14.4x**) | **Done** ✅ |
| 2 | **AVX/AVX2/FMA enablement** — expose SSE3–AVX2 via CPUID, enable CR4.OSXSAVE + XCR0, and compile GGML with vectorized kernels | generation | 4.27s → 1.46s (**2.9x**) | **Done** ✅ |
| 3 | **Bulk heap growth** — grow heap in 1 MB chunks instead of page-by-page to reduce `int 0x80` / VM-exit round trips per allocation | ctx_create | Reduced from ~1.94s | **Done** ✅ |
| 4 | **Reduce LOG_LEVEL=panic** — eliminates runtime log formatting | all phases | minor | **Done** ✅ |

### Remaining Bottlenecks

| # | Optimization | Target Phase | Expected Improvement | Complexity |
|---|-------------|--------------|---------------------|------------|
| 5 | **Eliminate memfs_init copy** — map RAMFS MMIO directly into user address space instead of copying 486 MB to heap at startup | memfs_init | Eliminate 14.5s overhead | Medium |
| 6 | **Reduce model_load FAT32 overhead** — bypass FAT32 cluster chain entirely (the zero-copy `DirectRead` is active, but remaining 18s suggests GGUF loader overhead) | model_load | 2–5x faster load | Medium |
| 7 | **Faster context allocation** — pre-map heap pages or batch page table updates for large allocations | ctx_create | 10–100x faster ctx | Medium |
| 8 | **x86_64 port** — run Nanvix in 64-bit mode (requires major kernel work) | generation | 1.3–1.5x faster decode | Very High |
| 9 | **Profile-guided optimization** — PGO build of llama.cpp for i686 | generation | 1.1–1.3x faster decode | Low |
| 10 | **Reduce timer interrupt frequency** — lower tick rate in KVM to reduce VM exits during compute | generation | 1.05–1.1x | Low |

### Projected Performance

With the remaining optimizations (#5–#7), end-to-end time could drop from 37s to ~5–8s. The generation phase (1.5s) is already within 1.4x of native and represents the theoretical floor for i686 execution. The primary remaining targets are memfs_init (14.5s) and model_load (17.9s).
