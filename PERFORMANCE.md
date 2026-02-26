# Performance Analysis: llama.cpp on Nanvix

> **TL;DR:** With BTreeMap page tables, batch frame allocation, skip-zero-fill, bitmap allocator optimization, batch mmap, setvbuf, and zero-copy memfs, Nanvix runs LLM inference end-to-end in **3.8 s** (32 tokens) — a **74x improvement** over the original virtio-fs baseline of 283 s. Model loading dropped from 279 s to 2.1 s (**133x faster**). At 256 tokens, the overhead ratio drops to **1.5x** as compute dominates.

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
| **Total wall-clock** | 1.62 s | 3.82 s | **2.4x** |
| **Model load (host)** | 0.504 s | 2.05 s | 4.1x |
| **Generation (host)** | 0.899 s | 1.19 s | 1.3x |
| **ctx_create (host)** | 0.010 s | 0.197 s | 20x |

### Where the Time Goes

```
Native (1.62 s total):
  █████████████████████████████████████████████████████████████░░ generation (55%)
  ██████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ model_load (31%)
  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ prompt_eval (9%)
  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ other (5%)

Nanvix memfs (3.82 s total):
  ██████████████████████████████████████████████████████████████░░ model_load (53.7%)
  ███████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ generation (31.2%)
  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ctx_create (5.2%)
  █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ prompt_eval (4.0%)
  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ boot/other (5.9%)
```

---

## Scaling Behavior

All times are host-side wall-clock measurements.

### Across Token Counts

| Metric | 32 tokens | 128 tokens | 256 tokens |
|--------|-----------|------------|------------|
| **Native total** | 1.62 s | 4.74 s | 9.43 s |
| **Nanvix total** | 3.82 s | 7.93 s | 13.73 s |
| **Ratio** | 2.4x | 1.7x | 1.5x |
| | | | |
| Native generation | 0.899 s | 4.025 s | 8.745 s |
| Nanvix generation | 1.190 s | 5.238 s | 10.944 s |
| Gen ratio | 1.3x | 1.3x | 1.3x |
| | | | |
| Native model_load | 0.504 s | 0.492 s | 0.462 s |
| Nanvix model_load | 2.054 s | 2.122 s | 2.102 s |
| Load ratio | 4.1x | 4.3x | 4.5x |
| | | | |
| Nanvix ctx_create | 0.197 s | 0.203 s | 0.196 s |

### Key Insight: Amortization

The fixed-cost phases (model load: ~2.1 s, ctx_create: ~0.2 s, boot: ~0.15 s) are amortized over more tokens. At 256 tokens, generation dominates and the Nanvix-to-native ratio drops to **1.5x**. Extrapolating:

| Tokens | Estimated Nanvix total | Ratio vs native |
|--------|----------------------|-----------------|
| 512 | ~24 s | 1.4x |
| 1024 | ~44 s | 1.3x |
| 4096 | ~170 s | 1.3x |

The asymptotic ratio converges toward the pure compute overhead (~1.3x host wall-clock), which is dominated by KVM scheduling overhead rather than ISA limitations.

### Guest-Reported Decode Performance

The guest-reported per-token decode is consistently competitive with native x86_64. This is because:
1. AVX2/FMA SIMD operations process the same width (256-bit) on both architectures
2. The Q4_K_M dequantization kernels are dominated by SIMD throughput, not register count
3. The guest clock does not account for KVM VM-exit time (timer interrupts, APIC emulation)

The difference between guest-reported and host-measured generation time represents KVM virtualization overhead.

---

## Phase-by-Phase Breakdown

All times are wall-clock measurements taken from the host side by timestamping `[PHASE]` log markers emitted by the guest application as they appear on the host's stdout.

### 32 Tokens (prompt: "Explain the concept of machine learning in simple terms")

| Phase | Native | Nanvix (memfs) | Ratio |
|-------|--------|----------------|-------|
| **`model_load`** | **0.504 s** | **2.054 s** | **4.1x** |
| `ctx_create` | 0.010 s | 0.197 s | 20x |
| `prompt_eval` | 0.148 s | 0.154 s | 1.0x |
| `generation` | 0.899 s | 1.190 s | 1.3x |
| **Total** | **1.616 s** | **3.824 s** | **2.4x** |

### 128 Tokens (prompt: "Explain the concept of machine learning in simple terms")

| Phase | Native | Nanvix (memfs) | Ratio |
|-------|--------|----------------|-------|
| **`model_load`** | **0.492 s** | **2.122 s** | **4.3x** |
| `ctx_create` | 0.012 s | 0.203 s | 17x |
| `prompt_eval` | 0.155 s | 0.160 s | 1.0x |
| `generation` | 4.025 s | 5.238 s | 1.3x |
| **Total** | **4.736 s** | **7.933 s** | **1.7x** |

### 256 Tokens (prompt: "Explain the concept of machine learning in simple terms")

| Phase | Native | Nanvix (memfs) | Ratio |
|-------|--------|----------------|-------|
| **`model_load`** | **0.462 s** | **2.102 s** | **4.5x** |
| `ctx_create` | 0.024 s | 0.196 s | 8.2x |
| `prompt_eval` | 0.155 s | 0.180 s | 1.2x |
| `generation` | 8.745 s | 10.944 s | 1.3x |
| **Total** | **9.434 s** | **13.732 s** | **1.5x** |

---

## Root Cause Analysis

### 1. Model Loading — 4.1x Slower

**Symptom:** Loading the 462 MB GGUF model takes ~2.1 seconds on Nanvix vs ~0.5 seconds natively.

**Root cause:** The dominant cost is **heap allocation and page mapping**. When the GGUF loader allocates ~462 MB for tensor weights via `malloc`, the Nanvix heap must grow by mapping ~118,000 pages. With batch mmap (Phase 5), pages are mapped in a single kernel trap. The bitmap next_free hint (Phase 6) reduced frame allocation from O(n²) to O(n). Skipping page zero-fill (Phase 7) eliminated 462 MB of memset. Batch contiguous frame allocation with per-page-table mapping (Phase 8) eliminated per-page overhead. The remaining ~1.6s is dominated by KVM EPT (Extended Page Table) overhead for per-page PTE writes.

The I/O itself is fast: with 256 KB stdio buffers (via setvbuf), reading 462 MB from the in-memory FAT32 takes only ~50 ms via `fread`.

### 2. Context Creation — 8–20x Slower

**Symptom:** `llama_init_from_model()` takes 0.2 seconds on Nanvix vs 10–24 ms natively.

**Root cause:** KV cache and compute buffer allocation (~33 MB for 256-token context). Same per-page kernel overhead as model loading, but smaller total. With skip-zero-fill, bitmap hint, and batch allocation, ctx_create dropped from 2.3 s → 0.2 s (92% improvement).

### 3. Generation — 1.3x Slower (host)

**Symptom:** Host wall-clock shows ~1.3x slowdown, but guest clock shows Nanvix is competitive per-token.

**Root cause:** The discrepancy is due to KVM virtualization overhead:
- **Guest clock**: Measures only time spent executing guest code (excludes VM exits). Nanvix with AVX2/FMA achieves competitive SIMD throughput for Q4_K_M kernels.
- **Host clock**: Includes VM-exit latency from timer interrupts (10 KHz), APIC emulation, and EPT page table walks. This adds ~10 ms per token of invisible overhead.

The ~10 ms/token VM-exit overhead is consistent across all token counts, representing the fundamental KVM scheduling cost at 10 KHz timer frequency.

### 4. Prompt Evaluation — 1.0–1.2x Slower

**Symptom:** Batch prompt processing takes 0.15–0.18 s on Nanvix vs 0.15–0.16 s natively.

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

### Phase 5: + batch mmap, setvbuf, 4MB heap growth — 16.6 s total

| Change | Improvement |
|--------|-------------|
| Batch mmap kernel call | Reduces 118K `int 0x80` traps to ~115 for heap growth |
| 256 KB stdio buffer (setvbuf) | Reduces fread syscalls from ~462K to ~1,800 |
| 4 MB heap growth chunks | Fewer OOM handler invocations |
| Single process lookup per batch | Eliminates per-page process table scan |
| model_load | 16.3s → 13.0s (**20% faster**) |
| ctx_create | 2.3s → 1.8s (**22% faster**) |

### Phase 6: + bitmap allocator next_free hint — 14.1 s total

| Change | Improvement |
|--------|-------------|
| Bitmap next_free hint | Frame allocation from O(n²) to O(n) for sequential patterns |
| model_load | 13.0s → 10.7s (**18% faster**) |
| Total (32 tok) | 16.6s → 14.1s (**15% faster**) |

The bitmap allocator previously scanned from bit 0 on every allocation. For 118K
sequential frame allocations, this caused ~7 billion bit checks. The `next_free`
hint starts scanning from where the last allocation ended, reducing total bit
checks to ~118K.

**End-to-end: 283s → 14.1s (20.1x improvement)**

### Phase 7: + skip page zero-fill — 4.5 s total

| Change | Improvement |
|--------|-------------|
| Skip memset(0) in mmap_range | Eliminates 462 MB of zero-fill per model load |
| model_load | 10.7s → 2.5s (**4.3x faster, -77%**) |
| ctx_create | 1.5s → 0.24s (**6.3x faster, -84%**) |
| Total (32 tok) | 14.1s → 4.5s (**3.1x faster**) |

Heap pages are immediately overwritten by the allocator's metadata and by
application data, so zero-filling them is wasted work. Skipping the per-page
`memset(vaddr, 0)` in `mmap_range` eliminates 118K × 4KB = 462 MB of
unnecessary memory writes during model loading.

**End-to-end: 283s → 4.5s (63x improvement)**

### Phase 8: + BTreeMap page tables & batch frame allocation — 3.8 s total

| Change | Improvement |
|--------|-------------|
| BTreeMap for page table lookup | O(n) → O(log n) per page |
| Contiguous frame allocation | Single bitmap.alloc_range(n) vs n individual allocs |
| Per-page-table batch mapping | 118K BTreeMap lookups → 116 (one per page table) |
| model_load | 2.5s → 2.1s (**-18%**) |
| ctx_create | 0.24s → 0.20s (**-17%**) |
| generation | 1.28s → 1.19s (**-7%**) |
| Total (32 tok) | 4.5s → 3.8s (**-15%**) |

Three kernel-level optimizations targeting per-page overhead in the virtual memory
subsystem:
1. **BTreeMap page tables:** Replaced LinkedList with BTreeMap for `user_page_tables`,
   reducing lookup from O(n) linear scan to O(log n) tree lookup per page operation.
2. **Contiguous frame allocation:** Added `alloc_contiguous(n)` using `bitmap.alloc_range(n)`
   to allocate N frames in a single bitmap scan instead of N separate allocations.
3. **Per-page-table batch mapping:** `map_range()` iterates per page table boundary
   (every 1024 pages), doing only one BTreeMap lookup per page table instead of per page.

**End-to-end: 283s → 3.8s (74x improvement)**

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
- The difference represents pure KVM overhead (~10 ms/token for decode)
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
./scripts/bench-native.sh -n 32  -p "Explain the concept of machine learning in simple terms"
./scripts/bench-native.sh -n 128 -p "Explain the concept of machine learning in simple terms"
./scripts/bench-native.sh -n 256 -p "Explain the concept of machine learning in simple terms"

# Nanvix memfs — 32, 128, 256 tokens
./scripts/bench-nanvix.sh -n 32  -p "Explain the concept of machine learning in simple terms" -f
./scripts/bench-nanvix.sh -n 128 -p "Explain the concept of machine learning in simple terms" -f
./scripts/bench-nanvix.sh -n 256 -p "Explain the concept of machine learning in simple terms" -f
```

---

## Optimization Opportunities

### Completed Optimizations

| # | Optimization | Target Phase | Result | Status |
|---|-------------|--------------|--------|--------|
| 1 | **Zero-copy in-memory FAT32 filesystem** — mount RAMFS MMIO region directly as FAT32 storage, intercept POSIX syscalls, serve reads via direct memcpy from guest memory | model_load | 279s → 16s (**17x**) | **Done** ✅ |
| 2 | **Eliminate RAMFS copy** — use MMIO pointer directly for FAT32 mount instead of copying 486 MB to heap | memfs_init | 14.3s → 0.025s (**572x**) | **Done** ✅ |
| 3 | **AVX/AVX2/FMA enablement** — expose SSE3–AVX2 via CPUID, configure XCR0/CR4.OSXSAVE via KVM ioctls | generation | 4.27s → 1.33s (**3.2x**) | **Done** ✅ |
| 4 | **Bulk heap growth** — grow heap in 4 MB chunks instead of page-by-page | ctx_create | Reduced from ~2.5s | **Done** ✅ |
| 5 | **Batch stdout output** — remove per-token fflush() to reduce VM exits during generation | generation | ~6 ms/tok saved | **Done** ✅ |
| 6 | **LOG_LEVEL=panic** — eliminates runtime log formatting | all phases | minor | **Done** ✅ |
| 7 | **Batch mmap kernel call** — map multiple pages in a single `int 0x80` trap, amortizing per-page trap overhead | model_load, ctx_create | model_load: 16.3s → 13.0s (**20%**); ctx_create: 2.3s → 1.8s (**22%**) | **Done** ✅ |
| 8 | **Larger stdio buffers (setvbuf)** — override `ggml_fopen` to use 256 KB buffer instead of newlib's 1 KB BUFSIZ | model_load | Reduces fread syscalls from 462K to 1,800 | **Done** ✅ |
| 9 | **Bitmap allocator next_free hint** — track lowest likely-free bit to avoid O(n²) rescans during sequential frame allocation | model_load, ctx_create | model_load: 13.0s → 10.7s (**18%**) | **Done** ✅ |
| 10 | **Skip page zero-fill** — pass `clear=false` to `alloc_upage` in `mmap_range` since heap pages are immediately overwritten | model_load, ctx_create | model_load: 10.7s → 2.5s (**77%**); ctx_create: 1.5s → 0.24s (**84%**) | **Done** ✅ |
| 11 | **BTreeMap page tables** — replace LinkedList with BTreeMap for user_page_tables, O(n) → O(log n) lookup | model_load, ctx_create | Reduced per-page overhead | **Done** ✅ |
| 12 | **Batch contiguous frame allocation** — alloc_range(n) + per-page-table map_range() | model_load, ctx_create | model_load: 2.5s → 2.1s (**18%**); total: 4.5s → 3.8s (**15%**) | **Done** ✅ |

### Remaining Bottlenecks

| # | Optimization | Target Phase | Expected Improvement | Complexity |
|---|-------------|--------------|---------------------|------------|
| 13 | **4MB PSE large pages** — use x86 PSE to map 4MB pages, reducing 118K PTEs to 114 PDEs | model_load | 4x faster (2.1s → ~0.5s) | High |
| 14 | **Reduce KVM VM-exit frequency** — lower timer interrupt rate, disable unnecessary APIC emulation | generation | 1.2–1.3x (eliminate ~10ms/tok overhead) | Low–Medium |
| 15 | **x86_64 port** — run Nanvix in 64-bit mode | generation | 1.2–1.5x faster decode | Very High |

### Projected Performance

With optimization #13 (4MB large pages), model_load could drop from 2.1s to ~0.5s (matching native), bringing the 32-token total from 3.8s to ~2.2s (1.4x native). The generation phase at 1.3x host overhead is the remaining bottleneck, dominated by KVM scheduling cost (~10ms/tok in VM exits).
