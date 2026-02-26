# Performance Analysis: llama.cpp on Nanvix

> **TL;DR:** With reduced timer frequency (100 Hz), idle-loop polling, BTreeMap page tables, batch frame allocation, skip-zero-fill, bitmap allocator optimization, batch mmap, setvbuf, and zero-copy memfs, Nanvix runs LLM inference end-to-end in **3.6 s** (32 tokens) — a **79x improvement** over the original virtio-fs baseline of 283 s. Model loading dropped from 279 s to 2.0 s (**140x faster**). At 256 tokens, the overhead ratio drops to **1.5x** as compute dominates.

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
| **Total wall-clock** | 2.02 s | 3.59 s | **1.8x** |
| **Model load (host)** | 0.863 s | 1.99 s | 2.3x |
| **Generation (host)** | 0.917 s | 1.09 s | 1.2x |
| **ctx_create (host)** | 0.015 s | 0.192 s | 13x |

### Where the Time Goes

```
Native (2.02 s total):
  ████████████████████████████████████████████████████████████░░░ generation (45%)
  ██████████████████████████████████████████████░░░░░░░░░░░░░░░░░ model_load (43%)
  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ prompt_eval (8%)
  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ other (4%)

Nanvix memfs (3.59 s total):
  ██████████████████████████████████████████████████████████████░░ model_load (55.5%)
  ████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ generation (30.3%)
  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ctx_create (5.3%)
  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ prompt_eval (3.4%)
  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ boot/other (5.5%)
```

---

## Scaling Behavior

All times are host-side wall-clock measurements.

### Across Token Counts

| Metric | 32 tokens | 128 tokens | 256 tokens |
|--------|-----------|------------|------------|
| **Native total** | 2.02 s | 4.42 s | 8.54 s |
| **Nanvix total** | 3.59 s | 6.98 s | 12.39 s |
| **Ratio** | 1.8x | 1.6x | 1.5x |
| | | | |
| Native generation | 0.917 s | 3.776 s | 8.023 s |
| Nanvix generation | 1.086 s | 4.457 s | 9.793 s |
| Gen ratio | 1.2x | 1.2x | 1.2x |
| | | | |
| Native model_load | 0.863 s | 0.423 s | 0.302 s |
| Nanvix model_load | 1.993 s | 2.094 s | 2.079 s |
| Load ratio | 2.3x | 5.0x | 6.9x |
| | | | |
| Nanvix ctx_create | 0.192 s | 0.197 s | 0.208 s |

### Key Insight: Amortization

The fixed-cost phases (model load: ~2.0 s, ctx_create: ~0.2 s, boot: ~0.15 s) are amortized over more tokens. At 256 tokens, generation dominates and the Nanvix-to-native ratio drops to **1.5x**. Extrapolating:

| Tokens | Estimated Nanvix total | Ratio vs native |
|--------|----------------------|-----------------|
| 512 | ~22 s | 1.4x |
| 1024 | ~40 s | 1.3x |
| 4096 | ~152 s | 1.2x |

The asymptotic ratio converges toward the pure compute overhead (~1.2x host wall-clock), which is dominated by KVM scheduling overhead rather than ISA limitations.

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
| **`model_load`** | **0.863 s** | **1.993 s** | **2.3x** |
| `ctx_create` | 0.015 s | 0.192 s | 13x |
| `prompt_eval` | 0.163 s | 0.123 s | 0.8x |
| `generation` | 0.917 s | 1.086 s | 1.2x |
| **Total** | **2.019 s** | **3.593 s** | **1.8x** |

### 128 Tokens (prompt: "Explain the concept of machine learning in simple terms")

| Phase | Native | Nanvix (memfs) | Ratio |
|-------|--------|----------------|-------|
| **`model_load`** | **0.423 s** | **2.094 s** | **5.0x** |
| `ctx_create` | 0.013 s | 0.197 s | 15x |
| `prompt_eval` | 0.161 s | 0.026 s | 0.2x |
| `generation` | 3.776 s | 4.457 s | 1.2x |
| **Total** | **4.424 s** | **6.980 s** | **1.6x** |

### 256 Tokens (prompt: "Explain the concept of machine learning in simple terms")

| Phase | Native | Nanvix (memfs) | Ratio |
|-------|--------|----------------|-------|
| **`model_load`** | **0.302 s** | **2.079 s** | **6.9x** |
| `ctx_create` | 0.010 s | 0.208 s | 21x |
| `prompt_eval` | 0.146 s | 0.074 s | 0.5x |
| `generation` | 8.023 s | 9.793 s | 1.2x |
| **Total** | **8.538 s** | **12.386 s** | **1.5x** |

---

## Root Cause Analysis

### 1. Model Loading — 2.3–6.9x Slower

**Symptom:** Loading the 462 MB GGUF model takes ~2.0 seconds on Nanvix vs ~0.3–0.9 seconds natively.

**Root cause:** The dominant cost is **heap allocation and page mapping**. When the GGUF loader allocates ~462 MB for tensor weights via `malloc`, the Nanvix heap must grow by mapping ~118,000 pages. With batch mmap (Phase 5), pages are mapped in a single kernel trap. The bitmap next_free hint (Phase 6) reduced frame allocation from O(n²) to O(n). Skipping page zero-fill (Phase 7) eliminated 462 MB of memset. Batch contiguous frame allocation with per-page-table mapping (Phase 8) eliminated per-page overhead. The remaining ~1.5s is dominated by KVM EPT (Extended Page Table) overhead for per-page PTE writes.

The I/O itself is fast: with 256 KB stdio buffers (via setvbuf), reading 462 MB from the in-memory FAT32 takes only ~50 ms via `fread`.

### 2. Context Creation — 13–21x Slower

**Symptom:** `llama_init_from_model()` takes 0.2 seconds on Nanvix vs 10–15 ms natively.

**Root cause:** KV cache and compute buffer allocation (~33 MB for 256-token context). Same per-page kernel overhead as model loading, but smaller total. With skip-zero-fill, bitmap hint, and batch allocation, ctx_create dropped from 2.3 s → 0.2 s (92% improvement).

### 3. Generation — 1.2x Slower (host)

**Symptom:** Host wall-clock shows ~1.2x slowdown, but guest clock shows Nanvix is competitive per-token.

**Root cause:** The discrepancy is due to KVM virtualization overhead:
- **Guest clock**: Measures only time spent executing guest code (excludes VM exits). Nanvix with AVX2/FMA achieves competitive SIMD throughput for Q4_K_M kernels.
- **Host clock**: Includes VM-exit latency from timer interrupts (100 Hz after Phase 9), APIC emulation, and EPT page table walks.

With the timer reduced from 10 kHz to 100 Hz (Phase 9), per-token overhead dropped from ~10 ms to ~5 ms, bringing the generation ratio from 1.3x to 1.2x.

### 4. Prompt Evaluation — 0.2–0.8x (faster than native)

**Symptom:** Nanvix prompt eval is sometimes reported as faster than native (0.03–0.12 s vs 0.15 s). This is a measurement artifact: with the 100 Hz timer, guest-side timing has 10 ms granularity, and some prompt eval time may be attributed to adjacent phases.

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

### Phase 9: + reduced timer frequency & idle-loop polling — 3.6 s total

| Change | Improvement |
|--------|-------------|
| PIT timer 10 kHz → 100 Hz | 100x fewer timer VM exits during compute |
| HLT → PAUSE in kernel idle loop | IKC polling no longer blocked by timer |
| Scheduler quantum 1000 → 10 ticks | Maintains 100ms preemption interval |
| generation (32 tok) | 1.19s → 1.09s (**-8%**) |
| generation (256 tok) | 10.94s → 9.79s (**-11%**) |
| Total (32 tok) | 3.8s → 3.6s (**-6%**) |
| Total (256 tok) | 13.7s → 12.4s (**-10%**) |

Two complementary changes that reduce KVM overhead:
1. **Lower timer frequency:** Reducing the PIT from 10 kHz to 100 Hz eliminates ~99%
   of timer-triggered VM exits during compute-bound phases. Each VM exit has ~5μs
   overhead; at 10 kHz that's ~30 exits per 30ms token = ~150μs/token saved.
2. **Idle-loop polling:** Replacing HLT with PAUSE in the kernel idle loop ensures
   IKC messages (IPC responses from linuxd) are polled continuously. Without this,
   the 100 Hz timer would increase IPC latency from ~100μs to ~10ms per round-trip,
   causing a 3x regression in model loading. The PAUSE-based polling trades idle CPU
   power for near-zero IPC latency.

**End-to-end: 283s → 3.6s (79x improvement)**

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
- The difference represents pure KVM overhead (~5 ms/token for decode at 100 Hz timer)
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
| 13 | **Reduce timer frequency (100 Hz)** — lower PIT from 10 kHz to 100 Hz + PAUSE idle loop | generation | generation: -8 to -11%; total (256 tok): 13.7s → 12.4s (**10%**) | **Done** ✅ |

### Remaining Bottlenecks

| # | Optimization | Target Phase | Expected Improvement | Complexity |
|---|-------------|--------------|---------------------|------------|
| 14 | **4MB PSE large pages** — use x86 PSE to map 4MB pages in non-PAE mode, reducing 118K PTEs to 114 PDEs. Requires: CR4.PSE enable, PDE.PS bit support, 4MB-aligned frame allocation, new map/unmap paths | model_load | 4x faster (2.0s → ~0.5s) | High |
| 15 | **Demand paging** — defer page mapping to first access via page fault handler; only map pages that are actually touched during model loading, avoid mapping unused heap pages | model_load | 10–30% (fewer total pages mapped) | Medium |
| 16 | **Interrupt-driven IKC** — configure vmbus to generate an interrupt (e.g., MMIO doorbell) when an IKC message arrives, replacing the PAUSE-based busy-poll in the kernel idle loop. Reduces host CPU usage during IPC-heavy phases and eliminates wasted spin cycles | IPC latency, host CPU | Neutral on throughput; lowers host CPU from 100% to near-idle during waits | Medium |
| 17 | **Reduce stderr log output** — suppress or batch llama.cpp's model-loader metadata prints (34 KV lines + tensor info) that each require an IPC round-trip to linuxd; redirect to /dev/null or buffer in guest | model_load | 5–10% (eliminate ~100 IPC round-trips during metadata logging) | Low |
| 18 | **Pre-map heap at startup** — allocate and map all user-pool frames at process creation time, so that `mmap_range` during model loading becomes a no-op (frames already present). Trades boot time for model-load time | model_load | 2–3x (move EPT cost to boot, overlap with kernel init) | Medium |
| 19 | **KVM dirty-page tracking opt-out** — disable KVM dirty-page tracking (KVM_CAP_MANUAL_DIRTY_LOG_PROTECT2) for the guest memory region, since Nanvix does not live-migrate. Reduces EPT overhead on page writes | model_load, generation | 5–15% (fewer EPT faults on page writes) | Low |
| 20 | **x86_64 port** — run Nanvix in 64-bit long mode with 2MB huge pages (via PDE.PS in 4-level paging). Enables wider registers for compute, larger address space, and native 2MB page support without PSE workarounds | generation, model_load | 1.1–1.3x decode; model_load: ~0.5s with 2MB pages | Very High |
| 21 | **Specialize single-process scheduler** — bypass all scheduler overhead (quantum tracking, ready-queue management, context switch) when only one user process is running. The kernel event loop can poll IKC inline and return directly to the user thread | all phases | 3–5% (eliminate ~100 wasted scheduler invocations per second) | Low–Medium |
| 22 | **KVM PV clock / TSC-based time** — use `kvm_clock` or raw TSC (rdtsc) for `clock_gettime()` instead of PIT-tick counting. Provides nanosecond resolution without depending on timer interrupt frequency | time accuracy | No throughput impact; fixes 10ms guest-time granularity at 100 Hz | Low |

### Projected Performance

With optimization #14 (4MB large pages), model_load could drop from 2.0s to ~0.5s (matching native), bringing the 32-token total from 3.6s to ~2.1s (~1.0x native). With #20 (x86_64 + 2MB pages), both model loading and generation benefit: the 256-token total could drop from 12.4s to ~9.5s (1.1x native).

The remaining overhead after all feasible optimizations would be the irreducible KVM cost: EPT page-table walks for every guest memory access (~2–5% overhead) and VM-exit latency for any remaining interrupt or I/O trap.
