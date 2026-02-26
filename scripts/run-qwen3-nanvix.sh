#!/usr/bin/env bash
# run-qwen3-nanvix.sh — Build and run Qwen3-0.6B-Q4_K_M on Nanvix
#
# This script performs all steps needed to run the Qwen3-0.6B model on Nanvix:
#   1. Configures nanvix-copilot memory for large-model inference
#   2. Builds nanvix-copilot from clean
#   3. Downloads the model (if not already present)
#   4. Builds llama.cpp static libraries for Nanvix
#   5. Compiles and links the inference program
#   6. Runs inference via nanvixd.elf
#
# Prerequisites:
#   - nanvix-copilot checked out at NANVIX_COPILOT_DIR (default: ~/src/nanvix/nanvix-copilot)
#   - nanvix-copilot on the enhancement-libs-sysalloc branch (for unified heap and fast IPC)
#   - KVM available (/dev/kvm)
#
# Usage:
#   ./scripts/run-qwen3-nanvix.sh [options]
#
# Options:
#   -d DIR    Path to nanvix-copilot directory (default: ~/src/nanvix/nanvix-copilot)
#   -n N      Number of tokens to generate (default: 8)
#   -p TEXT   Prompt text (default: "Hello")
#   -s        Skip nanvix-copilot rebuild (use existing sysroot)
#   -m        Skip model download (assume already in place)
#   -h        Show this help message

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

LLAMA_CPP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NANVIX_COPILOT_DIR="${HOME}/src/nanvix/nanvix-copilot"
NUM_TOKENS=8
PROMPT="Hello"
SKIP_NANVIX_BUILD=false
SKIP_MODEL_DOWNLOAD=false
USE_MEMFS=false

MODEL_NAME="qwen3-0.6b-q4_k_m.gguf"
MODEL_URL="https://huggingface.co/rippertnt/Qwen3-0.6B-Q4_K_M-GGUF/resolve/main/${MODEL_NAME}"
MODEL_SIZE=484220000  # expected bytes

# Nanvix memory configuration for Qwen3-0.6B
KERNEL_MEMORY_SIZE=2147483648       # 2 GB VM
USER_HEAP_CAPACITY_MB=1008          # 1008 MB heap

# =============================================================================
# Argument parsing
# =============================================================================

usage() {
    sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# //' | sed 's/^#//'
    exit 0
}

while getopts "d:n:p:smfh" opt; do
    case $opt in
        d) NANVIX_COPILOT_DIR="$OPTARG" ;;
        n) NUM_TOKENS="$OPTARG" ;;
        p) PROMPT="$OPTARG" ;;
        s) SKIP_NANVIX_BUILD=true ;;
        m) SKIP_MODEL_DOWNLOAD=true ;;
        f) USE_MEMFS=true ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Derived paths
TOOLCHAIN="${NANVIX_COPILOT_DIR}/toolchain"
SYSROOT="${NANVIX_COPILOT_DIR}/sysroot"
BUILD_DIR="${LLAMA_CPP_DIR}/build-nanvix"

# =============================================================================
# Helpers
# =============================================================================

log()   { echo -e "\033[1;32m==>\033[0m $*"; }
warn()  { echo -e "\033[1;33m==> WARNING:\033[0m $*"; }
fail()  { echo -e "\033[1;31m==> ERROR:\033[0m $*" >&2; exit 1; }

check_file() {
    [ -f "$1" ] || fail "Required file not found: $1"
}

# =============================================================================
# Preflight checks
# =============================================================================

log "Preflight checks..."

[ -d "$NANVIX_COPILOT_DIR" ] || fail "nanvix-copilot not found at $NANVIX_COPILOT_DIR (use -d to override)"
[ -d "$TOOLCHAIN" ]          || fail "Toolchain not found at $TOOLCHAIN"
check_file "$TOOLCHAIN/bin/i686-nanvix-g++"
check_file "$TOOLCHAIN/bin/clang++"
check_file "$TOOLCHAIN/bin/clang"

if [ ! -e /dev/kvm ]; then
    fail "/dev/kvm not found — KVM is required to run nanvixd.elf"
fi

BRANCH=$(cd "$NANVIX_COPILOT_DIR" && git branch --show-current 2>/dev/null || echo "unknown")
if [[ "$BRANCH" != enhancement-libs-sysalloc && "$BRANCH" != feature-llama-perf ]]; then
    warn "nanvix-copilot is on branch '$BRANCH', not 'enhancement-libs-sysalloc' or 'feature-llama-perf'."
    warn "The unified heap and IPC performance fixes are required."
    warn "Switch with: cd $NANVIX_COPILOT_DIR && git checkout enhancement-libs-sysalloc"
fi

log "  llama.cpp dir:       $LLAMA_CPP_DIR"
log "  nanvix-copilot dir:  $NANVIX_COPILOT_DIR"
log "  toolchain:           $TOOLCHAIN"
log "  branch:              $BRANCH"

# =============================================================================
# Step 1: Configure nanvix-copilot memory
# =============================================================================

if [ "$SKIP_NANVIX_BUILD" = false ]; then
    log "Step 1/6: Configuring nanvix-copilot memory..."

    KERNEL_CONFIG="$NANVIX_COPILOT_DIR/build/kernel_config.toml"
    CONFIG_LIB="$NANVIX_COPILOT_DIR/src/libs/config/src/lib.rs"

    check_file "$KERNEL_CONFIG"
    check_file "$CONFIG_LIB"

    # kernel_config.toml: memory_size = 2 GB
    sed -i "s/^memory_size = [0-9]*/memory_size = ${KERNEL_MEMORY_SIZE}/" "$KERNEL_CONFIG"
    grep -q "memory_size = ${KERNEL_MEMORY_SIZE}" "$KERNEL_CONFIG" \
        || fail "Failed to set memory_size in $KERNEL_CONFIG"
    log "  kernel_config.toml: memory_size = ${KERNEL_MEMORY_SIZE}"

    # config/lib.rs: USER_HEAP_CAPACITY = 1008 MB
    sed -i "s/pub const USER_HEAP_CAPACITY: usize = [^;]*/pub const USER_HEAP_CAPACITY: usize = ${USER_HEAP_CAPACITY_MB} * crate::constants::MEGABYTE/" "$CONFIG_LIB"
    grep -q "USER_HEAP_CAPACITY: usize = ${USER_HEAP_CAPACITY_MB} \* crate::constants::MEGABYTE" "$CONFIG_LIB" \
        || fail "Failed to set USER_HEAP_CAPACITY in $CONFIG_LIB"
    log "  config/lib.rs: USER_HEAP_CAPACITY = ${USER_HEAP_CAPACITY_MB} MB"

# =============================================================================
# Step 2: Build nanvix-copilot
# =============================================================================

    log "Step 2/6: Building nanvix-copilot (cargo clean + release build, LOG_LEVEL=panic)..."
    log "  This takes several minutes..."

    cd "$NANVIX_COPILOT_DIR"
    cargo clean 2>&1 | tail -1
    MEMFS_FLAG=""
    if [ "$USE_MEMFS" = true ]; then
        MEMFS_FLAG="MEMFS=yes"
        log "  Building with MEMFS=yes (in-memory FAT32 filesystem)"
    fi
    ./z build -- RELEASE=yes LOG_LEVEL=panic $MEMFS_FLAG 2>&1 | tail -5
    make install RELEASE=yes LOG_LEVEL=panic 2>&1 | tail -3

    check_file "$SYSROOT/lib/libposix.a"
    check_file "$SYSROOT/bin/nanvixd.elf"
    log "  nanvix-copilot built and installed to $SYSROOT"
else
    log "Step 1-2/6: Skipping nanvix-copilot build (-s flag)"
    check_file "$SYSROOT/lib/libposix.a"
    check_file "$SYSROOT/bin/nanvixd.elf"
fi

# =============================================================================
# Step 3: Download model
# =============================================================================

MODEL_PATH="$SYSROOT/$MODEL_NAME"

if [ "$SKIP_MODEL_DOWNLOAD" = false ]; then
    log "Step 3/6: Downloading model..."
    if [ -f "$MODEL_PATH" ] && [ "$(wc -c < "$MODEL_PATH")" -eq "$MODEL_SIZE" ]; then
        log "  Model already present at $MODEL_PATH (${MODEL_SIZE} bytes) — skipping download"
    else
        log "  Downloading $MODEL_NAME (~462 MB)..."
        curl -L --progress-bar -o "$MODEL_PATH" "$MODEL_URL"
        ACTUAL_SIZE=$(wc -c < "$MODEL_PATH")
        if [ "$ACTUAL_SIZE" -ne "$MODEL_SIZE" ]; then
            fail "Model size mismatch: expected ${MODEL_SIZE}, got ${ACTUAL_SIZE}"
        fi
        log "  Downloaded: $MODEL_PATH (${ACTUAL_SIZE} bytes)"
    fi
else
    log "Step 3/6: Skipping model download (-m flag)"
    check_file "$MODEL_PATH"
fi

# =============================================================================
# Step 4: Build llama.cpp static libraries
# =============================================================================

log "Step 4/6: Building llama.cpp static libraries..."
cd "$LLAMA_CPP_DIR"

make -f Makefile.nanvix CONFIG_NANVIX=y \
    NANVIX_TOOLCHAIN="$TOOLCHAIN" \
    NANVIX_HOME="$SYSROOT" all 2>&1 | tail -5

check_file "$BUILD_DIR/src/libllama.a"
check_file "$BUILD_DIR/ggml/src/libggml.a"
check_file "$BUILD_DIR/ggml/src/libggml-base.a"
check_file "$BUILD_DIR/ggml/src/libggml-cpu.a"
log "  Static libraries built"

# =============================================================================
# Step 5: Compile and link llama_simple.elf
# =============================================================================

log "Step 5/6: Compiling and linking llama_simple.elf..."

# Optional memfs define for in-memory model loading
MEMFS_CFLAGS=""
if [ "$USE_MEMFS" = true ]; then
    MEMFS_CFLAGS="-DNANVIX_MEMFS"
    log "  Compiling with -DNANVIX_MEMFS (in-memory FAT32 model loading)"
fi

# Compile nanvix_simple.cpp (with AVX/SSE4.2 SIMD support)
"$TOOLCHAIN/bin/clang++" --target=i686-unknown-nanvix \
    --sysroot="$TOOLCHAIN/i686-nanvix" -B"$TOOLCHAIN/bin" \
    -std=c++17 -O2 -march=pentium4 -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 \
    -mavx -mavx2 -mfma -mf16c -mpopcnt -mfpmath=sse \
    $MEMFS_CFLAGS \
    -Iinclude -Iggml/include -idirafter nanvix/stubs/include \
    -c nanvix/nanvix_simple.cpp -o nanvix/nanvix_simple.o

# Compile stubs
"$TOOLCHAIN/bin/clang" --target=i686-unknown-nanvix \
    --sysroot="$TOOLCHAIN/i686-nanvix" -B"$TOOLCHAIN/bin" \
    -m32 -O2 -march=pentium4 -msse -msse2 -mfpmath=sse \
    -c nanvix/stubs/nanvix_stubs.c -o nanvix/stubs/nanvix_stubs.o

# Link
"$TOOLCHAIN/bin/i686-nanvix-g++" -static -nostdlib \
    -T"$SYSROOT/lib/user.ld" -Tnanvix/nanvix_extra.ld -Wl,-z,noexecstack \
    -Wl,--wrap=ggml_fopen \
    nanvix/nanvix_simple.o nanvix/stubs/nanvix_stubs.o \
    -L"$BUILD_DIR/src" -lllama \
    -L"$BUILD_DIR/ggml/src" -lggml -lggml-base -lggml-cpu \
    -Wl,--start-group \
    "$TOOLCHAIN/lib/i686-unknown-nanvix/libc++.a" \
    "$TOOLCHAIN/lib/i686-unknown-nanvix/libc++abi.a" \
    "$TOOLCHAIN/lib/i686-unknown-nanvix/libunwind.a" \
    "$SYSROOT/lib/libposix.a" \
    "$TOOLCHAIN/i686-nanvix/lib/libc.a" \
    "$TOOLCHAIN/i686-nanvix/lib/libm.a" \
    -lgcc -Wl,--end-group \
    -o llama_simple.elf

check_file "llama_simple.elf"
ELF_SIZE=$(wc -c < llama_simple.elf)
log "  llama_simple.elf built (${ELF_SIZE} bytes)"

# =============================================================================
# Step 6: Run inference
# =============================================================================

log "Step 6/6: Running Qwen3-0.6B inference on Nanvix..."
log "  Prompt:     \"$PROMPT\""
log "  Tokens:     $NUM_TOKENS"

cd "$SYSROOT"

if [ "$USE_MEMFS" = true ]; then
    # Create FAT32 image containing the model for in-memory loading via RAMFS.
    FAT_IMAGE="${MODEL_PATH%.gguf}.fat"
    if [ ! -f "$FAT_IMAGE" ] || [ "$FAT_IMAGE" -ot "$MODEL_PATH" ]; then
        log "  Creating FAT32 image from model..."
        "$LLAMA_CPP_DIR/scripts/create-model-fat32.sh" "$MODEL_PATH" "$FAT_IMAGE"
    else
        log "  FAT32 image already up-to-date: $FAT_IMAGE"
    fi
    log "  Using memfs: model served from in-memory FAT32 via -ramfs"
    log ""
    timeout --foreground 3600 ./bin/nanvixd.elf -ramfs "$FAT_IMAGE" -- \
        "$LLAMA_CPP_DIR/llama_simple.elf" \
        -m "$MODEL_NAME" -n "$NUM_TOKENS" "$PROMPT"
else
    log "  Model load takes ~2 minutes via virtio-fs, then inference"
    log ""
    timeout --foreground 3600 ./bin/nanvixd.elf -- "$LLAMA_CPP_DIR/llama_simple.elf" \
        -m "$MODEL_NAME" -n "$NUM_TOKENS" "$PROMPT"
fi

log "Done!"
