#pragma once

// this file contains the various compile time settings/switches

// can be tuned as per need
// 256 works best for Maxwell
// 384 best for newer chips!
#define BLOCK_SIZE 416

// limit max used registers to 64 for some kernels
// improves occupancy and performance (but doesn't work with debug info or debug builds)
#define LIMIT_REGISTER_USE 1

// 3 works best with 384 block size on new chips
#define  MIN_BLOCKS_PER_MP 3

// preallocated memory size (for holding the perft tree in GPU memory)
#define PREALLOCATED_MEMORY_SIZE (16 * 1024 * 1024 * 1024ull)

// use fused final two-ply leaf kernel instead of materializing the last move frontier
#define USE_FUSED_2LEVEL_LEAF 1

// generated child move buffer inside the fused final leaf
#define FUSED_CHILD_MOVES_CAP 80

// use combined magic entry struct on GPU (mask + magic in one 32-byte struct)
// merges sqBishopAttacksMasked/sqRookAttacksMasked with FancyMagicEntry for
// single cache-line access instead of two separate loads
#define USE_COMBINED_MAGIC_GPU 1

// transposition table settings (runtime flag: default on, disable with -nott)
#define HASH_IN_LEAF_KERNEL 1        // TT probe/store in fused 2-level leaf kernel
#define DEVICE_TT_BUDGET_MB 0        // GPU memory budget for device TTs (MB). 0 = auto (99% of free VRAM)
#define HOST_TT_BUDGET_MB 0          // host memory budget for host TTs (MB). 0 = auto (90% of system RAM)

// Shallow TT for remaining-depth 3 — 8B packed entry (high-40 hash | 24-bit count)
// instead of the 16B XOR-locked TTEntry. Halves entry size (2x entries in same
// memory budget, lower load factor) and halves probe/store bandwidth at the
// expense of dropping hash.hi cross-verification (collision resistance 128->64
// bit; still negligible at TT sizes <= 2^32). Depth-3 perft fits 24-bit count
// safely (218^3 = 10.4M < 2^24).
// Set to 0 to fall back to the regular 16B TTTable at depth 3.
#define USE_SHALLOW_TT_DEPTH3 1

// verbose diagnostics: call size/time histograms, per-BFS-level stats, progress reporting
#define VERBOSE_LOGGING 0


#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#ifndef CPU_FORCE_INLINE
#ifdef _MSC_VER
    #define CPU_FORCE_INLINE __forceinline
#else
    #define CPU_FORCE_INLINE inline
#endif
#endif
