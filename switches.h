#pragma once

// this file contains the various compile time settings/switches

// can be tuned as per need
// 256 works best for Maxwell
// 384 best for newer chips!
#define BLOCK_SIZE 384

// limit max used registers to 64 for some kernels
// improves occupancy and performance (but doesn't work with debug info or debug builds)
#define LIMIT_REGISTER_USE 1

// 3 works best with 384 block size on new chips
#define  MIN_BLOCKS_PER_MP 3

// preallocated memory size (for holding the perft tree in GPU memory)
#define PREALLOCATED_MEMORY_SIZE (16 * 1024 * 1024 * 1024ull)

// use combined magic entry struct on GPU (mask + magic in one 32-byte struct)
// merges sqBishopAttacksMasked/sqRookAttacksMasked with FancyMagicEntry for
// single cache-line access instead of two separate loads
#define USE_COMBINED_MAGIC_GPU 1

// transposition table settings
#define USE_TT 1                     // master switch for transposition tables
#define HASH_IN_LEAF_KERNEL 0        // TT probe/store in fused 2-level leaf kernel
#define DEVICE_TT_BUDGET_MB 0        // GPU memory budget for device TTs (MB). 0 = auto (50% of free VRAM)
#define HOST_TT_BUDGET_MB 0          // host memory budget for host TTs (MB). 0 = auto (90% of system RAM)

// verbose diagnostics: call size/time histograms, per-BFS-level stats, progress reporting
#define VERBOSE_LOGGING 0


#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#ifndef CPU_FORCE_INLINE
#ifdef __linux__
    #define CPU_FORCE_INLINE inline
#else
    #define CPU_FORCE_INLINE __forceinline
#endif
#endif
