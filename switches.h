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


#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif
