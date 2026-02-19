# perft_gpu Development Notes

## Git
- Push to `https://github.com/ankan-ban/perft_gpu_2026` (NOT perft_gpu)
- Never add `Co-Authored-By` lines to commit messages
- Always do clean rebuilds (`--clean-first`) after editing headers - NVCC doesn't reliably track transitive header dependencies

## Architecture
- Host-driven BFS with CUDA kernels per phase per level
- QuadBitBoard (4x uint64 = 32 bytes) + GameState (uint16 = 2 bytes)
- Bump allocator from single pre-allocated GPU buffer (no per-level cudaMalloc)
- CUB InclusiveSum (in-place), merge-path interval expand, no CDP

## Performance (RTX 4090, 16GB buffer)
- Startpos perft 9: ~503B nps (4.85s)
- Position 2 perft 7: ~815B nps (0.46s)

## Kernel Bottleneck Analysis (from Nsight Compute profiling)

### makeMove_and_perft_leaf_kernel (leaf, longest running)
- Compute-bound at 73% SM utilization, IPC 2.76
- Healthy - not much to optimize without changing the algorithm

### generate_moves_kernel (2nd longest, was the main optimization target)
- Originally had catastrophic scattered stores: 2-byte CMoves written to prefix-sum-indexed global offsets
- Only 2/32 bytes per sector utilized (6.25%), 85% wasted traffic, L2 bottleneck at 77%
- **Fixed with shared memory staging**: generate into per-CTA shared buffer, then cooperative coalesced copy to global. Gave +25-33% overall speedup.

### mp_interval_expand_kernel (3rd)
- Latency-bound from syncthreads + shared memory access. Hard to improve.

## Optimization Attempts and Results

### Shared memory staging for generate_moves_kernel [COMMITTED - big win]
- Generate moves into `__shared__ uint16[BLOCK_SIZE * 48]`, then all threads cooperatively copy to global
- Store sector utilization went from 6.25% to ~100%
- +25% startpos, +33% position 2

### Blackwell 256-bit load/store (ld.global.v4.u64) [COMMITTED - no measurable gain]
- PTX helpers loadQuadBB/storeQuadBB using v4.u64 on SM100+
- Falls back to default struct copy on older GPUs
- No measurable improvement on Blackwell RTX 6000 Pro

### Reducing shared memory buffer size (48 → 32 moves/thread) [REJECTED]
- Reduces shared mem from 37KB to 24KB, enabling 3 blocks/SM (75% occupancy vs 50%)
- But position 2 (avg ~47 moves) overflows the buffer frequently, falling back to scattered writes
- Net regression on position 2: 813B → 625B nps
- 40 moves/thread was a compromise (676B) but still worse than 48

### Local buffer intermediary for bank conflicts [REJECTED]
- Generate into per-thread CMove[48] local array, then copy to shared memory
- Intended to fix 2.6-way shared store bank conflicts (61% excess wavefronts)
- Local array (96 bytes/thread) spills to L1-backed local memory, adding ~8% overhead
- Net regression: 813B → 751B nps. The bank conflicts only cost ~1%.

### uint32 shared memory reads in copy loop [REJECTED]
- Read 2 CMoves as uint32 to avoid 2-way bank conflicts on shared loads
- Alignment issues when blockGlobalStart is odd (requires split aligned/unaligned paths)
- Marginal gain (+1.7% pos2) not worth the complexity. No gain on Blackwell.

### __launch_bounds__ + cudaSharedmemCarveoutMaxShared [REJECTED]
- Added __launch_bounds__(384, 3) to generate_moves_kernel + max shared carve-out
- Only +1-2% gain. Both RTX 4090 and Blackwell have same register/shared limits.

### Fused strided approach (makemove + countmoves + genmoves into fixed-stride buffer) [REJECTED]
- Eliminated generate_moves_kernel by having makemove_and_count also generate moves into moves[index * STRIDE]
- Theory: scattered writes overlap with heavy compute (makeMove+countMoves) from other warps
- Reality: strided writes are just as scattered (STRIDE*2 bytes between threads = different sectors)
- Plus extra memory usage (currentCount * STRIDE * 2 bytes per level)
- Net regression: -27% on both benchmarks. Shared memory staging is far superior.

## Key Lessons
- Shared memory staging is the right pattern for scattered small writes on GPU
- Local memory spilling (even L1-cached) is expensive - avoid per-thread arrays > ~32 bytes
- Bank conflicts in shared memory matter less than expected (~1% for 2.6-way conflicts)
- Compute/memory overlap via kernel fusion only helps when the fused kernel has enough compute to hide latency AND the memory access pattern actually improves
- Fixed-stride buffers don't improve coalescing - the stride between threads is what matters, not the stride within a thread
