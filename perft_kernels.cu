// All GPU kernels, device helpers, host-driven BFS, and move generator init
#include <cub/cub.cuh>
#include "MoveGeneratorBitboard.h"
#include "launcher.h"
#include "zobrist.h"
#include "tt.h"
#define ALIGN_UP(addr, align)   (((addr) + (align) - 1) & (~((align) - 1)))
#define MEM_ALIGNMENT 32    // 32-byte alignment needed for v4.u64 loads/stores
// -------------------------------------------------------------------------
// 256-bit load/store helpers for QuadBitBoard
// On Blackwell+ (SM100), use ld.global.v4.u64 / st.global.v4.u64 for
// single-instruction 32-byte transfers. Falls back to default on older GPUs.
// -------------------------------------------------------------------------
__device__ __forceinline__ QuadBitBoard loadQuadBB(const QuadBitBoard *ptr)
{
    QuadBitBoard ret;
#if __CUDA_ARCH__ >= 1000
    asm volatile("ld.global.v4.u64 {%0,%1,%2,%3}, [%4];"
        : "=l"(ret.bb[0]), "=l"(ret.bb[1]), "=l"(ret.bb[2]), "=l"(ret.bb[3])
        : "l"(ptr));
#else
    ret = *ptr;
#endif
    return ret;
}
__device__ __forceinline__ void storeQuadBB(QuadBitBoard *ptr, const QuadBitBoard &val)
{
#if __CUDA_ARCH__ >= 1000
    asm volatile("st.global.v4.u64 [%0], {%1,%2,%3,%4};"
        :: "l"(ptr), "l"(val.bb[0]), "l"(val.bb[1]), "l"(val.bb[2]), "l"(val.bb[3]));
#else
    *ptr = val;
#endif
}
// makes the given move on the given position
__device__ __forceinline__ void makeMove(QuadBitBoard *pos, GameState *gs, CMove move, int chance)
{
    if (chance == BLACK)
    {
        MoveGeneratorBitboard::makeMove<BLACK>(pos, gs, move);
    }
    else
    {
        MoveGeneratorBitboard::makeMove<WHITE>(pos, gs, move);
    }
}
__host__ __device__ __forceinline__ uint32 countMoves(QuadBitBoard *pos, GameState *gs, uint8 color)
{
    if (color == BLACK)
    {
        return MoveGeneratorBitboard::countMoves<BLACK>(pos, gs);
    }
    else
    {
        return MoveGeneratorBitboard::countMoves<WHITE>(pos, gs);
    }
}
// fast reduction for the warp
__device__ __forceinline__ void warpReduce(int &x)
{
    #pragma unroll
    for(int mask = 16; mask > 0 ; mask >>= 1)
        x += __shfl_xor_sync(0xFFFFFFFF, x, mask);
}
// -------------------------------------------------------------------------
// Merge-path interval expand kernels
// -------------------------------------------------------------------------
#define IE_VT 7
#define IE_NV (BLOCK_SIZE * IE_VT)
// Phase 1: Compute merge-path partition points.
__global__ void merge_path_partitions(
    const int *inclPrefixSums,
    int aCount,
    int bCount,
    int numPartitions,
    int *partitions)
{
    int partition = blockIdx.x * blockDim.x + threadIdx.x;
    if (partition >= numPartitions)
        return;
    long long diag_ll = (long long)partition * IE_NV;
    long long total_ll = (long long)aCount + bCount;
    int diag = (int)(diag_ll < total_ll ? diag_ll : total_ll);
    int lo = (diag - bCount) > 0 ? (diag - bCount) : 0;
    int hi = diag < aCount ? diag : aCount;
    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        int b_idx = diag - 1 - mid;
        if (b_idx < 0 || mid < inclPrefixSums[b_idx])
            lo = mid + 1;
        else
            hi = mid;
    }
    partitions[partition] = lo;
}
// Phase 2: Merge-path interval expand kernel.
__global__ void mp_interval_expand_kernel(
    int *output,
    const int *inclPrefixSums,
    int aCount,
    int bCount,
    const int *partitions)
{
    __shared__ int shared[BLOCK_SIZE * (IE_VT + 1)];
    int block = blockIdx.x;
    int tid = threadIdx.x;
    int a0 = partitions[block];
    int a1 = partitions[block + 1];
    long long d0_ll = (long long)block * IE_NV;
    long long d1_ll = (long long)(block + 1) * IE_NV;
    long long total_ll = (long long)aCount + bCount;
    int diag0 = (int)(d0_ll < total_ll ? d0_ll : total_ll);
    int diag1 = (int)(d1_ll < total_ll ? d1_ll : total_ll);
    int b0 = diag0 - a0;
    int b1 = diag1 - a1;
    int aCount_cta = a1 - a0;
    int bCount_cta = b1 - b0;
    int *s_output = shared;
    int *s_prefix = shared + aCount_cta;
    for (int i = tid; i < bCount_cta; i += BLOCK_SIZE)
        s_prefix[i] = inclPrefixSums[b0 + i];
    __syncthreads();
    int merge_total = aCount_cta + bCount_cta;
    int diag_local_raw = IE_VT * tid;
    int diag_local = diag_local_raw < merge_total ? diag_local_raw : merge_total;
    int lo = (diag_local - bCount_cta) > 0 ? (diag_local - bCount_cta) : 0;
    int hi = diag_local < aCount_cta ? diag_local : aCount_cta;
    while (lo < hi)
    {
        int mid = (lo + hi) / 2;
        int b_idx = diag_local - 1 - mid;
        if (b_idx < 0 || (a0 + mid) < s_prefix[b_idx])
            lo = mid + 1;
        else
            hi = mid;
    }
    int a_cur = lo;
    int b_cur = diag_local - a_cur;
    int b_val = (b_cur < bCount_cta) ? s_prefix[b_cur] : INT_MAX;
    #pragma unroll
    for (int i = 0; i < IE_VT; i++)
    {
        bool consumeA = (a_cur < aCount_cta) &&
                         ((b_cur >= bCount_cta) || ((a0 + a_cur) < b_val));
        if (consumeA)
        {
            s_output[a_cur] = b0 + b_cur;
            a_cur++;
        }
        else if (b_cur < bCount_cta)
        {
            b_cur++;
            b_val = (b_cur < bCount_cta) ? s_prefix[b_cur] : INT_MAX;
        }
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < IE_VT; i++)
    {
        int index = BLOCK_SIZE * i + tid;
        if (index < aCount_cta)
        {
            output[a0 + index] = s_output[index];
        }
    }
}
// Host-side memory allocator from preallocated GPU buffer
struct GpuBumpAllocator
{
    void *base;
    size_t offset;
    size_t capacity;
    GpuBumpAllocator(void *buf, size_t cap) : base(buf), offset(0), capacity(cap) {}
    template<typename T>
    T* alloc(size_t count)
    {
        size_t bytes = count * sizeof(T);
        bytes = ALIGN_UP(bytes, MEM_ALIGNMENT);
        if (offset + bytes > capacity)
        {
            printf("\nOOM: need %llu bytes, only %llu available\n", (unsigned long long)(offset + bytes), (unsigned long long)capacity);
            return nullptr;
        }
        T* ptr = (T*)((uint8*)base + offset);
        offset += bytes;
        return ptr;
    }
    void reset() { offset = 0; }
};
// -------------------------------------------------------------------------
// GPU kernels for the host-driven BFS perft
// -------------------------------------------------------------------------
// Kernel: make move on parent board, produce child board, compute hash, count child moves, probe TT
#if LIMIT_REGISTER_USE == 1
__launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_MP)
#endif
__global__ void makemove_and_count_moves_kernel(QuadBitBoard *parentBoards, GameState *parentStates,
                                                 Hash128 *parentHashes,
                                                 int *indices, CMove *moves,
                                                 QuadBitBoard *outPositions, GameState *outStates,
                                                 Hash128 *outHashes,
                                                 int *moveCounts, uint64 *ttHitCounts,
                                                 TTTable deviceTT,
                                                 int nThreads, uint8 color)
{
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
    int nMoves = 0;
    uint64 ttHit = 0;
    if (index < nThreads)
    {
        int parentIndex = indices[index];
        QuadBitBoard pos = loadQuadBB(&parentBoards[parentIndex]);
        GameState gs = parentStates[parentIndex];
        Hash128 hash = parentHashes[parentIndex];
        CMove move = moves[index];
        // Pre-move state for hash update
        uint8 srcPiece = getPieceAt(&pos, move.getFrom());
        uint8 capPiece = getPieceAt(&pos, move.getTo());
        uint8 oldCastleRaw = gs.raw;
        uint8 oldEP = gs.enPassent;
        makeMove(&pos, &gs, move, color);
        hash = updateHashAfterMove(hash, move, color,
            srcPiece, capPiece, oldCastleRaw, gs.raw, oldEP, gs.enPassent);
#if USE_TT
        // Probe device TT
        uint64 ttCount;
        if (ttProbe(deviceTT, hash, &ttCount))
        {
            ttHit = ttCount;
            nMoves = 0;  // Don't expand this position
        }
        else
#endif
        {
            nMoves = countMoves(&pos, &gs, !color);
        }
        storeQuadBB(&outPositions[index], pos);
        outStates[index] = gs;
        outHashes[index] = hash;
        moveCounts[index] = nMoves;
        if (ttHitCounts) ttHitCounts[index] = ttHit;
    }
}
// Kernel: generate moves for each position (uses inclusive prefix sums for offsets)
// Uses shared memory staging for coalesced global writes.
// Without this, each thread writes 2-byte CMoves to scattered global addresses
// (6.25% sector utilization, 85% wasted traffic). With staging, the CTA
// cooperatively copies the compacted move buffer to global memory at 100% utilization.
#define SMEM_MOVES_CAPACITY (BLOCK_SIZE * 64)
__global__ void generate_moves_kernel(QuadBitBoard *positions, GameState *states,
                                       CMove *generatedMovesBase, int *inclPrefixSums, int nThreads, uint8 color)
{
    __shared__ CMove smemMoves[SMEM_MOVES_CAPACITY];
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    // Compute this CTA's output range from the inclusive prefix sums
    int firstIndex = blockIdx.x * blockDim.x;
    int lastIndex = firstIndex + blockDim.x - 1;
    if (lastIndex >= nThreads) lastIndex = nThreads - 1;
    int blockGlobalStart = (firstIndex == 0) ? 0 : inclPrefixSums[firstIndex - 1];
    int totalBlockMoves = inclPrefixSums[lastIndex] - blockGlobalStart;
    // Per-thread: derive local offset within shared buffer from global prefix sums
    int localOffset = 0;
    int myMoveCount = 0;
    QuadBitBoard pos;
    GameState gs;
    if (index < nThreads)
    {
        pos = loadQuadBB(&positions[index]);
        gs = states[index];
        int myGlobalStart = (index == 0) ? 0 : inclPrefixSums[index - 1];
        myMoveCount = inclPrefixSums[index] - myGlobalStart;
        localOffset = myGlobalStart - blockGlobalStart;
    }
    if (totalBlockMoves <= SMEM_MOVES_CAPACITY)
    {
        // Fast path: generate into shared memory, then coalesced copy to global
        if (index < nThreads && myMoveCount > 0)
        {
            generateMoves(&pos, &gs, color, &smemMoves[localOffset]);
        }
        __syncthreads();
        // Cooperative coalesced copy from shared memory to global memory
        for (int i = tid; i < totalBlockMoves; i += blockDim.x)
        {
            generatedMovesBase[blockGlobalStart + i] = smemMoves[i];
        }
    }
    else
    {
        // Fallback for rare blocks with very many moves: direct scattered writes
        if (index < nThreads && myMoveCount > 0)
        {
            generateMoves(&pos, &gs, color, &generatedMovesBase[blockGlobalStart + localOffset]);
        }
    }
}
// Kernel: leaf level - make move and count, add to global perft counter
#if LIMIT_REGISTER_USE == 1
__launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_MP)
#endif
__global__ void makeMove_and_perft_leaf_kernel(QuadBitBoard *positions, GameState *states,
                                                Hash128 *hashes,
                                                int *indices, CMove *moves,
                                                uint64 *globalPerftCounter,
                                                uint64 *perThreadCounts,
                                                int nThreads, uint8 color)
{
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nThreads)
        return;
    uint32 boardIndex = indices[index];
    QuadBitBoard pos = loadQuadBB(&positions[boardIndex]);
    GameState gs = states[boardIndex];
    CMove move = moves[index];
    makeMove(&pos, &gs, move, color);
    // count moves at this position
    int nMoves = countMoves(&pos, &gs, !color);
    // Write per-thread count for upsweep
    if (perThreadCounts)
        perThreadCounts[index] = (uint64)nMoves;
    // warp-wide reduction before atomic add
    warpReduce(nMoves);
    int laneId = threadIdx.x & 0x1f;
    if (laneId == 0)
    {
        atomicAdd(globalPerftCounter, nMoves);
    }
}
// -------------------------------------------------------------------------
// Fused 2-level leaf from boards: each thread takes a board directly
// (no input move), generates all moves, and for each child makes the
// move and counts grandchildren. Used as dynamic OOM fallback.
// -------------------------------------------------------------------------
__global__ void perft_2level_from_board_kernel(QuadBitBoard *boards, GameState *states,
                                                uint64 *globalPerftCounter, int nThreads, uint8 color)
{
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nThreads)
        return;
    QuadBitBoard pos = loadQuadBB(&boards[index]);
    GameState gs = states[index];
    CMove childMoves[MAX_MOVES];
    int nChildMoves = generateMoves(&pos, &gs, color, childMoves);
    int totalCount = 0;
    for (int i = 0; i < nChildMoves; i++)
    {
        QuadBitBoard childPos = pos;
        GameState childGs = gs;
        makeMove(&childPos, &childGs, childMoves[i], color);
        totalCount += countMoves(&childPos, &childGs, !color);
    }
    warpReduce(totalCount);
    int laneId = threadIdx.x & 0x1f;
    if (laneId == 0)
    {
        atomicAdd(globalPerftCounter, (uint64)totalCount);
    }
}
// -------------------------------------------------------------------------
// Fused 2-level leaf kernel: replaces the last BFS level + leaf kernel.
// Each thread makes a move to get a child position, generates all legal
// moves for that child, then for each grandchild makes the move and counts.
// Saves massive memory (no need for the huge move/index arrays of the last
// BFS level) and improves cache locality (siblings processed together).
// -------------------------------------------------------------------------
__launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_MP)
__global__ void fused_2level_leaf_kernel(QuadBitBoard *positions, GameState *states,
                                          Hash128 *hashes,
                                          int *indices, CMove *moves,
                                          uint64 *globalPerftCounter,
                                          uint64 *perThreadCounts,
                                          TTTable leafTT,
                                          int nThreads, uint8 color)
{
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nThreads)
        return;
    // Load parent position and make the first move (level N -> level N-1)
    uint32 boardIndex = indices[index];
    QuadBitBoard pos = loadQuadBB(&positions[boardIndex]);
    GameState gs = states[boardIndex];
    CMove move = moves[index];
#if USE_TT && HASH_IN_LEAF_KERNEL
    Hash128 parentHash = hashes[boardIndex];
    // Pre-move state for hash update
    uint8 srcPiece = getPieceAt(&pos, move.getFrom());
    uint8 capPiece = getPieceAt(&pos, move.getTo());
    uint8 oldCastleRaw = gs.raw;
    uint8 oldEP = gs.enPassent;
#endif
    makeMove(&pos, &gs, move, color);
#if USE_TT && HASH_IN_LEAF_KERNEL
    Hash128 hash = updateHashAfterMove(parentHash, move, color,
        srcPiece, capPiece, oldCastleRaw, gs.raw, oldEP, gs.enPassent);
    // Probe TT for this position (remaining depth = bfsMinLevel - 1)
    uint64 ttCount;
    if (ttProbe(leafTT, hash, &ttCount))
    {
        if (perThreadCounts) perThreadCounts[index] = ttCount;
        // Still need warp-reduce + atomicAdd for global counter
        int tc = (int)ttCount;
        warpReduce(tc);
        if ((threadIdx.x & 0x1f) == 0)
            atomicAdd(globalPerftCounter, (uint64)tc);
        return;
    }
#endif
    // Generate all legal moves at level N-1 into local memory.
    // 218 moves covers the worst case
    CMove childMoves[218];
    uint8 childColor = !color;  // after makeMove, chance is flipped
    int nChildMoves = generateMoves(&pos, &gs, childColor, childMoves);
    // For each child move: make it and count grandchildren
    int totalCount = 0;
    for (int i = 0; i < nChildMoves; i++)
    {
        QuadBitBoard childPos = pos;
        GameState childGs = gs;
        makeMove(&childPos, &childGs, childMoves[i], childColor);
        totalCount += countMoves(&childPos, &childGs, !childColor);
    }
#if USE_TT && HASH_IN_LEAF_KERNEL
    // Store result in TT
    ttStore(leafTT, hash, (uint64)totalCount);
#endif
    // Write per-thread count for upsweep
    if (perThreadCounts)
        perThreadCounts[index] = (uint64)totalCount;
    // Warp-wide reduction before atomic add
    warpReduce(totalCount);
    int laneId = threadIdx.x & 0x1f;
    if (laneId == 0)
    {
        atomicAdd(globalPerftCounter, (uint64)totalCount);
    }
}
// -------------------------------------------------------------------------
// Upsweep kernels: propagate per-position counts back through BFS levels
// -------------------------------------------------------------------------
#if USE_TT
// -------------------------------------------------------------------------
// BFS-level deduplication: detect identical positions within a BFS level.
// Uses a hash table of packed (hashHi_32 | index) entries.
// Duplicates get moveCounts=0 and a redirect to the original position.
// -------------------------------------------------------------------------
// Write position's packed entry (upper 32 bits of hash.hi | 1-indexed position) to dedup table.
// Uses 1-indexed positions so that 0 = empty (distinguishes from valid entries).
__global__ void write_dedup_table(Hash128 *hashes, uint64 *dedupTable, uint64 dedupMask, int n)
{
    uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    Hash128 h = hashes[idx];
    uint64 slot = h.lo & dedupMask;
    uint64 packed = (h.hi & 0xFFFFFFFF00000000ULL) | (uint64)(idx + 1);  // 1-indexed
    dedupTable[slot] = packed;
}

// Check each position against the dedup table. If hash matches but index differs,
// mark as duplicate: moveCounts=0, redirect stores the original's index.
__global__ void check_duplicates(Hash128 *hashes, uint64 *dedupTable, uint64 dedupMask,
                                  int *moveCounts, uint64 *ttHitCounts, int *redirects, int n)
{
    uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    Hash128 h = hashes[idx];
    uint64 slot = h.lo & dedupMask;
    uint64 entry = dedupTable[slot];

    if (entry != 0)  // skip empty slots
    {
        uint32 storedHi32 = (uint32)(entry >> 32);
        uint32 storedIdx  = (uint32)(entry & 0xFFFFFFFF) - 1;  // back to 0-indexed
        uint32 myHi32 = (uint32)(h.hi >> 32);

        if (storedHi32 == myHi32 && storedIdx != idx)
        {
            // Duplicate detected — skip expansion, redirect to original
            moveCounts[idx] = 0;
            if (ttHitCounts) ttHitCounts[idx] = 0;
            redirects[idx] = (int)storedIdx;
            return;
        }
    }
    redirects[idx] = -1;
}
// After upsweep computes per-position counts, copy original's count to duplicates
__global__ void apply_redirects(uint64 *positionCounts, int *redirects, int n)
{
    uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int redir = redirects[idx];
    if (redir >= 0)
        positionCounts[idx] = positionCounts[redir];
}
// Sum each child's count into its parent using atomicAdd
__global__ void reduce_by_parent(uint64 *childCounts, int *childToParent,
                                  uint64 *parentCounts, int numChildren)
{
    uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numChildren) return;
    uint64 myCount = childCounts[idx];
    if (myCount > 0)
        atomicAdd(&parentCounts[childToParent[idx]], myCount);
}
// Add TT hit counts and store results in device TT
// redirects: if non-null, skip TT store for duplicate positions (redirect >= 0)
// to avoid overwriting the original's correct entry with the duplicate's zero
__global__ void add_tt_hits_and_store(uint64 *positionCounts, uint64 *ttHitCounts,
                                      Hash128 *posHashes, TTTable tt,
                                      int *redirects, int numPositions,
                                      int remainingDepth)
{
    uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPositions) return;
    uint64 totalCount = positionCounts[idx] + ttHitCounts[idx];
    positionCounts[idx] = totalCount;
    // Disable upsweep TT store — causes correctness issues at low launch depths
    // TODO: investigate root cause (32-bit dedup verification? TT collision?)
    // if (!redirects || redirects[idx] < 0)
    //     ttStore(tt, posHashes[idx], totalCount);
}
// Warp-reduce uint64 and atomically add to global counter
__global__ void reduce_sum_uint64(uint64 *values, uint64 *globalSum, int n)
{
    uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64 val = (idx < n) ? values[idx] : 0;
    // Warp reduction for uint64
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(globalSum, val);
}
#endif // USE_TT
// Per-level saved state for the upsweep
struct BFSLevelSave
{
    int     *indicesToParent;   // maps positions at this level to parent level
    Hash128 *hashes;            // hash of each position at this level
    uint64  *ttHitCounts;       // TT hit counts (0 for misses)
    int     *redirects;         // dedup redirects: -1 or index of original position
    int     count;              // number of positions at this level
    int     remainingDepth;     // for TT indexing
};
// -------------------------------------------------------------------------
// Host-driven BFS perft
// -------------------------------------------------------------------------
// Run perft on GPU for a single position using host-driven breadth-first search
uint64 perft_gpu_host_bfs(QuadBitBoard *pos, GameState *gs, uint8 rootColor, int depth, void *gpuBuffer, size_t bufferSize)
{
    if (depth <= 0)
        return 1;
    if (depth == 1)
        return countMoves(pos, gs, rootColor);
    GpuBumpAllocator alloc(gpuBuffer, bufferSize);
    // Pre-allocate CUB temp storage once
    void *d_cubTemp = nullptr;
    size_t cubTempBytes = 0;
    cub::DeviceScan::InclusiveSum(d_cubTemp, cubTempBytes, (int*)nullptr, (int*)nullptr, 256 * 1024 * 1024);
    d_cubTemp = alloc.alloc<uint8>(cubTempBytes);
    // Copy root position to GPU (separate arrays)
    QuadBitBoard *d_rootPos = alloc.alloc<QuadBitBoard>(1);
    GameState *d_rootState = alloc.alloc<GameState>(1);
    cudaMemcpy(d_rootPos, pos, sizeof(QuadBitBoard), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rootState, gs, sizeof(GameState), cudaMemcpyHostToDevice);
    // Copy root hash to GPU
    Hash128 rootHash = computeHash(pos, gs, rootColor);
    Hash128 *d_rootHash = alloc.alloc<Hash128>(1);
    cudaMemcpy(d_rootHash, &rootHash, sizeof(Hash128), cudaMemcpyHostToDevice);
    // Generate root moves on CPU
    CMove rootMoves[MAX_MOVES];
    int rootMoveCount = generateMoves(pos, gs, rootColor, rootMoves);
    if (rootMoveCount == 0)
        return 0;
    // Copy root moves to GPU
    CMove *d_moves = alloc.alloc<CMove>(rootMoveCount);
    cudaMemcpy(d_moves, rootMoves, sizeof(CMove) * rootMoveCount, cudaMemcpyHostToDevice);
    // Create root indices (all point to index 0 = the root position)
    int *d_indices = alloc.alloc<int>(rootMoveCount);
    cudaMemset(d_indices, 0, sizeof(int) * rootMoveCount);
    // Perft counter on GPU
    uint64 *d_perftCounter = alloc.alloc<uint64>(1);
    cudaMemset(d_perftCounter, 0, sizeof(uint64));
    // Pre-allocate partition array for merge-path interval expand
    const int maxPartitions = 500000;
    int *d_partitions = alloc.alloc<int>(maxPartitions);
    QuadBitBoard *d_prevBoards = d_rootPos;
    GameState *d_prevStates = d_rootState;
    Hash128 *d_prevHashes = d_rootHash;
    int currentCount = rootMoveCount;
    // Use fused 2-level leaf for depth >= 4: saves memory and improves locality.
    // The 2-level leaf handles the last 2 levels, so BFS stops one level earlier.
    bool use2LevelLeaf = (depth >= 4);
    int bfsMinLevel = use2LevelLeaf ? 3 : 2;
    // Track color through BFS levels: root positions have rootColor,
    // first makeMove uses rootColor, producing positions with !rootColor, etc.
    uint8 levelColor = rootColor;
    // Per-level state saved for the upsweep
    const int MAX_BFS_LEVELS = 20;
    BFSLevelSave levelSaves[MAX_BFS_LEVELS];
    int numLevels = 0;
    // BFS loop: depth-1 down to bfsMinLevel are intermediate levels
    for (int level = depth - 1; level >= bfsMinLevel; level--)
    {
        int remainingDepth = level;
        // Allocate boards, states, hashes, move counts, TT hit counts
        QuadBitBoard *d_curBoards = alloc.alloc<QuadBitBoard>(currentCount);
        GameState *d_curStates = alloc.alloc<GameState>(currentCount);
        Hash128 *d_curHashes = alloc.alloc<Hash128>(currentCount);
        int *d_moveCounts = alloc.alloc<int>(currentCount);
        uint64 *d_ttHitCounts = nullptr;
#if USE_TT
        d_ttHitCounts = alloc.alloc<uint64>(currentCount);
#endif
        if (!d_curBoards || !d_curStates || !d_curHashes || !d_moveCounts)
        {
            printf("\nOOM during BFS at level %d with %d positions\n", level, currentCount);
            return 0;
        }
        // Step 1: make moves, compute hashes, count child moves, probe TT
        int nBlocks = (currentCount - 1) / BLOCK_SIZE + 1;
        TTTable curTT = {nullptr, 0};
#if USE_TT
        if (remainingDepth < MAX_TT_DEPTH)
            curTT = deviceTTs[remainingDepth];
#endif
        makemove_and_count_moves_kernel<<<nBlocks, BLOCK_SIZE>>>(d_prevBoards, d_prevStates,
                                                                   d_prevHashes,
                                                                   d_indices, d_moves,
                                                                   d_curBoards, d_curStates,
                                                                   d_curHashes,
                                                                   d_moveCounts, d_ttHitCounts,
                                                                   curTT,
                                                                   currentCount, levelColor);
        // BFS-level dedup: detect identical positions and skip expanding duplicates
        int *d_redirects = nullptr;
#if USE_TT
        if (currentCount > 1)
        {
            // Dedup table: 2× position count, rounded up to power of 2
            uint64 dedupSize = 1;
            while (dedupSize < (uint64)currentCount * 2) dedupSize <<= 1;
            uint64 *d_dedupTable = alloc.alloc<uint64>(dedupSize);
            d_redirects = alloc.alloc<int>(currentCount);

            if (d_dedupTable && d_redirects)
            {
                cudaMemset(d_dedupTable, 0, dedupSize * sizeof(uint64));  // 0 = empty
                uint64 dedupMask = dedupSize - 1;

                write_dedup_table<<<nBlocks, BLOCK_SIZE>>>(d_curHashes, d_dedupTable, dedupMask, currentCount);
                check_duplicates<<<nBlocks, BLOCK_SIZE>>>(d_curHashes, d_dedupTable, dedupMask,
                    d_moveCounts, d_ttHitCounts, d_redirects, currentCount);
            }
            else
            {
                d_redirects = nullptr;  // OOM, skip dedup
            }
        }
#endif
        // Save state for upsweep (before d_indices gets overwritten)
        levelSaves[numLevels].indicesToParent = d_indices;
        levelSaves[numLevels].hashes = d_curHashes;
        levelSaves[numLevels].ttHitCounts = d_ttHitCounts;
        levelSaves[numLevels].redirects = d_redirects;
        levelSaves[numLevels].count = currentCount;
        levelSaves[numLevels].remainingDepth = remainingDepth;
        numLevels++;
        // After makeMove, positions are now at the flipped color
        levelColor = !levelColor;
        // Step 2: in-place inclusive prefix sum
        cub::DeviceScan::InclusiveSum(d_cubTemp, cubTempBytes, d_moveCounts, d_moveCounts, currentCount);
        // Step 3: read back total
        int nextLevelCount = 0;
        cudaMemcpy(&nextLevelCount, d_moveCounts + currentCount - 1, sizeof(int), cudaMemcpyDeviceToHost);
        if (nextLevelCount == 0)
            break;
        // Allocate next level arrays
        int *d_newIndices = alloc.alloc<int>(nextLevelCount);
        CMove *d_newMoves = alloc.alloc<CMove>(nextLevelCount);
        if (!d_newIndices || !d_newMoves)
        {
            // Dynamic OOM fallback: use 2-level leaf from the boards we just created.
            if (level >= 2)
            {
                int nBlk = (currentCount - 1) / BLOCK_SIZE + 1;
                perft_2level_from_board_kernel<<<nBlk, BLOCK_SIZE>>>(d_curBoards, d_curStates,
                                                                      d_perftCounter, currentCount, levelColor);
                cudaDeviceSynchronize();
                uint64 result = 0;
                cudaMemcpy(&result, d_perftCounter, sizeof(uint64), cudaMemcpyDeviceToHost);
                return result;
            }
            printf("\nOOM for next level with %d moves\n", nextLevelCount);
            return 0;
        }
        // Step 4: merge-path interval expand
        int mergeItems = nextLevelCount + currentCount;
        int numExpandCTAs = (mergeItems + IE_NV - 1) / IE_NV;
        int numPartitions = numExpandCTAs + 1;
        if (numPartitions > maxPartitions)
        {
            printf("\nToo many partitions (%d) for pre-allocated buffer (%d)\n", numPartitions, maxPartitions);
            return 0;
        }
        int partBlocks = (numPartitions + 127) / 128;
        merge_path_partitions<<<partBlocks, 128>>>(d_moveCounts, nextLevelCount, currentCount,
                                                    numPartitions, d_partitions);
        mp_interval_expand_kernel<<<numExpandCTAs, BLOCK_SIZE>>>(d_newIndices, d_moveCounts,
                                                                   nextLevelCount, currentCount, d_partitions);
        // Step 5: generate moves (positions at this level have color = levelColor)
        generate_moves_kernel<<<nBlocks, BLOCK_SIZE>>>(d_curBoards, d_curStates, d_newMoves, d_moveCounts, currentCount, levelColor);
        // Advance to next level
        d_prevBoards = d_curBoards;
        d_prevStates = d_curStates;
        d_prevHashes = d_curHashes;
        d_indices = d_newIndices;
        d_moves = d_newMoves;
        currentCount = nextLevelCount;
    }
    // Final level: leaf kernel
    uint64 *d_leafCounts = nullptr;
#if USE_TT
    d_leafCounts = alloc.alloc<uint64>(currentCount);
#endif
    // Determine leaf TT (for fused kernel, remaining depth = bfsMinLevel - 1)
    TTTable leafTT = {nullptr, 0};
#if USE_TT && HASH_IN_LEAF_KERNEL
    int leafRemDepth = bfsMinLevel - 1;
    if (leafRemDepth >= 2 && leafRemDepth < MAX_TT_DEPTH)
        leafTT = deviceTTs[leafRemDepth];
#endif
    {
        int nBlocks = (currentCount - 1) / BLOCK_SIZE + 1;
        if (use2LevelLeaf)
        {
            fused_2level_leaf_kernel<<<nBlocks, BLOCK_SIZE>>>(d_prevBoards, d_prevStates,
                                                               d_prevHashes,
                                                               d_indices, d_moves,
                                                               d_perftCounter,
                                                               d_leafCounts,
                                                               leafTT,
                                                               currentCount, levelColor);
        }
        else
        {
            makeMove_and_perft_leaf_kernel<<<nBlocks, BLOCK_SIZE>>>(d_prevBoards, d_prevStates,
                                                                      d_prevHashes,
                                                                      d_indices, d_moves,
                                                                      d_perftCounter,
                                                                      d_leafCounts,
                                                                      currentCount, levelColor);
        }
        cudaDeviceSynchronize();
    }
    uint64 result = 0;
#if USE_TT
    // Upsweep: propagate per-position counts back through BFS levels and store in TT.
    // The upsweep result is the authoritative total (includes TT hit counts from
    // intermediate levels that the leaf kernel's global atomic counter misses).
    if (numLevels > 0 && d_leafCounts)
    {
        uint64 *currentChildCounts = d_leafCounts;
        int currentChildCount = currentCount;
        int *currentChildIndices = d_indices;  // leaf indices
        for (int lvl = numLevels - 1; lvl >= 0; lvl--)
        {
            int parentCount = levelSaves[lvl].count;
            // Allocate parent counts (zeroed)
            uint64 *d_parentCounts = alloc.alloc<uint64>(parentCount);
            if (!d_parentCounts)
            {
                // OOM during upsweep — fall back to leaf-only counter
                cudaMemcpy(&result, d_perftCounter, sizeof(uint64), cudaMemcpyDeviceToHost);
                return result;
            }
            cudaMemset(d_parentCounts, 0, parentCount * sizeof(uint64));
            // Reduce children into parents
            int nBlk = (currentChildCount - 1) / BLOCK_SIZE + 1;
            reduce_by_parent<<<nBlk, BLOCK_SIZE>>>(currentChildCounts, currentChildIndices,
                                                    d_parentCounts, currentChildCount);
            // Add TT hit counts and store in device TT
            TTTable upsweepTT = {nullptr, 0};
            int rd = levelSaves[lvl].remainingDepth;
            if (rd >= 2 && rd < MAX_TT_DEPTH)
                upsweepTT = deviceTTs[rd];
            nBlk = (parentCount - 1) / BLOCK_SIZE + 1;
            add_tt_hits_and_store<<<nBlk, BLOCK_SIZE>>>(d_parentCounts, levelSaves[lvl].ttHitCounts,
                                                         levelSaves[lvl].hashes, upsweepTT,
                                                         levelSaves[lvl].redirects, parentCount,
                                                         rd);
            // Apply dedup redirects: copy original's count to duplicates
            if (levelSaves[lvl].redirects)
            {
                apply_redirects<<<nBlk, BLOCK_SIZE>>>(d_parentCounts, levelSaves[lvl].redirects, parentCount);
            }
            // Move up
            currentChildCounts = d_parentCounts;
            currentChildCount = parentCount;
            currentChildIndices = levelSaves[lvl].indicesToParent;
        }
        // Sum the root children's counts = total perft result
        // First save the leaf atomic counter for debug comparison
        cudaMemset(d_perftCounter, 0, sizeof(uint64));
        int nBlk = (currentChildCount - 1) / BLOCK_SIZE + 1;
        reduce_sum_uint64<<<nBlk, BLOCK_SIZE>>>(currentChildCounts, d_perftCounter, currentChildCount);
        cudaDeviceSynchronize();
        cudaMemcpy(&result, d_perftCounter, sizeof(uint64), cudaMemcpyDeviceToHost);
    }
    else
#endif
    {
        // No upsweep: use leaf kernel's global atomic counter
        cudaMemcpy(&result, d_perftCounter, sizeof(uint64), cudaMemcpyDeviceToHost);
    }
    return result;
}
// -------------------------------------------------------------------------
// Move generator initialization (called via initMoveGen() in launcher.cu)
// Must stay here: cudaMemcpyToSymbol needs the __device__ symbols above.
// -------------------------------------------------------------------------
void MoveGeneratorBitboard::init()
{
    // initialize the empty board attack tables
    for (uint8 i=0; i < 64; i++)
    {
        uint64 x = BIT(i);
        uint64 north = northAttacks(x, ALLSET);
        uint64 south = southAttacks(x, ALLSET);
        uint64 east  = eastAttacks (x, ALLSET);
        uint64 west  = westAttacks (x, ALLSET);
        uint64 ne    = northEastAttacks(x, ALLSET);
        uint64 nw    = northWestAttacks(x, ALLSET);
        uint64 se    = southEastAttacks(x, ALLSET);
        uint64 sw    = southWestAttacks(x, ALLSET);
        RookAttacks  [i] = north | south | east | west;
        BishopAttacks[i] = ne | nw | se | sw;
        QueenAttacks [i] = RookAttacks[i] | BishopAttacks[i];
        KnightAttacks[i] = knightAttacks(x);
        KingAttacks[i]   = kingAttacks(x);
    }
    // initialize the Between and Line tables
    for (uint8 i=0; i<64; i++)
        for (uint8 j=0; j<64; j++)
        {
            if (i <= j)
            {
                Between[i][j] = squaresInBetween(i, j);
                Between[j][i] = Between[i][j];
            }
            Line[i][j] = squaresInLine(i, j);
        }
    // initialize magic lookup tables
    srand (time(NULL));
    for (int square = A1; square <= H8; square++)
    {
        uint64 thisSquare = BIT(square);
        uint64 mask    = sqRookAttacks(square) & (~thisSquare);
        // mask off squares that don't matter
        if ((thisSquare & RANK1) == 0)
            mask &= ~RANK1;
        if ((thisSquare & RANK8) == 0)
            mask &= ~RANK8;
        if ((thisSquare & FILEA) == 0)
            mask &= ~FILEA;
        if ((thisSquare & FILEH) == 0)
            mask &= ~FILEH;
        RookAttacksMasked[square] = mask;
        mask = sqBishopAttacks(square)  & (~thisSquare) & CENTRAL_SQUARES;
        BishopAttacksMasked[square] = mask;
    }
    // initialize fancy magic lookup table
    memset(fancy_magic_lookup_table, 0, sizeof(fancy_magic_lookup_table));
    for (int square = A1; square <= H8; square++)
    {
        uint64 rookMagic = findRookMagicForSquare(square, &fancy_magic_lookup_table[rook_magics_fancy[square].position], rook_magics_fancy[square].factor);
        assert(rookMagic == rook_magics_fancy[square].factor);
        uint64 bishopMagic = findBishopMagicForSquare(square, &fancy_magic_lookup_table[bishop_magics_fancy[square].position], bishop_magics_fancy[square].factor);
        assert(bishopMagic == bishop_magics_fancy[square].factor);
    }
    // copy all the lookup tables from CPU's memory to GPU memory
    cudaError_t err = cudaMemcpyToSymbol(gBetween, Between, sizeof(Between));
    if (err != S_OK) printf("For copying between table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(gLine, Line, sizeof(Line));
    if (err != S_OK) printf("For copying line table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(gRookAttacks, RookAttacks, sizeof(RookAttacks));
    if (err != S_OK) printf("For copying RookAttacks table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(gBishopAttacks, BishopAttacks, sizeof(BishopAttacks));
    if (err != S_OK) printf("For copying BishopAttacks table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(gQueenAttacks, QueenAttacks, sizeof(QueenAttacks));
    if (err != S_OK) printf("For copying QueenAttacks table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(gKnightAttacks, KnightAttacks, sizeof(KnightAttacks));
    if (err != S_OK) printf("For copying KnightAttacks table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(gKingAttacks, KingAttacks, sizeof(KingAttacks));
    if (err != S_OK) printf("For copying KingAttacks table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
    // Copy magic tables
    err = cudaMemcpyToSymbol(gRookAttacksMasked, RookAttacksMasked, sizeof(RookAttacksMasked));
    if (err != S_OK) printf("For copying RookAttacksMasked table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(gBishopAttacksMasked, BishopAttacksMasked , sizeof(BishopAttacksMasked));
    if (err != S_OK) printf("For copying BishopAttacksMasked  table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(g_fancy_magic_lookup_table, fancy_magic_lookup_table, sizeof(fancy_magic_lookup_table));
    if (err != S_OK) printf("For copying fancy_magic_lookup_table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(g_bishop_magics_fancy, bishop_magics_fancy, sizeof(bishop_magics_fancy));
    if (err != S_OK) printf("For copying bishop_magics_fancy, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(g_rook_magics_fancy, rook_magics_fancy, sizeof(rook_magics_fancy));
    if (err != S_OK) printf("For copying rook_magics_fancy, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
    // Build and copy combined magic entries (mask + magic in one struct)
    {
        CombinedMagicEntry bishopCombined[64], rookCombined[64];
        for (int sq = 0; sq < 64; sq++)
        {
            bishopCombined[sq].mask = BishopAttacksMasked[sq];
            bishopCombined[sq].factor = bishop_magics_fancy[sq].factor;
            bishopCombined[sq].position = bishop_magics_fancy[sq].position;
            bishopCombined[sq]._pad = 0;
            bishopCombined[sq]._pad2 = 0ull;
            rookCombined[sq].mask = RookAttacksMasked[sq];
            rookCombined[sq].factor = rook_magics_fancy[sq].factor;
            rookCombined[sq].position = rook_magics_fancy[sq].position;
            rookCombined[sq]._pad = 0;
            rookCombined[sq]._pad2 = 0ull;
        }
        err = cudaMemcpyToSymbol(g_bishop_combined, bishopCombined, sizeof(bishopCombined));
        if (err != S_OK) printf("For copying bishop_combined, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
        err = cudaMemcpyToSymbol(g_rook_combined, rookCombined, sizeof(rookCombined));
        if (err != S_OK) printf("For copying rook_combined, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
    }
    // Build and copy castle clear LUT
    // Bits [1:0] = whiteCastle bits to clear, bits [3:2] = blackCastle bits to clear
    {
        uint8 castleClear[64];
        memset(castleClear, 0, sizeof(castleClear));
        castleClear[H1] = CASTLE_FLAG_KING_SIDE;         // bit 0: white king-side
        castleClear[A1] = CASTLE_FLAG_QUEEN_SIDE;        // bit 1: white queen-side
        castleClear[E1] = CASTLE_FLAG_KING_SIDE | CASTLE_FLAG_QUEEN_SIDE;  // bits [1:0]: both white
        castleClear[H8] = CASTLE_FLAG_KING_SIDE  << 2;   // bit 2: black king-side
        castleClear[A8] = CASTLE_FLAG_QUEEN_SIDE << 2;   // bit 3: black queen-side
        castleClear[E8] = (CASTLE_FLAG_KING_SIDE | CASTLE_FLAG_QUEEN_SIDE) << 2;  // bits [3:2]: both black
        err = cudaMemcpyToSymbol(g_castleClear, castleClear, sizeof(castleClear));
        if (err != S_OK) printf("For copying castleClear, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
    }
    // Build and copy EP target LUTs
    {
        uint64 epBlack[9] = {0}, epWhite[9] = {0};
        for (int f = 1; f <= 8; f++)
        {
            epBlack[f] = BIT(f - 1) << (8 * 2);  // rank 3
            epWhite[f] = BIT(f - 1) << (8 * 5);  // rank 6
        }
        err = cudaMemcpyToSymbol(g_epTargetBlack, epBlack, sizeof(epBlack));
        if (err != S_OK) printf("For copying epTargetBlack, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
        err = cudaMemcpyToSymbol(g_epTargetWhite, epWhite, sizeof(epWhite));
        if (err != S_OK) printf("For copying epTargetWhite, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
    }
    // Initialize Zobrist random tables (CPU) and copy to GPU
    initZobrist();
    err = cudaMemcpyToSymbol(g_zobrist1, &zobrist1, sizeof(ZobristRandoms));
    if (err != S_OK) printf("For copying zobrist1, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(g_zobrist2, &zobrist2, sizeof(ZobristRandoms));
    if (err != S_OK) printf("For copying zobrist2, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
}
