// All GPU kernels, device helpers, host-driven BFS, and move generator init

#include <cub/cub.cuh>

// the routines that actually generate the moves
#include "MoveGeneratorBitboard.h"

// preallocated GPU memory buffer (host-side pointer)
void *preAllocatedBufferHost;

// helper routines for CPU perft
static uint32 countMoves(QuadBitBoard *pos, GameState *gs)
{
    uint32 nMoves;
    int chance = gs->chance;

#if USE_TEMPLATE_CHANCE_OPT == 1
    if (chance == BLACK)
    {
        nMoves = MoveGeneratorBitboard::countMoves<BLACK>(pos, gs);
    }
    else
    {
        nMoves = MoveGeneratorBitboard::countMoves<WHITE>(pos, gs);
    }
#else
    nMoves = MoveGeneratorBitboard::countMoves(pos, gs, chance);
#endif
    return nMoves;
}

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
#if USE_TEMPLATE_CHANCE_OPT == 1
    if (chance == BLACK)
    {
        MoveGeneratorBitboard::makeMove<BLACK>(pos, gs, move);
    }
    else
    {
        MoveGeneratorBitboard::makeMove<WHITE>(pos, gs, move);
    }
#else
    MoveGeneratorBitboard::makeMove(pos, gs, move, chance);
#endif
}

__host__ __device__ __forceinline__ uint32 countMoves(QuadBitBoard *pos, GameState *gs, uint8 color)
{
#if USE_TEMPLATE_CHANCE_OPT == 1
    if (color == BLACK)
    {
        return MoveGeneratorBitboard::countMoves<BLACK>(pos, gs);
    }
    else
    {
        return MoveGeneratorBitboard::countMoves<WHITE>(pos, gs);
    }
#else
    return MoveGeneratorBitboard::countMoves(pos, gs, color);
#endif
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

// Kernel: make move on parent board, produce child board, count child moves
#if LIMIT_REGISTER_USE == 1
__launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_MP)
#endif
__global__ void makemove_and_count_moves_kernel(QuadBitBoard *parentBoards, GameState *parentStates,
                                                 int *indices, CMove *moves,
                                                 QuadBitBoard *outPositions, GameState *outStates,
                                                 int *moveCounts, int nThreads)
{
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    int nMoves = 0;

    if (index < nThreads)
    {
        int parentIndex = indices[index];
        QuadBitBoard pos = loadQuadBB(&parentBoards[parentIndex]);
        GameState gs = parentStates[parentIndex];
        CMove move = moves[index];

        uint8 color = gs.chance;
        makeMove(&pos, &gs, move, color);
        nMoves = countMoves(&pos, &gs, !color);

        storeQuadBB(&outPositions[index], pos);
        outStates[index] = gs;
        moveCounts[index] = nMoves;
    }
}

// Kernel: generate moves for each position (uses inclusive prefix sums for offsets)
// Uses shared memory staging for coalesced global writes.
// Without this, each thread writes 2-byte CMoves to scattered global addresses
// (6.25% sector utilization, 85% wasted traffic). With staging, the CTA
// cooperatively copies the compacted move buffer to global memory at 100% utilization.
#define SMEM_MOVES_CAPACITY (BLOCK_SIZE * 64)

__global__ void generate_moves_kernel(QuadBitBoard *positions, GameState *states,
                                       CMove *generatedMovesBase, int *inclPrefixSums, int nThreads)
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
            generateMoves(&pos, &gs, gs.chance, &smemMoves[localOffset]);
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
            generateMoves(&pos, &gs, gs.chance, &generatedMovesBase[blockGlobalStart + localOffset]);
        }
    }
}

// Kernel: leaf level - make move and count, add to global perft counter
#if LIMIT_REGISTER_USE == 1
__launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_MP)
#endif
__global__ void makeMove_and_perft_leaf_kernel(QuadBitBoard *positions, GameState *states,
                                                int *indices, CMove *moves,
                                                uint64 *globalPerftCounter, int nThreads)
{
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= nThreads)
        return;

    uint32 boardIndex = indices[index];
    QuadBitBoard pos = loadQuadBB(&positions[boardIndex]);
    GameState gs = states[boardIndex];
    int color = gs.chance;

    CMove move = moves[index];

    makeMove(&pos, &gs, move, color);

    // count moves at this position
    int nMoves = countMoves(&pos, &gs, !color);

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
                                                uint64 *globalPerftCounter, int nThreads)
{
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= nThreads)
        return;

    QuadBitBoard pos = loadQuadBB(&boards[index]);
    GameState gs = states[index];
    uint8 color = gs.chance;

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
                                          int *indices, CMove *moves,
                                          uint64 *globalPerftCounter, int nThreads)
{
    uint32 index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= nThreads)
        return;

    // Load parent position and make the first move (level N -> level N-1)
    uint32 boardIndex = indices[index];
    QuadBitBoard pos = loadQuadBB(&positions[boardIndex]);
    GameState gs = states[boardIndex];
    uint8 color = gs.chance;

    CMove move = moves[index];
    makeMove(&pos, &gs, move, color);

    // Generate all legal moves at level N-1 into local memory.
    // 218 moves covers the wrost case
    CMove childMoves[218];
    uint8 childColor = gs.chance;  // after makeMove, chance is flipped
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

    // Warp-wide reduction before atomic add
    warpReduce(totalCount);

    int laneId = threadIdx.x & 0x1f;

    if (laneId == 0)
    {
        atomicAdd(globalPerftCounter, (uint64)totalCount);
    }
}


// -------------------------------------------------------------------------
// Host-driven BFS perft
// -------------------------------------------------------------------------

// Run perft on GPU for a single position using host-driven breadth-first search
uint64 perft_gpu_host_bfs(QuadBitBoard *pos, GameState *gs, int depth, void *gpuBuffer, size_t bufferSize)
{
    if (depth <= 0)
        return 1;
    if (depth == 1)
        return countMoves(pos, gs);

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

    // Generate root moves on CPU
    CMove rootMoves[MAX_MOVES];
    int rootMoveCount = generateMoves(pos, gs, gs->chance, rootMoves);

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
    int currentCount = rootMoveCount;

    // Use fused 2-level leaf for depth >= 4: saves memory and improves locality.
    // The 2-level leaf handles the last 2 levels, so BFS stops one level earlier.
    bool use2LevelLeaf = (depth >= 4);
    int bfsMinLevel = use2LevelLeaf ? 3 : 2;

    // BFS loop: depth-1 down to bfsMinLevel are intermediate levels
    for (int level = depth - 1; level >= bfsMinLevel; level--)
    {
        // Allocate boards, states, and move counts
        QuadBitBoard *d_curBoards = alloc.alloc<QuadBitBoard>(currentCount);
        GameState *d_curStates = alloc.alloc<GameState>(currentCount);
        int *d_moveCounts = alloc.alloc<int>(currentCount);

        if (!d_curBoards || !d_curStates || !d_moveCounts)
        {
            printf("\nOOM during BFS at level %d with %d positions\n", level, currentCount);
            return 0;
        }

        // Step 1: make moves and count child moves
        int nBlocks = (currentCount - 1) / BLOCK_SIZE + 1;
        makemove_and_count_moves_kernel<<<nBlocks, BLOCK_SIZE>>>(d_prevBoards, d_prevStates,
                                                                   d_indices, d_moves,
                                                                   d_curBoards, d_curStates,
                                                                   d_moveCounts, currentCount);

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
            // This avoids allocating the huge move/index arrays for this level.
            if (level >= 2)
            {
                int nBlk = (currentCount - 1) / BLOCK_SIZE + 1;
                perft_2level_from_board_kernel<<<nBlk, BLOCK_SIZE>>>(d_curBoards, d_curStates,
                                                                      d_perftCounter, currentCount);
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

        // Step 5: generate moves
        generate_moves_kernel<<<nBlocks, BLOCK_SIZE>>>(d_curBoards, d_curStates, d_newMoves, d_moveCounts, currentCount);

        // Advance to next level
        d_prevBoards = d_curBoards;
        d_prevStates = d_curStates;
        d_indices = d_newIndices;
        d_moves = d_newMoves;
        currentCount = nextLevelCount;
    }

    // Final level: leaf kernel
    {
        int nBlocks = (currentCount - 1) / BLOCK_SIZE + 1;
        if (use2LevelLeaf)
        {
            // Fused 2-level leaf: each thread handles 2 levels (make move + generate + count all)
            fused_2level_leaf_kernel<<<nBlocks, BLOCK_SIZE>>>(d_prevBoards, d_prevStates,
                                                               d_indices, d_moves,
                                                               d_perftCounter, currentCount);
        }
        else
        {
            // Standard 1-level leaf: each thread handles 1 level (make move + count)
            makeMove_and_perft_leaf_kernel<<<nBlocks, BLOCK_SIZE>>>(d_prevBoards, d_prevStates,
                                                                      d_indices, d_moves,
                                                                      d_perftCounter, currentCount);
        }
        cudaDeviceSynchronize();
    }

    // Read back result
    uint64 result = 0;
    cudaMemcpy(&result, d_perftCounter, sizeof(uint64), cudaMemcpyDeviceToHost);

    return result;
}


// -------------------------------------------------------------------------
// CPU perft (for depth estimation)
// -------------------------------------------------------------------------

// A very simple CPU routine - only for estimating launch depth
uint64 perft_bb(QuadBitBoard *pos, GameState *gs, uint32 depth)
{
    if (depth == 1)
    {
        return countMoves(pos, gs);
    }

    CMove moves[MAX_MOVES];
    int nMoves = generateMoves(pos, gs, gs->chance, moves);

    uint64 count = 0;
    for (int i = 0; i < nMoves; i++)
    {
        QuadBitBoard childPos = *pos;
        GameState childGs = *gs;
#if USE_TEMPLATE_CHANCE_OPT == 1
        if (gs->chance == WHITE)
            MoveGeneratorBitboard::makeMove<WHITE>(&childPos, &childGs, moves[i]);
        else
            MoveGeneratorBitboard::makeMove<BLACK>(&childPos, &childGs, moves[i]);
#else
        MoveGeneratorBitboard::makeMove(&childPos, &childGs, moves[i], gs->chance);
#endif
        count += perft_bb(&childPos, &childGs, depth - 1);
    }
    return count;
}


// -------------------------------------------------------------------------
// Move generator initialization
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
#if USE_SLIDING_LUT == 1
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
#if USE_FANCY_MAGICS != 1
        rookMagics  [square] = findRookMagicForSquare  (square, rookMagicAttackTables  [square]);
        bishopMagics[square] = findBishopMagicForSquare(square, bishopMagicAttackTables[square]);
#endif
    }

    // initialize fancy magic lookup table
    memset(fancy_magic_lookup_table, 0, sizeof(fancy_magic_lookup_table));
    int globalOffsetRook = 0;
    int globalOffsetBishop = 0;

    for (int square = A1; square <= H8; square++)
    {
        int uniqueBishopAttacks = 0, uniqueRookAttacks=0;

        uint64 rookMagic = findRookMagicForSquare  (square, &fancy_magic_lookup_table[rook_magics_fancy[square].position], rook_magics_fancy[square].factor,
                                                    &fancy_byte_RookLookup[globalOffsetRook], &fancy_byte_magic_lookup_table[rook_magics_fancy[square].position], &uniqueRookAttacks);
        assert(rookMagic == rook_magics_fancy[square].factor);

        uint64 bishopMagic = findBishopMagicForSquare  (square, &fancy_magic_lookup_table[bishop_magics_fancy[square].position], bishop_magics_fancy[square].factor,
                                                       &fancy_byte_BishopLookup[globalOffsetBishop], &fancy_byte_magic_lookup_table[bishop_magics_fancy[square].position], &uniqueBishopAttacks);
        assert(bishopMagic == bishop_magics_fancy[square].factor);

        rook_magics_fancy  [square].offset = globalOffsetRook;
        globalOffsetRook += uniqueRookAttacks;

        bishop_magics_fancy[square].offset = globalOffsetBishop;
        globalOffsetBishop += uniqueBishopAttacks;
    }
#endif

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

    // Copy magical tables
    err = cudaMemcpyToSymbol(gRookAttacksMasked, RookAttacksMasked, sizeof(RookAttacksMasked));
    if (err != S_OK) printf("For copying RookAttacksMasked table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(gBishopAttacksMasked, BishopAttacksMasked , sizeof(BishopAttacksMasked));
    if (err != S_OK) printf("For copying BishopAttacksMasked  table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(gRookMagics, rookMagics, sizeof(rookMagics));
    if (err != S_OK) printf("For copying rookMagics  table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(gBishopMagics, bishopMagics, sizeof(bishopMagics));
    if (err != S_OK) printf("For copying bishopMagics table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(gRookMagicAttackTables, rookMagicAttackTables, sizeof(rookMagicAttackTables));
    if (err != S_OK) printf("For copying RookMagicAttackTables, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(gBishopMagicAttackTables, bishopMagicAttackTables, sizeof(bishopMagicAttackTables));
    if (err != S_OK) printf("For copying bishopMagicAttackTables, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

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

    err = cudaMemcpyToSymbol(g_fancy_byte_magic_lookup_table, fancy_byte_magic_lookup_table, sizeof(fancy_byte_magic_lookup_table));
    if (err != S_OK) printf("For copying fancy_byte_magic_lookup_table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(g_fancy_byte_BishopLookup, fancy_byte_BishopLookup, sizeof(fancy_byte_BishopLookup));
    if (err != S_OK) printf("For copying fancy_byte_BishopLookup, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(g_fancy_byte_RookLookup, fancy_byte_RookLookup, sizeof(fancy_byte_RookLookup));
    if (err != S_OK) printf("For copying fancy_byte_RookLookup, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

#if USE_CONSTANT_MEMORY_FOR_LUT == 1
    printf("Copying tables to constant memory...\n");
    err = cudaMemcpyToSymbol(cLine, Line, sizeof(Line));
    if (err != S_OK) printf("For copying line table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(cRookAttacks, RookAttacks, sizeof(RookAttacks));
    if (err != S_OK) printf("For copying RookAttacks table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(cBishopAttacks, BishopAttacks, sizeof(BishopAttacks));
    if (err != S_OK) printf("For copying BishopAttacks table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(cKnightAttacks, KnightAttacks, sizeof(KnightAttacks));
    if (err != S_OK) printf("For copying KnightAttacks table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(cKingAttacks, KingAttacks, sizeof(KingAttacks));
    if (err != S_OK) printf("For copying KingAttacks table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(cRookAttacksMasked, RookAttacksMasked, sizeof(RookAttacksMasked));
    if (err != S_OK) printf("For copying RookAttacksMasked table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(cBishopAttacksMasked, BishopAttacksMasked , sizeof(BishopAttacksMasked));
    if (err != S_OK) printf("For copying BishopAttacksMasked  table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(cRookMagics, rookMagics, sizeof(rookMagics));
    if (err != S_OK) printf("For copying rookMagics  table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(cBishopMagics, bishopMagics, sizeof(bishopMagics));
    if (err != S_OK) printf("For copying bishopMagics table, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(c_bishop_magics_fancy, bishop_magics_fancy, sizeof(bishop_magics_fancy));
    if (err != S_OK) printf("For copying bishop_magics_fancy, Err id: %d, str: %s\n", err, cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(c_rook_magics_fancy, rook_magics_fancy, sizeof(rook_magics_fancy));
    if (err != S_OK) printf("For copying rook_magics_fancy, Err id: %d, str: %s\n", err, cudaGetErrorString(err));
#endif
}

void initMoveGen()
{
    MoveGeneratorBitboard::init();
}
