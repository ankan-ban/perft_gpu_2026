// launcher utils — GPU init, CPU perft, TT management

#include <cuda_runtime.h>
#include <math.h>
#include "MoveGeneratorBitboard.h"
#include "launcher.h"
#include "utils.h"
#include "zobrist.h"
#include "tt.h"

// GPU memory buffer for BFS tree storage (allocated in initGPU, freed in main)
void *preAllocatedBufferHost;

// Global TT arrays
TTTable deviceTTs[MAX_TT_DEPTH];
TTTable hostTTs[MAX_TT_DEPTH];

void initGPU(int gpu)
{
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(gpu);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! error: %s\n", cudaGetErrorString(cudaStatus));
        exit(1);
    }

    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    printf("\ngpu: %d, memory total: %llu, free: %llu\n", gpu, (unsigned long long)total, (unsigned long long)free);

    // allocate the preallocated buffer for BFS tree storage
    cudaStatus = cudaMalloc(&preAllocatedBufferHost, PREALLOCATED_MEMORY_SIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "error in malloc for preAllocatedBuffer, error desc: %s\n", cudaGetErrorString(cudaStatus));
        exit(1);
    }
    else
    {
        printf("Allocated preAllocatedBuffer of %llu bytes\n", (unsigned long long)PREALLOCATED_MEMORY_SIZE);
    }
}

// -------------------------------------------------------------------------
// Transposition table allocation
// -------------------------------------------------------------------------

// Round down to largest power of 2 <= n
static uint64 floorPow2(uint64 n)
{
    if (n == 0) return 0;
    n |= (n >> 1);
    n |= (n >> 2);
    n |= (n >> 4);
    n |= (n >> 8);
    n |= (n >> 16);
    n |= (n >> 32);
    return (n + 1) >> 1;
}

// Overridable TT budgets (set from CLI before calling initTT)
int g_deviceTTBudgetMB = DEVICE_TT_BUDGET_MB;
int g_hostTTBudgetMB = HOST_TT_BUDGET_MB;

void initTT(int launchDepth, int maxDepth)
{
#if USE_TT
    memset(deviceTTs, 0, sizeof(deviceTTs));
    memset(hostTTs, 0, sizeof(hostTTs));

    // Device TTs: depths 3 through launchDepth-1 (GPU BFS levels)
    // Depth 2 is unused: bfsMinLevel=3 means no BFS level probes TT[2],
    // and HASH_IN_LEAF_KERNEL=0 means the fused leaf doesn't probe it either.
    int numDeviceTTs = 0;
    for (int d = 3; d < launchDepth && d < MAX_TT_DEPTH; d++)
        numDeviceTTs++;

    if (numDeviceTTs > 0)
    {
        uint64 budgetBytes;
        if (g_deviceTTBudgetMB > 0)
        {
            budgetBytes = (uint64)g_deviceTTBudgetMB * 1024 * 1024;
        }
        else
        {
            // Auto-size: use 75% of free VRAM for device TTs
            size_t freeMem = 0, totalMem = 0;
            cudaMemGetInfo(&freeMem, &totalMem);
            budgetBytes = (uint64)(freeMem * 3 / 4);
            printf("Auto device TT budget: %llu MB (75%% of %llu MB free)\n",
                   (unsigned long long)(budgetBytes / (1024*1024)),
                   (unsigned long long)(freeMem / (1024*1024)));
        }
        uint64 perTableBytes = budgetBytes / numDeviceTTs;
        uint64 entriesPerTable = floorPow2(perTableBytes / sizeof(TTEntry));
        if (entriesPerTable < 1024) entriesPerTable = 1024;

        for (int d = 3; d < launchDepth && d < MAX_TT_DEPTH; d++)
        {
            cudaError_t err = cudaMalloc(&deviceTTs[d].entries, entriesPerTable * sizeof(TTEntry));
            if (err != cudaSuccess)
            {
                printf("Warning: failed to allocate device TT[%d] (%llu MB): %s\n",
                       d, (unsigned long long)(entriesPerTable * sizeof(TTEntry) / (1024*1024)),
                       cudaGetErrorString(err));
                deviceTTs[d].entries = nullptr;
                deviceTTs[d].mask = 0;
                continue;
            }
            cudaMemset(deviceTTs[d].entries, 0, entriesPerTable * sizeof(TTEntry));
            deviceTTs[d].mask = entriesPerTable - 1;
            printf("Device TT[%d]: %lluM entries (%llu MB)\n", d,
                   (unsigned long long)(entriesPerTable / (1024*1024)),
                   (unsigned long long)(entriesPerTable * sizeof(TTEntry) / (1024*1024)));
        }
    }

    // Host TTs: depths launchDepth through maxDepth (CPU levels)
    int numHostTTs = 0;
    for (int d = launchDepth; d <= maxDepth && d < MAX_TT_DEPTH; d++)
        numHostTTs++;

    if (numHostTTs > 0)
    {
        uint64 budgetBytes = (uint64)g_hostTTBudgetMB * 1024 * 1024;
        uint64 perTableBytes = budgetBytes / numHostTTs;
        uint64 entriesPerTable = floorPow2(perTableBytes / sizeof(TTEntry));
        if (entriesPerTable < 1024) entriesPerTable = 1024;

        for (int d = launchDepth; d <= maxDepth && d < MAX_TT_DEPTH; d++)
        {
            hostTTs[d].entries = (TTEntry *)malloc(entriesPerTable * sizeof(TTEntry));
            if (!hostTTs[d].entries)
            {
                printf("Warning: failed to allocate host TT[%d]\n", d);
                hostTTs[d].mask = 0;
                continue;
            }
            memset(hostTTs[d].entries, 0, entriesPerTable * sizeof(TTEntry));
            hostTTs[d].mask = entriesPerTable - 1;
            printf("Host TT[%d]: %lluM entries (%llu MB)\n", d,
                   (unsigned long long)(entriesPerTable / (1024*1024)),
                   (unsigned long long)(entriesPerTable * sizeof(TTEntry) / (1024*1024)));
        }
    }

    printf("\n");
    fflush(stdout);
#endif
}

void clearDeviceTTs()
{
#if USE_TT
    for (int d = 0; d < MAX_TT_DEPTH; d++)
    {
        if (deviceTTs[d].entries)
            cudaMemset(deviceTTs[d].entries, 0, (deviceTTs[d].mask + 1) * sizeof(TTEntry));
    }
#endif
}

void freeTT()
{
#if USE_TT
    for (int d = 0; d < MAX_TT_DEPTH; d++)
    {
        if (deviceTTs[d].entries)
        {
            cudaFree(deviceTTs[d].entries);
            deviceTTs[d].entries = nullptr;
        }
        if (hostTTs[d].entries)
        {
            free(hostTTs[d].entries);
            hostTTs[d].entries = nullptr;
        }
    }
#endif
}

uint32 estimateLaunchDepth(QuadBitBoard *pos, GameState *gs, uint8 rootColor)
{
    // estimate branching factor near the root
    double perft1 = (double)perft_cpu_dispatch(pos, gs, rootColor, 1);
    double perft2 = (double)perft_cpu_dispatch(pos, gs, rootColor, 2);
    double perft3 = (double)perft_cpu_dispatch(pos, gs, rootColor, 3);

    // this works well when the root position has very low branching factor (e.g, in case king is in check)
    float geoMean = sqrt((perft3 / perft2) * (perft2 / perft1));
    float arithMean = ((perft3 / perft2) + (perft2 / perft1)) / 2;

    float branchingFactor = (geoMean + arithMean) / 2;
    if (arithMean / geoMean > 2.0f)
    {
        printf("\nUnstable position, defaulting to launch depth = 5\n");
        return 5;
    }

    // With the fused 2-level leaf kernel, the last BFS level's huge move/index
    // arrays are eliminated. This effectively multiplies the memory budget by
    // the branching factor, allowing a higher launch depth (fewer CPU calls).
    float memLimit = (float)PREALLOCATED_MEMORY_SIZE / 2.0f * branchingFactor;

    // estimated depth is log of memLimit in base 'branchingFactor'
    uint32 depth = (uint32)(log(memLimit) / log(branchingFactor));

    return depth;
}


// Serial CPU recursion at top levels, launching GPU BFS at launchDepth
static uint64 perft_cpu_recurse(QuadBitBoard *pos, GameState *gs, uint8 color, int depth, int launchDepth, Hash128 hash, void *gpuBuffer, size_t bufferSize)
{
#if USE_TT
    // Probe host TT
    if (depth >= 2)
    {
        uint64 ttCount;
        if (ttProbeHost(hostTTs[depth], hash, &ttCount))
            return ttCount;
    }
#endif

    uint64 count;

    if (depth <= launchDepth)
    {
        count = perft_gpu_host_bfs(pos, gs, color, depth, gpuBuffer, bufferSize);
    }
    else
    {
        // Serial CPU recursion
        CMove moves[MAX_MOVES];
        QuadBitBoard childPos;
        GameState childGs;
        int nMoves = generateMoves(pos, gs, color, moves);

        count = 0;
        for (int i = 0; i < nMoves; i++)
        {
            childPos = *pos;
            childGs = *gs;

            // Pre-move state for hash update
            uint8 srcPiece = getPieceAt(pos, moves[i].getFrom());
            uint8 capPiece = getPieceAt(pos, moves[i].getTo());
            uint8 oldCastleRaw = gs->raw;
            uint8 oldEP = gs->enPassent;

            if (color == WHITE)
                MoveGeneratorBitboard::makeMove<WHITE>(&childPos, &childGs, moves[i]);
            else
                MoveGeneratorBitboard::makeMove<BLACK>(&childPos, &childGs, moves[i]);

            Hash128 childHash = updateHashAfterMove(hash, moves[i], color,
                srcPiece, capPiece, oldCastleRaw, childGs.raw, oldEP, childGs.enPassent);

            count += perft_cpu_recurse(&childPos, &childGs, !color, depth - 1, launchDepth, childHash, gpuBuffer, bufferSize);
        }
    }

#if USE_TT
    // Store in host TT
    if (depth >= 2)
        ttStoreHost(hostTTs[depth], hash, count);
#endif

    return count;
}


void perftLauncher(QuadBitBoard *pos, GameState *gs, uint8 rootColor, uint32 depth, int launchDepth)
{
    Hash128 rootHash = computeHash(pos, gs, rootColor);

    Timer timer;
    timer.start();

    uint64 result = perft_cpu_recurse(pos, gs, rootColor, depth, launchDepth, rootHash, preAllocatedBufferHost, PREALLOCATED_MEMORY_SIZE);

    timer.stop();
    double seconds = timer.elapsed();

    printf("\nPerft(%02d): %llu, time: %g seconds", depth, (unsigned long long)result, seconds);
    if (seconds > 0)
        printf(", nps: %llu", (unsigned long long)((double)result / seconds));
    printf("\n");
    fflush(stdout);
}

void perftCPU(QuadBitBoard *pos, GameState *gs, uint8 rootColor, uint32 depth)
{
    Timer timer;
    timer.start();

    // Note: perft_cpu_dispatch computes root hash internally
    uint64 result = perft_cpu_dispatch(pos, gs, rootColor, depth);

    timer.stop();
    double seconds = timer.elapsed();

    printf("\nPerft(%02d): %llu, time: %g seconds", depth, (unsigned long long)result, seconds);
    if (seconds > 0)
        printf(", nps: %llu", (unsigned long long)((double)result / seconds));
    printf("\n");
    fflush(stdout);
}


// -------------------------------------------------------------------------
// Template-optimized CPU perft
// -------------------------------------------------------------------------

template <uint8 chance>
static uint64 perft_cpu(QuadBitBoard *pos, GameState *gs, uint32 depth, Hash128 hash)
{
    if (depth == 1)
        return MoveGeneratorBitboard::countMoves<chance>(pos, gs);

#if USE_TT
    uint64 ttCount;
    if (ttProbeHost(hostTTs[depth], hash, &ttCount))
        return ttCount;
#endif

    CMove moves[MAX_MOVES];
    int nMoves = MoveGeneratorBitboard::generateMoves<chance>(pos, gs, moves);

    uint64 count = 0;
    for (int i = 0; i < nMoves; i++)
    {
        QuadBitBoard childPos = *pos;
        GameState childGs = *gs;

        uint8 srcPiece = getPieceAt(pos, moves[i].getFrom());
        uint8 capPiece = getPieceAt(pos, moves[i].getTo());
        uint8 oldCastleRaw = gs->raw;
        uint8 oldEP = gs->enPassent;

        MoveGeneratorBitboard::makeMove<chance>(&childPos, &childGs, moves[i]);

        Hash128 childHash = updateHashAfterMove(hash, moves[i], chance,
            srcPiece, capPiece, oldCastleRaw, childGs.raw, oldEP, childGs.enPassent);

        count += perft_cpu<!chance>(&childPos, &childGs, depth - 1, childHash);
    }

#if USE_TT
    ttStoreHost(hostTTs[depth], hash, count);
#endif

    return count;
}

uint64 perft_cpu_dispatch(QuadBitBoard *pos, GameState *gs, uint8 color, uint32 depth)
{
    if (depth == 0)
        return 1;
    Hash128 hash = computeHash(pos, gs, color);
    if (color == WHITE)
        return perft_cpu<WHITE>(pos, gs, depth, hash);
    else
        return perft_cpu<BLACK>(pos, gs, depth, hash);
}


// -------------------------------------------------------------------------
// Move generator initialization
// -------------------------------------------------------------------------

void initMoveGen()
{
    MoveGeneratorBitboard::init();
}
