// launcher utils — GPU init, CPU perft, TT management

#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <thread>
#include "MoveGeneratorBitboard.h"
#include "launcher.h"
#include "utils.h"
#include "zobrist.h"
#include "tt.h"

// GPU memory buffer for BFS tree storage (backward compat, points to g_workers[0].gpuBuffer)
void *preAllocatedBufferHost;

// Global TT arrays
TTTable deviceTTs[MAX_TT_DEPTH];
LosslessTT hostLosslessTTs[MAX_TT_DEPTH];

// Per-worker state (streams, buffers, diagnostics)
WorkerState g_workers[NUM_WORKERS];

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

    // Allocate per-worker GPU resources (full-size buffer each)
    size_t perWorkerBuffer = PREALLOCATED_MEMORY_SIZE;

    for (int w = 0; w < NUM_WORKERS; w++)
    {
        memset(&g_workers[w], 0, sizeof(WorkerState));

        cudaStatus = cudaMalloc(&g_workers[w].gpuBuffer, perWorkerBuffer);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "error allocating buffer for worker %d: %s\n", w, cudaGetErrorString(cudaStatus));
            exit(1);
        }
        g_workers[w].bufferSize = perWorkerBuffer;

        cudaStreamCreate(&g_workers[w].stream);
        cudaMallocHost(&g_workers[w].pinnedInt, sizeof(int));
        cudaMallocHost(&g_workers[w].pinnedU64, sizeof(uint64));

        g_workers[w].dedupGeneration = 1;

        printf("Worker %d: %llu MB buffer, stream %p\n",
               w, (unsigned long long)(perWorkerBuffer / (1024*1024)), (void*)g_workers[w].stream);
    }

    // Backward compat
    preAllocatedBufferHost = g_workers[0].gpuBuffer;
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

void initTT(int launchDepth, int maxLaunchDepth, int maxDepth, float branchingFactor)
{
#if USE_TT
    memset(deviceTTs, 0, sizeof(deviceTTs));
    memset(hostLosslessTTs, 0, sizeof(hostLosslessTTs));

    // Device TTs: depths 3 through maxLaunchDepth-1 (GPU BFS levels)
    int numDeviceTTs = 0;
    for (int d = 3; d < maxLaunchDepth && d < MAX_TT_DEPTH; d++)
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

        for (int d = 3; d < maxLaunchDepth && d < MAX_TT_DEPTH; d++)
        {
            uint64 entries = (d == 4) ? entriesPerTable * 2 : entriesPerTable;

            cudaError_t err = cudaMalloc(&deviceTTs[d].entries, entries * sizeof(TTEntry));
            if (err != cudaSuccess)
            {
                printf("Warning: failed to allocate device TT[%d] (%llu MB): %s\n",
                       d, (unsigned long long)(entries * sizeof(TTEntry) / (1024*1024)),
                       cudaGetErrorString(err));
                deviceTTs[d].entries = nullptr;
                deviceTTs[d].mask = 0;
                continue;
            }
            cudaMemset(deviceTTs[d].entries, 0, entries * sizeof(TTEntry));
            deviceTTs[d].mask = entries - 1;
            printf("Device TT[%d]: %lluM entries (%llu MB)\n", d,
                   (unsigned long long)(entries / (1024*1024)),
                   (unsigned long long)(entries * sizeof(TTEntry) / (1024*1024)));
        }
    }

    // Host TTs: all depths launchDepth through maxDepth use lossless chained tables.
    // Budget split proportionally to branchingFactor^(maxDepth - d).
    int numHostTTs = 0;
    for (int d = launchDepth; d <= maxDepth && d < MAX_TT_DEPTH; d++)
        numHostTTs++;

    if (numHostTTs > 0)
    {
        uint64 budgetBytes = (uint64)g_hostTTBudgetMB * 1024 * 1024;

        double weights[MAX_TT_DEPTH];
        memset(weights, 0, sizeof(weights));
        double totalWeight = 0;
        for (int d = launchDepth; d <= maxDepth && d < MAX_TT_DEPTH; d++)
        {
            weights[d] = pow((double)branchingFactor, maxDepth - d);
            totalWeight += weights[d];
        }

        uint64 bytesPerSlot = sizeof(LosslessEntry) + sizeof(int32_t);

        for (int d = launchDepth; d <= maxDepth && d < MAX_TT_DEPTH; d++)
        {
            uint64 depthBytes = (uint64)(budgetBytes * weights[d] / totalWeight);
            uint64 numSlots = depthBytes / bytesPerSlot;
            numSlots = floorPow2(numSlots);
            if (numSlots < 4 * 1024 * 1024) numSlots = 4 * 1024 * 1024;
            if (numSlots > (1ull << 30)) numSlots = (1ull << 30);

            int32_t poolCap = (int32_t)numSlots;
            uint64 numBuckets = numSlots;

            hostLosslessTTs[d].buckets = (int32_t *)malloc(numBuckets * sizeof(int32_t));
            hostLosslessTTs[d].pool = (LosslessEntry *)malloc((uint64)poolCap * sizeof(LosslessEntry));
            if (hostLosslessTTs[d].buckets && hostLosslessTTs[d].pool)
            {
                memset(hostLosslessTTs[d].buckets, 0xFF, numBuckets * sizeof(int32_t));
                hostLosslessTTs[d].bucketMask = numBuckets - 1;
                hostLosslessTTs[d].nextFree = 0;
                hostLosslessTTs[d].poolCapacity = poolCap;

                uint64 totalMB = (numBuckets * sizeof(int32_t) + (uint64)poolCap * sizeof(LosslessEntry)) / (1024*1024);
                const char *unit = ""; uint64 dispSlots = numSlots;
                if (numSlots >= 1024*1024) { unit = "M"; dispSlots = numSlots / (1024*1024); }
                else if (numSlots >= 1024) { unit = "K"; dispSlots = numSlots / 1024; }
                printf("Host TT[%d] (lossless): %llu%s entries (%llu MB)\n",
                       d, (unsigned long long)dispSlots, unit, (unsigned long long)totalMB);
            }
            else
            {
                free(hostLosslessTTs[d].buckets);
                free(hostLosslessTTs[d].pool);
                memset(&hostLosslessTTs[d], 0, sizeof(LosslessTT));
                printf("Warning: failed to allocate host TT[%d]\n", d);
            }
        }
    }

    printf("\n");
    fflush(stdout);
#endif
}

void resetCallStats()
{
    for (int w = 0; w < NUM_WORKERS; w++)
    {
        g_workers[w].oomFallbackCount = 0;
#if VERBOSE_LOGGING
        g_workers[w].gpuCallCount = 0;
        g_workers[w].gpuCallTotalTime = 0.0;
        g_workers[w].gpuCallMinTime = 1e30;
        g_workers[w].gpuCallMaxTime = 0.0;
        g_workers[w].hostTTHits = 0;
        g_workers[w].hostTTMisses = 0;
        g_workers[w].iterPeakBfsMemory = 0;
        memset(g_workers[w].callSizeHist, 0, sizeof(g_workers[w].callSizeHist));
        memset(g_workers[w].callSizeLeafSum, 0, sizeof(g_workers[w].callSizeLeafSum));
        memset(g_workers[w].callSizeTimeSum, 0, sizeof(g_workers[w].callSizeTimeSum));
        memset(g_workers[w].callTimeHist, 0, sizeof(g_workers[w].callTimeHist));
        memset(g_workers[w].callTimeTimeSum, 0, sizeof(g_workers[w].callTimeTimeSum));
        memset(g_workers[w].bfsLevelPositionSum, 0, sizeof(g_workers[w].bfsLevelPositionSum));
        memset(g_workers[w].bfsLevelCallCount, 0, sizeof(g_workers[w].bfsLevelCallCount));
        g_workers[w].totalLeafPositions = 0;
        g_workers[w].iterStartWallTime = std::chrono::steady_clock::now();
        g_workers[w].lastProgressWallTime = 0;
#endif
    }
}

uint64 getOomFallbackCount()
{
    uint64 total = 0;
    for (int w = 0; w < NUM_WORKERS; w++)
        total += g_workers[w].oomFallbackCount;
    return total;
}

int getEffectiveLD()
{
    int minLD = g_workers[0].effectiveLD;
    for (int w = 1; w < NUM_WORKERS; w++)
        if (g_workers[w].effectiveLD < minLD) minLD = g_workers[w].effectiveLD;
    return minLD;
}

void setEffectiveLD(int ld)
{
    for (int w = 0; w < NUM_WORKERS; w++)
        g_workers[w].effectiveLD = ld;
}

void printTTStats()
{
#if VERBOSE_LOGGING
    // Aggregate stats from all workers
    uint64 gpuCallCount = 0;
    double gpuCallTotalTime = 0;
    double gpuCallMinTime = 1e30;
    double gpuCallMaxTime = 0;
    uint64 hostTTHits = 0, hostTTMisses = 0;
    size_t iterPeakBfsMemory = 0;
    uint64 oomFallbackCount = 0;
    uint64 callSizeHist[7] = {0};
    uint64 callSizeLeafSum[7] = {0};
    double callSizeTimeSum[7] = {0};
    uint64 callTimeHist[6] = {0};
    double callTimeTimeSum[6] = {0};
    uint64 bfsLevelPositionSum[20] = {0};
    uint64 bfsLevelCallCount[20] = {0};
    uint64 totalLeafPositions = 0;
    int lastBfsDepth = 0, lastBfsNumLevels = 0;

    for (int w = 0; w < NUM_WORKERS; w++)
    {
        gpuCallCount += g_workers[w].gpuCallCount;
        gpuCallTotalTime += g_workers[w].gpuCallTotalTime;
        if (g_workers[w].gpuCallMinTime < gpuCallMinTime) gpuCallMinTime = g_workers[w].gpuCallMinTime;
        if (g_workers[w].gpuCallMaxTime > gpuCallMaxTime) gpuCallMaxTime = g_workers[w].gpuCallMaxTime;
        hostTTHits += g_workers[w].hostTTHits;
        hostTTMisses += g_workers[w].hostTTMisses;
        if (g_workers[w].iterPeakBfsMemory > iterPeakBfsMemory) iterPeakBfsMemory = g_workers[w].iterPeakBfsMemory;
        oomFallbackCount += g_workers[w].oomFallbackCount;
        totalLeafPositions += g_workers[w].totalLeafPositions;
        if (g_workers[w].lastBfsDepth > lastBfsDepth) lastBfsDepth = g_workers[w].lastBfsDepth;
        if (g_workers[w].lastBfsNumLevels > lastBfsNumLevels) lastBfsNumLevels = g_workers[w].lastBfsNumLevels;
        for (int i = 0; i < 7; i++) {
            callSizeHist[i] += g_workers[w].callSizeHist[i];
            callSizeLeafSum[i] += g_workers[w].callSizeLeafSum[i];
            callSizeTimeSum[i] += g_workers[w].callSizeTimeSum[i];
        }
        for (int i = 0; i < 6; i++) {
            callTimeHist[i] += g_workers[w].callTimeHist[i];
            callTimeTimeSum[i] += g_workers[w].callTimeTimeSum[i];
        }
        for (int i = 0; i < 20; i++) {
            bfsLevelPositionSum[i] += g_workers[w].bfsLevelPositionSum[i];
            bfsLevelCallCount[i] += g_workers[w].bfsLevelCallCount[i];
        }
    }

    if (gpuCallCount > 0)
    {
        double avgMs = (gpuCallTotalTime / gpuCallCount) * 1000.0;
        printf("  GPU BFS calls: %llu, total: %.3fs, avg: %.3fms, min: %.3fms, max: %.3fms",
               (unsigned long long)gpuCallCount, gpuCallTotalTime,
               avgMs, gpuCallMinTime * 1000.0, gpuCallMaxTime * 1000.0);
        if (oomFallbackCount > 0)
            printf(", OOM fallbacks: %llu", (unsigned long long)oomFallbackCount);
        printf("\n");
    }
    if (hostTTHits > 0 || hostTTMisses > 0)
    {
        uint64 total = hostTTHits + hostTTMisses;
        printf("  Host TT probes: %llu hits / %llu total (%.1f%%)\n",
               (unsigned long long)hostTTHits, (unsigned long long)total,
               total > 0 ? 100.0 * hostTTHits / total : 0.0);
    }
    if (iterPeakBfsMemory > 0)
    {
        size_t perWorkerBuf = PREALLOCATED_MEMORY_SIZE / NUM_WORKERS;
        printf("  Peak BFS memory: %llu MB / %llu MB (%.1f%%) per worker\n",
               (unsigned long long)(iterPeakBfsMemory / (1024*1024)),
               (unsigned long long)(perWorkerBuf / (1024*1024)),
               100.0 * iterPeakBfsMemory / perWorkerBuf);
    }
    if (gpuCallCount > 10)
    {
        const char *sizeLabels[] = {"0", "1-100", "101-1K", "1K-10K", "10K-100K", "100K-1M", "1M+"};
        printf("  Call size distribution (by leaf positions):\n");
        for (int i = 0; i < 7; i++)
        {
            if (callSizeHist[i] == 0) continue;
            double avgMs = callSizeTimeSum[i] / callSizeHist[i] * 1000.0;
            printf("    %-10s: %6llu calls (%5.1f%%), avg %.2fms",
                   sizeLabels[i],
                   (unsigned long long)callSizeHist[i],
                   100.0 * callSizeHist[i] / gpuCallCount,
                   avgMs);
            if (callSizeLeafSum[i] > 1000)
                printf(", %.1f ms/Mleaf", callSizeTimeSum[i] * 1000.0 / (callSizeLeafSum[i] / 1e6));
            printf("\n");
        }

        const char *timeLabels[] = {"<0.01ms", "0.01-0.1ms", "0.1-1ms", "1-10ms", "10-100ms", "100ms+"};
        printf("  Call time distribution:\n");
        for (int i = 0; i < 6; i++)
        {
            if (callTimeHist[i] == 0) continue;
            printf("    %-12s: %6llu calls (%5.1f%%), %.3fs total (%5.1f%%)\n",
                   timeLabels[i],
                   (unsigned long long)callTimeHist[i],
                   100.0 * callTimeHist[i] / gpuCallCount,
                   callTimeTimeSum[i],
                   gpuCallTotalTime > 0 ? 100.0 * callTimeTimeSum[i] / gpuCallTotalTime : 0.0);
        }

        if (bfsLevelCallCount[0] > 0)
        {
            printf("  Avg BFS level sizes (across %llu calls, perft(%d)):\n",
                   (unsigned long long)gpuCallCount, lastBfsDepth);
            for (int i = 0; i < 20 && bfsLevelCallCount[i] > 0; i++)
            {
                uint64 avgPos = bfsLevelPositionSum[i] / bfsLevelCallCount[i];
                int remDepth = lastBfsDepth - 1 - i;
                bool isLeaf = (i == lastBfsNumLevels - 1);
                printf("    Level %d (RD=%d%s): avg %llu positions",
                       i, remDepth >= 0 ? remDepth : 0, isLeaf ? ", leaf" : "",
                       (unsigned long long)avgPos);
                if (i > 0 && bfsLevelCallCount[i-1] > 0)
                {
                    uint64 prevAvg = bfsLevelPositionSum[i-1] / bfsLevelCallCount[i-1];
                    if (prevAvg > 0)
                        printf(" (%.1fx)", (double)avgPos / prevAvg);
                }
                printf("\n");
            }
        }
    }
#if USE_TT
    // Lossless host TT fill levels
    for (int d = 0; d < MAX_TT_DEPTH; d++)
    {
        if (!hostLosslessTTs[d].pool) continue;
        int32_t used = hostLosslessTTs[d].nextFree;
        int32_t capacity = hostLosslessTTs[d].poolCapacity;
        double pct = capacity > 0 ? 100.0 * used / capacity : 0.0;
        const char *usedUnit = ""; int32_t usedVal = used;
        if (used >= 1024*1024) { usedUnit = "M"; usedVal = used / (1024*1024); }
        else if (used >= 1024) { usedUnit = "K"; usedVal = used / 1024; }
        printf("  Host TT[%d]: %d%s / %dM entries (%.2f%%)",
               d, usedVal, usedUnit, capacity / (1024*1024), pct);
        if (used >= capacity)
            printf(" *** FULL - new stores dropped! ***");
        printf("\n");
    }
#endif // USE_TT
    fflush(stdout);
#endif // VERBOSE_LOGGING
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
        free(hostLosslessTTs[d].buckets);
        free(hostLosslessTTs[d].pool);
        memset(&hostLosslessTTs[d], 0, sizeof(LosslessTT));
    }
#endif
}

uint32 estimateLaunchDepth(QuadBitBoard *pos, GameState *gs, uint8 rootColor, float *outBranchingFactor)
{
    double perft1 = (double)perft_cpu_dispatch(pos, gs, rootColor, 1);
    double perft2 = (double)perft_cpu_dispatch(pos, gs, rootColor, 2);
    double perft3 = (double)perft_cpu_dispatch(pos, gs, rootColor, 3);

    float geoMean = sqrt((perft3 / perft2) * (perft2 / perft1));
    float arithMean = ((perft3 / perft2) + (perft2 / perft1)) / 2;

    float branchingFactor = (geoMean + arithMean) / 2;
    if (outBranchingFactor) *outBranchingFactor = branchingFactor;
    if (arithMean / geoMean > 2.0f)
    {
        printf("\nUnstable position, defaulting to launch depth = 5\n");
        return 5;
    }

    // With the fused 2-level leaf kernel, the last BFS level's huge move/index
    // arrays are eliminated. Effective budget per worker = bufferSize * branchingFactor.
    float memLimit = (float)PREALLOCATED_MEMORY_SIZE / 2.0f * branchingFactor;

    uint32 depth = (uint32)(log(memLimit) / log(branchingFactor));

    return depth;
}


// -------------------------------------------------------------------------
// CPU recursion with WorkerState (for GPU perft path)
// -------------------------------------------------------------------------

static uint64 perft_cpu_recurse(QuadBitBoard *pos, GameState *gs, uint8 color, int depth, int launchDepth, Hash128 hash, WorkerState *ws)
{
#if USE_TT
    // Probe host TT (all depths use lossless tables)
    if (depth >= 2)
    {
        uint64 ttCount;
        if (losslessProbe(hostLosslessTTs[depth], hash, &ttCount))
        {
#if VERBOSE_LOGGING
            ws->hostTTHits++;
#endif
            return ttCount;
        }
#if VERBOSE_LOGGING
        ws->hostTTMisses++;
#endif
    }
#endif

    uint64 count;
    bool needCpuFallback = false;

    if (depth <= ws->effectiveLD)
    {
#if VERBOSE_LOGGING
        auto t0 = std::chrono::high_resolution_clock::now();
#endif
        count = perft_gpu_host_bfs(pos, gs, color, depth, ws);

        if (ws->lastBfsOom)
        {
            needCpuFallback = true;
            ws->oomFallbackCount++;
            if (ws->effectiveLD > 2 && depth >= ws->effectiveLD)
            {
                printf("  >> Worker OOM at GPU depth %d, decreasing effective LD: %d -> %d\n",
                       depth, ws->effectiveLD, ws->effectiveLD - 1);
                ws->effectiveLD--;
            }
        }
#if VERBOSE_LOGGING
        else
        {
            double secs = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t0).count();
            ws->gpuCallCount++;
            ws->gpuCallTotalTime += secs;
            if (secs < ws->gpuCallMinTime) ws->gpuCallMinTime = secs;
            if (secs > ws->gpuCallMaxTime) ws->gpuCallMaxTime = secs;
            if (ws->lastBfsPeakMemory > ws->iterPeakBfsMemory) ws->iterPeakBfsMemory = ws->lastBfsPeakMemory;

            int leafCount = ws->lastLeafCount;
            int bucket;
            if (leafCount == 0) bucket = 0;
            else if (leafCount <= 100) bucket = 1;
            else if (leafCount <= 1000) bucket = 2;
            else if (leafCount <= 10000) bucket = 3;
            else if (leafCount <= 100000) bucket = 4;
            else if (leafCount <= 1000000) bucket = 5;
            else bucket = 6;
            ws->callSizeHist[bucket]++;
            ws->callSizeLeafSum[bucket] += leafCount;
            ws->callSizeTimeSum[bucket] += secs;
            ws->totalLeafPositions += leafCount;

            double ms = secs * 1000.0;
            int tbucket;
            if (ms < 0.01) tbucket = 0;
            else if (ms < 0.1) tbucket = 1;
            else if (ms < 1.0) tbucket = 2;
            else if (ms < 10.0) tbucket = 3;
            else if (ms < 100.0) tbucket = 4;
            else tbucket = 5;
            ws->callTimeHist[tbucket]++;
            ws->callTimeTimeSum[tbucket] += secs;

            for (int i = 0; i < ws->lastBfsNumLevels && i < 20; i++)
            {
                ws->bfsLevelPositionSum[i] += ws->lastBfsLevelCounts[i];
                ws->bfsLevelCallCount[i]++;
            }

            double wallElapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - ws->iterStartWallTime).count();
            if (wallElapsed - ws->lastProgressWallTime >= 10.0)
            {
                uint64 totalProbes = ws->hostTTHits + ws->hostTTMisses;
                printf("  [%.0fs] %llu GPU calls (%.0f/s), host TT %.1f%%, avg call: %.2fms, avg leaf: %llu, OOM: %llu\n",
                       wallElapsed,
                       (unsigned long long)ws->gpuCallCount,
                       (double)ws->gpuCallCount / wallElapsed,
                       totalProbes > 0 ? 100.0 * ws->hostTTHits / totalProbes : 0.0,
                       ws->gpuCallTotalTime / ws->gpuCallCount * 1000.0,
                       (unsigned long long)(ws->totalLeafPositions / ws->gpuCallCount),
                       (unsigned long long)ws->oomFallbackCount);
                fflush(stdout);
                ws->lastProgressWallTime = wallElapsed;
            }
        }
#endif
    }

    if (depth > ws->effectiveLD || needCpuFallback)
    {
        // Serial CPU recursion (normal path or GPU OOM fallback)
        CMove moves[MAX_MOVES];
        QuadBitBoard childPos;
        GameState childGs;
        int nMoves = generateMoves(pos, gs, color, moves);

        count = 0;
        for (int i = 0; i < nMoves; i++)
        {
            childPos = *pos;
            childGs = *gs;

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

            count += perft_cpu_recurse(&childPos, &childGs, !color, depth - 1, launchDepth, childHash, ws);
        }
    }

#if USE_TT
    // Store in host TT (all depths use lossless tables)
    if (depth >= 2)
    {
        losslessStore(hostLosslessTTs[depth], hash, count);
    }
#endif

    return count;
}


// Worker function: process a subset of root moves
static uint64 workerPerft(QuadBitBoard *pos, GameState *gs, uint8 rootColor, int depth,
                           int launchDepth, Hash128 rootHash,
                           CMove *rootMoves, int *moveIndices, int numMoves,
                           WorkerState *ws)
{
    uint64 total = 0;
    for (int i = 0; i < numMoves; i++)
    {
        QuadBitBoard childPos = *pos;
        GameState childGs = *gs;

        uint8 srcPiece = getPieceAt(pos, rootMoves[moveIndices[i]].getFrom());
        uint8 capPiece = getPieceAt(pos, rootMoves[moveIndices[i]].getTo());
        uint8 oldCastleRaw = gs->raw;
        uint8 oldEP = gs->enPassent;

        if (rootColor == WHITE)
            MoveGeneratorBitboard::makeMove<WHITE>(&childPos, &childGs, rootMoves[moveIndices[i]]);
        else
            MoveGeneratorBitboard::makeMove<BLACK>(&childPos, &childGs, rootMoves[moveIndices[i]]);

        Hash128 childHash = updateHashAfterMove(rootHash, rootMoves[moveIndices[i]], rootColor,
            srcPiece, capPiece, oldCastleRaw, childGs.raw, oldEP, childGs.enPassent);

        total += perft_cpu_recurse(&childPos, &childGs, !rootColor, depth - 1,
                                    launchDepth, childHash, ws);
    }
    return total;
}

void perftLauncher(QuadBitBoard *pos, GameState *gs, uint8 rootColor, uint32 depth, int launchDepth)
{
    Hash128 rootHash = computeHash(pos, gs, rootColor);

    // For depths <= launchDepth, single GPU call suffices (no threading)
    if (depth <= (uint32)launchDepth)
    {
        Timer timer;
        timer.start();
        uint64 result = perft_gpu_host_bfs(pos, gs, rootColor, depth, &g_workers[0]);
        timer.stop();
        double seconds = timer.elapsed();
        printf("\nPerft(%02d): %llu, time: %g seconds", depth, (unsigned long long)result, seconds);
        if (seconds > 0)
            printf(", nps: %llu", (unsigned long long)((double)result / seconds));
        printf("\n");
        fflush(stdout);
        return;
    }

    // Generate root moves on CPU
    CMove rootMoves[MAX_MOVES];
    int rootMoveCount = generateMoves(pos, gs, rootColor, rootMoves);

    if (rootMoveCount == 0)
    {
        printf("\nPerft(%02d): 0, time: 0 seconds\n", depth);
        fflush(stdout);
        return;
    }

    // Split moves: even indices -> worker 0, odd indices -> worker 1
    int moves0[MAX_MOVES], moves1[MAX_MOVES];
    int n0 = 0, n1 = 0;
    for (int i = 0; i < rootMoveCount; i++)
    {
        if (i % 2 == 0) moves0[n0++] = i;
        else             moves1[n1++] = i;
    }

    Timer timer;
    timer.start();

    uint64 result0 = 0, result1 = 0;

    if (n1 > 0)
    {
        // Spawn thread for worker 1 (odd moves)
        std::thread t1([&]() {
            result1 = workerPerft(pos, gs, rootColor, depth, launchDepth, rootHash,
                                   rootMoves, moves1, n1, &g_workers[1]);
        });

        // Main thread handles worker 0 (even moves)
        result0 = workerPerft(pos, gs, rootColor, depth, launchDepth, rootHash,
                               rootMoves, moves0, n0, &g_workers[0]);

        t1.join();
    }
    else
    {
        result0 = workerPerft(pos, gs, rootColor, depth, launchDepth, rootHash,
                               rootMoves, moves0, n0, &g_workers[0]);
    }

    timer.stop();
    uint64 result = result0 + result1;
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
    if (losslessProbe(hostLosslessTTs[depth], hash, &ttCount))
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
    losslessStore(hostLosslessTTs[depth], hash, count);
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
