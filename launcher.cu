// launcher utils — GPU init, CPU perft, TT management

#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include "MoveGeneratorBitboard.h"
#include "launcher.h"
#include "utils.h"
#include "zobrist.h"
#include "tt.h"

// GPU memory buffer for BFS tree storage (allocated in initGPU, freed in main)
void *preAllocatedBufferHost;
uint64 g_preAllocatedMemorySize = 0;

// Runtime TT toggle (default: enabled, disable with -nott CLI flag)
bool g_useTT = true;

// Global TT arrays
TTTable deviceTTs[MAX_TT_DEPTH];
LosslessTT hostLosslessTTs[MAX_TT_DEPTH];

static uint64 g_oomFallbackCount = 0;
#if VERBOSE_LOGGING
// GPU BFS call tracking (reset per depth iteration)
static uint64 g_gpuCallCount = 0;
static double g_gpuCallTotalTime = 0.0;
static double g_gpuCallMinTime = 1e30;
static double g_gpuCallMaxTime = 0.0;
static uint64 g_hostTTHits = 0;
static uint64 g_hostTTMisses = 0;
static size_t g_iterPeakBfsMemory = 0;
#endif
#if VERBOSE_LOGGING
// GPU call diagnostics (histograms, per-BFS-level stats, progress)
static uint64 g_callSizeHist[7] = {0};      // by leaf count: 0, 1-100, 101-1K, 1K-10K, 10K-100K, 100K-1M, 1M+
static uint64 g_callSizeLeafSum[7] = {0};   // total leaf positions per size bucket
static double g_callSizeTimeSum[7] = {0};   // total time per size bucket
static uint64 g_callTimeHist[6] = {0};      // by duration: <0.01ms, 0.01-0.1ms, 0.1-1ms, 1-10ms, 10-100ms, 100ms+
static double g_callTimeTimeSum[6] = {0};   // total time per time bucket
static uint64 g_bfsLevelPositionSum[20] = {0};  // sum of positions at each BFS level across all calls
static uint64 g_bfsLevelCallCount[20] = {0};    // number of calls that had each BFS level
static uint64 g_totalLeafPositions = 0;
static std::chrono::steady_clock::time_point g_iterStartWallTime;
static double g_lastProgressWallTime = 0;
#endif

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

    // BFS buffer: with TT, use min(16GB, 33% of device mem); without TT, min(16GB, 95%)
    uint64 target = g_useTT ? (uint64)(total * 33 / 100) : (uint64)(total * 95 / 100);
    uint64 cap = PREALLOCATED_MEMORY_SIZE;
    g_preAllocatedMemorySize = (target < cap) ? target : cap;

    cudaStatus = cudaMalloc(&preAllocatedBufferHost, g_preAllocatedMemorySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "error in malloc for preAllocatedBuffer, error desc: %s\n", cudaGetErrorString(cudaStatus));
        exit(1);
    }
    printf("Allocated BFS buffer: %llu MB (%.0f%% of %llu MB total)\n",
           (unsigned long long)(g_preAllocatedMemorySize / (1024*1024)),
           100.0 * g_preAllocatedMemorySize / total,
           (unsigned long long)(total / (1024*1024)));
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
    if (!g_useTT) return;

    memset(deviceTTs, 0, sizeof(deviceTTs));
    memset(hostLosslessTTs, 0, sizeof(hostLosslessTTs));

    // Device TTs: depths 3 through maxLaunchDepth-1 (GPU BFS levels)
    // Allocate for the widest possible LD range to support dynamic LD increase.
    // Depth 2 is unused: bfsMinLevel=3 means no BFS level probes TT[2],
    // and HASH_IN_LEAF_KERNEL=0 means the fused leaf doesn't probe it either.
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
            // Auto-size: use 95% of free VRAM (after BFS buffer) for device TTs
            size_t freeMem = 0, totalMem = 0;
            cudaMemGetInfo(&freeMem, &totalMem);
            budgetBytes = (uint64)(freeMem * 95 / 100);
            printf("Auto device TT budget: %llu MB (95%% of %llu MB free)\n",
                   (unsigned long long)(budgetBytes / (1024*1024)),
                   (unsigned long long)(freeMem / (1024*1024)));
        }

        // Pick two power-of-2 entry counts: smallPow2 fits all TTs evenly,
        // largePow2 = 2x for upgrading shallow levels with leftover budget.
        uint64 perTableBytes = budgetBytes / numDeviceTTs;
        uint64 smallPow2 = floorPow2(perTableBytes / sizeof(TTEntry));
        if (smallPow2 < 1024) smallPow2 = 1024;
        uint64 largePow2 = smallPow2 * 2;

        // Determine which depths get the larger allocation.
        // Start with all at smallPow2, then upgrade from depth 3 upward.
        uint64 baseTotal = (uint64)numDeviceTTs * smallPow2 * sizeof(TTEntry);
        uint64 remaining = (baseTotal <= budgetBytes) ? budgetBytes - baseTotal : 0;
        uint64 upgradeBytes = (largePow2 - smallPow2) * sizeof(TTEntry);

        uint64 entryCount[MAX_TT_DEPTH];
        memset(entryCount, 0, sizeof(entryCount));
        for (int d = 3; d < maxLaunchDepth && d < MAX_TT_DEPTH; d++)
            entryCount[d] = smallPow2;

        // Upgrade shallow depths first (d=3, 4, 5, ...) while budget allows
        for (int d = 3; d < maxLaunchDepth && d < MAX_TT_DEPTH; d++)
        {
            if (remaining < upgradeBytes) break;
            entryCount[d] = largePow2;
            remaining -= upgradeBytes;
        }

        for (int d = 3; d < maxLaunchDepth && d < MAX_TT_DEPTH; d++)
        {
            uint64 entries = entryCount[d];

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
    // Budget split proportionally to branchingFactor^(maxDepth - d) so deeper
    // remaining depths (more unique positions) get larger tables.
    int numHostTTs = 0;
    for (int d = launchDepth; d <= maxDepth && d < MAX_TT_DEPTH; d++)
        numHostTTs++;

    if (numHostTTs > 0)
    {
        // Auto-detect: 90% of total system RAM
        if (g_hostTTBudgetMB <= 0)
        {
            uint64 totalRAM = 0;
#ifdef _WIN32
            MEMORYSTATUSEX memInfo;
            memInfo.dwLength = sizeof(memInfo);
            if (GlobalMemoryStatusEx(&memInfo))
                totalRAM = memInfo.ullTotalPhys;
#else
            long pages = sysconf(_SC_PHYS_PAGES);
            long pageSize = sysconf(_SC_PAGE_SIZE);
            if (pages > 0 && pageSize > 0)
                totalRAM = (uint64)pages * (uint64)pageSize;
#endif
            if (totalRAM > 0)
                g_hostTTBudgetMB = (int)((totalRAM * 9 / 10) / (1024 * 1024));
            else
                g_hostTTBudgetMB = 65536;  // fallback
            printf("Host TT budget: auto %d MB (90%% of %llu MB system RAM)\n",
                   g_hostTTBudgetMB, (unsigned long long)(totalRAM / (1024 * 1024)));
        }
        uint64 budgetBytes = (uint64)g_hostTTBudgetMB * 1024 * 1024;

        // Compute proportional weights
        double weights[MAX_TT_DEPTH];
        memset(weights, 0, sizeof(weights));
        double totalWeight = 0;
        for (int d = launchDepth; d <= maxDepth && d < MAX_TT_DEPTH; d++)
        {
            weights[d] = pow((double)branchingFactor, maxDepth - d);
            totalWeight += weights[d];
        }

        uint64 bytesPerSlot = sizeof(LosslessEntry) + sizeof(int32_t);  // pool entry + bucket

        for (int d = launchDepth; d <= maxDepth && d < MAX_TT_DEPTH; d++)
        {
            uint64 depthBytes = (uint64)(budgetBytes * weights[d] / totalWeight);
            uint64 numSlots = depthBytes / bytesPerSlot;
            numSlots = floorPow2(numSlots);
            if (numSlots < 4 * 1024 * 1024) numSlots = 4 * 1024 * 1024;  // 4M min — positions near root are expensive
            if (numSlots > (1ull << 30)) numSlots = (1ull << 30);  // cap for int32_t safety

            int32_t poolCap = (int32_t)numSlots;
            uint64 numBuckets = numSlots;

            hostLosslessTTs[d].buckets = (int32_t *)malloc(numBuckets * sizeof(int32_t));
            hostLosslessTTs[d].pool = (LosslessEntry *)malloc((uint64)poolCap * sizeof(LosslessEntry));
            if (hostLosslessTTs[d].buckets && hostLosslessTTs[d].pool)
            {
                memset(hostLosslessTTs[d].buckets, 0xFF, numBuckets * sizeof(int32_t));  // -1 = empty
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
}

void resetCallStats()
{
    g_oomFallbackCount = 0;
#if VERBOSE_LOGGING
    g_gpuCallCount = 0;
    g_gpuCallTotalTime = 0.0;
    g_gpuCallMinTime = 1e30;
    g_gpuCallMaxTime = 0.0;
    g_hostTTHits = 0;
    g_hostTTMisses = 0;
    g_iterPeakBfsMemory = 0;
    memset(g_callSizeHist, 0, sizeof(g_callSizeHist));
    memset(g_callSizeLeafSum, 0, sizeof(g_callSizeLeafSum));
    memset(g_callSizeTimeSum, 0, sizeof(g_callSizeTimeSum));
    memset(g_callTimeHist, 0, sizeof(g_callTimeHist));
    memset(g_callTimeTimeSum, 0, sizeof(g_callTimeTimeSum));
    memset(g_bfsLevelPositionSum, 0, sizeof(g_bfsLevelPositionSum));
    memset(g_bfsLevelCallCount, 0, sizeof(g_bfsLevelCallCount));
    g_totalLeafPositions = 0;
    g_iterStartWallTime = std::chrono::steady_clock::now();
    g_lastProgressWallTime = 0;
#endif
}

#if VERBOSE_LOGGING
size_t getIterPeakBfsMemory() { return g_iterPeakBfsMemory; }
#endif
uint64 getOomFallbackCount() { return g_oomFallbackCount; }

void printTTStats()
{
#if VERBOSE_LOGGING
    if (g_gpuCallCount > 0)
    {
        double avgMs = (g_gpuCallTotalTime / g_gpuCallCount) * 1000.0;
        printf("  GPU BFS calls: %llu, total: %.3fs, avg: %.3fms, min: %.3fms, max: %.3fms",
               (unsigned long long)g_gpuCallCount, g_gpuCallTotalTime,
               avgMs, g_gpuCallMinTime * 1000.0, g_gpuCallMaxTime * 1000.0);
        if (g_oomFallbackCount > 0)
            printf(", OOM fallbacks: %llu", (unsigned long long)g_oomFallbackCount);
        printf("\n");
    }
    if (g_hostTTHits > 0 || g_hostTTMisses > 0)
    {
        uint64 total = g_hostTTHits + g_hostTTMisses;
        printf("  Host TT probes: %llu hits / %llu total (%.1f%%)\n",
               (unsigned long long)g_hostTTHits, (unsigned long long)total,
               total > 0 ? 100.0 * g_hostTTHits / total : 0.0);
    }
    if (g_iterPeakBfsMemory > 0)
    {
        printf("  Peak BFS memory: %llu MB / %llu MB (%.1f%%)\n",
               (unsigned long long)(g_iterPeakBfsMemory / (1024*1024)),
               (unsigned long long)(g_preAllocatedMemorySize / (1024*1024)),
               100.0 * g_iterPeakBfsMemory / g_preAllocatedMemorySize);
    }
    // Detailed call diagnostics (only for iterations with significant GPU calls)
    if (g_gpuCallCount > 10)
    {
        const char *sizeLabels[] = {"0", "1-100", "101-1K", "1K-10K", "10K-100K", "100K-1M", "1M+"};
        printf("  Call size distribution (by leaf positions):\n");
        for (int i = 0; i < 7; i++)
        {
            if (g_callSizeHist[i] == 0) continue;
            double avgMs = g_callSizeTimeSum[i] / g_callSizeHist[i] * 1000.0;
            printf("    %-10s: %6llu calls (%5.1f%%), avg %.2fms",
                   sizeLabels[i],
                   (unsigned long long)g_callSizeHist[i],
                   100.0 * g_callSizeHist[i] / g_gpuCallCount,
                   avgMs);
            if (g_callSizeLeafSum[i] > 1000)
                printf(", %.1f ms/Mleaf", g_callSizeTimeSum[i] * 1000.0 / (g_callSizeLeafSum[i] / 1e6));
            printf("\n");
        }

        const char *timeLabels[] = {"<0.01ms", "0.01-0.1ms", "0.1-1ms", "1-10ms", "10-100ms", "100ms+"};
        printf("  Call time distribution:\n");
        for (int i = 0; i < 6; i++)
        {
            if (g_callTimeHist[i] == 0) continue;
            printf("    %-12s: %6llu calls (%5.1f%%), %.3fs total (%5.1f%%)\n",
                   timeLabels[i],
                   (unsigned long long)g_callTimeHist[i],
                   100.0 * g_callTimeHist[i] / g_gpuCallCount,
                   g_callTimeTimeSum[i],
                   100.0 * g_callTimeTimeSum[i] / g_gpuCallTotalTime);
        }

        // Per-BFS-level expansion stats
        if (g_bfsLevelCallCount[0] > 0)
        {
            int bfsDepth = g_lastBfsDepth;
            printf("  Avg BFS level sizes (across %llu calls, perft(%d)):\n",
                   (unsigned long long)g_gpuCallCount, bfsDepth);
            for (int i = 0; i < 20 && g_bfsLevelCallCount[i] > 0; i++)
            {
                uint64 avgPos = g_bfsLevelPositionSum[i] / g_bfsLevelCallCount[i];
                int remDepth = bfsDepth - 1 - i;
                bool isLeaf = (i == g_lastBfsNumLevels - 1);
                printf("    Level %d (RD=%d%s): avg %llu positions",
                       i, remDepth >= 0 ? remDepth : 0, isLeaf ? ", leaf" : "",
                       (unsigned long long)avgPos);
                if (i > 0 && g_bfsLevelCallCount[i-1] > 0)
                {
                    uint64 prevAvg = g_bfsLevelPositionSum[i-1] / g_bfsLevelCallCount[i-1];
                    if (prevAvg > 0)
                        printf(" (%.1fx)", (double)avgPos / prevAvg);
                }
                printf("\n");
            }
        }
    }
    if (g_useTT)
    {
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
    }
    fflush(stdout);
#endif // VERBOSE_LOGGING
}

void clearDeviceTTs()
{
    for (int d = 0; d < MAX_TT_DEPTH; d++)
    {
        if (deviceTTs[d].entries)
            cudaMemset(deviceTTs[d].entries, 0, (deviceTTs[d].mask + 1) * sizeof(TTEntry));
    }
}

void freeTT()
{
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
}

uint32 estimateLaunchDepth(QuadBitBoard *pos, GameState *gs, uint8 rootColor, float *outBranchingFactor)
{
    // estimate branching factor near the root
    double perft1 = (double)perft_cpu_dispatch(pos, gs, rootColor, 1);
    double perft2 = (double)perft_cpu_dispatch(pos, gs, rootColor, 2);
    double perft3 = (double)perft_cpu_dispatch(pos, gs, rootColor, 3);

    // this works well when the root position has very low branching factor (e.g, in case king is in check)
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
    // arrays are eliminated. This effectively multiplies the memory budget by
    // the branching factor, allowing a higher launch depth (fewer CPU calls).
    float memLimit = (float)g_preAllocatedMemorySize / 2.0f * branchingFactor;

    // estimated depth is log of memLimit in base 'branchingFactor'
    uint32 depth = (uint32)(log(memLimit) / log(branchingFactor));

    return depth;
}


// Effective launch depth — can be decreased mid-iteration on OOM
static int g_effectiveLD = 0;
int getEffectiveLD() { return g_effectiveLD; }
void setEffectiveLD(int ld) { g_effectiveLD = ld; }

// Serial CPU recursion at top levels, launching GPU BFS at launchDepth
static uint64 perft_cpu_recurse(QuadBitBoard *pos, GameState *gs, uint8 color, int depth, int launchDepth, Hash128 hash, void *gpuBuffer, size_t bufferSize)
{
    if (g_useTT && depth >= 2)
    {
        uint64 ttCount;
        if (losslessProbe(hostLosslessTTs[depth], hash, &ttCount))
        {
#if VERBOSE_LOGGING
            g_hostTTHits++;
#endif
            return ttCount;
        }
#if VERBOSE_LOGGING
        g_hostTTMisses++;
#endif
    }

    uint64 count;
    bool needCpuFallback = false;

    if (depth <= g_effectiveLD)
    {
#if VERBOSE_LOGGING
        auto t0 = std::chrono::high_resolution_clock::now();
#endif
        count = perft_gpu_host_bfs(pos, gs, color, depth, gpuBuffer, bufferSize);

        if (g_lastBfsOom)
        {
            needCpuFallback = true;
            g_oomFallbackCount++;
            if (g_effectiveLD > 2 && depth >= g_effectiveLD)
            {
                printf("  >> OOM at GPU depth %d, decreasing effective LD: %d -> %d\n",
                       depth, g_effectiveLD, g_effectiveLD - 1);
                g_effectiveLD--;
            }
        }
#if VERBOSE_LOGGING
        else
        {
            double secs = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t0).count();
            g_gpuCallCount++;
            g_gpuCallTotalTime += secs;
            if (secs < g_gpuCallMinTime) g_gpuCallMinTime = secs;
            if (secs > g_gpuCallMaxTime) g_gpuCallMaxTime = secs;
            if (g_lastBfsPeakMemory > g_iterPeakBfsMemory) g_iterPeakBfsMemory = g_lastBfsPeakMemory;

            int leafCount = g_lastLeafCount;
            int bucket;
            if (leafCount == 0) bucket = 0;
            else if (leafCount <= 100) bucket = 1;
            else if (leafCount <= 1000) bucket = 2;
            else if (leafCount <= 10000) bucket = 3;
            else if (leafCount <= 100000) bucket = 4;
            else if (leafCount <= 1000000) bucket = 5;
            else bucket = 6;
            g_callSizeHist[bucket]++;
            g_callSizeLeafSum[bucket] += leafCount;
            g_callSizeTimeSum[bucket] += secs;
            g_totalLeafPositions += leafCount;

            double ms = secs * 1000.0;
            int tbucket;
            if (ms < 0.01) tbucket = 0;
            else if (ms < 0.1) tbucket = 1;
            else if (ms < 1.0) tbucket = 2;
            else if (ms < 10.0) tbucket = 3;
            else if (ms < 100.0) tbucket = 4;
            else tbucket = 5;
            g_callTimeHist[tbucket]++;
            g_callTimeTimeSum[tbucket] += secs;

            for (int i = 0; i < g_lastBfsNumLevels && i < 20; i++)
            {
                g_bfsLevelPositionSum[i] += g_lastBfsLevelCounts[i];
                g_bfsLevelCallCount[i]++;
            }

            double wallElapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - g_iterStartWallTime).count();
            if (wallElapsed - g_lastProgressWallTime >= 10.0)
            {
                uint64 totalProbes = g_hostTTHits + g_hostTTMisses;
                printf("  [%.0fs] %llu GPU calls (%.0f/s), host TT %.1f%%, avg call: %.2fms, avg leaf: %llu, OOM: %llu\n",
                       wallElapsed,
                       (unsigned long long)g_gpuCallCount,
                       (double)g_gpuCallCount / wallElapsed,
                       totalProbes > 0 ? 100.0 * g_hostTTHits / totalProbes : 0.0,
                       g_gpuCallTotalTime / g_gpuCallCount * 1000.0,
                       (unsigned long long)(g_totalLeafPositions / g_gpuCallCount),
                       (unsigned long long)g_oomFallbackCount);
                fflush(stdout);
                g_lastProgressWallTime = wallElapsed;
            }
        }
#endif
    }

    if (depth > g_effectiveLD || needCpuFallback)
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

    if (g_useTT && depth >= 2)
    {
        losslessStore(hostLosslessTTs[depth], hash, count);
    }

    return count;
}


void perftLauncher(QuadBitBoard *pos, GameState *gs, uint8 rootColor, uint32 depth, int launchDepth)
{
    Hash128 rootHash = computeHash(pos, gs, rootColor);

    Timer timer;
    timer.start();

    uint64 result = perft_cpu_recurse(pos, gs, rootColor, depth, launchDepth, rootHash, preAllocatedBufferHost, g_preAllocatedMemorySize);

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

    if (g_useTT)
    {
        uint64 ttCount;
        if (losslessProbe(hostLosslessTTs[depth], hash, &ttCount))
            return ttCount;
    }

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

    if (g_useTT)
        losslessStore(hostLosslessTTs[depth], hash, count);

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
