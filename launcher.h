#pragma once

#include "chess.h"
#include <chrono>

// Number of parallel workers (streams + threads)
#define NUM_WORKERS 2

// Per-worker state for multi-stream execution.
// Bundles all mutable state that was previously global/static.
struct WorkerState
{
    // GPU resources
    void        *gpuBuffer;
    size_t       bufferSize;
    cudaStream_t stream;

    // Pinned host memory for async D2H readback
    int          *pinnedInt;      // for nextLevelCount
    uint64       *pinnedU64;      // for result

    // Per-worker dedup generation (monotonically increasing, never reset)
    uint32       dedupGeneration;

    // Per-worker OOM / LD state
    int          effectiveLD;
    bool         lastBfsOom;
    size_t       lastBfsPeakMemory;
    uint64       oomFallbackCount;

#if VERBOSE_LOGGING
    // GPU BFS call tracking
    uint64       gpuCallCount;
    double       gpuCallTotalTime;
    double       gpuCallMinTime;
    double       gpuCallMaxTime;
    uint64       hostTTHits;
    uint64       hostTTMisses;
    size_t       iterPeakBfsMemory;
    int          lastLeafCount;
    int          lastBfsLevelCounts[20];
    int          lastBfsNumLevels;
    int          lastBfsDepth;
    uint64       callSizeHist[7];
    uint64       callSizeLeafSum[7];
    double       callSizeTimeSum[7];
    uint64       callTimeHist[6];
    double       callTimeTimeSum[6];
    uint64       bfsLevelPositionSum[20];
    uint64       bfsLevelCallCount[20];
    uint64       totalLeafPositions;
    std::chrono::steady_clock::time_point iterStartWallTime;
    double       lastProgressWallTime;
#endif
};

// Global workers array
extern WorkerState g_workers[NUM_WORKERS];

// GPU memory buffer for BFS tree storage (backward compat, points to g_workers[0].gpuBuffer)
extern void *preAllocatedBufferHost;

// Initialization
void initGPU(int gpu);
void initMoveGen();

// Launch depth estimation
uint32 estimateLaunchDepth(QuadBitBoard *pos, GameState *gs, uint8 rootColor, float *outBranchingFactor = nullptr);

// GPU perft
void perftLauncher(QuadBitBoard *pos, GameState *gs, uint8 rootColor, uint32 depth, int launchDepth);
uint64 perft_gpu_host_bfs(QuadBitBoard *pos, GameState *gs, uint8 rootColor, int depth, WorkerState *ws);

// CPU perft
void perftCPU(QuadBitBoard *pos, GameState *gs, uint8 rootColor, uint32 depth);
uint64 perft_cpu_dispatch(QuadBitBoard *pos, GameState *gs, uint8 color, uint32 depth);

// Transposition table management
void initTT(int launchDepth, int maxLaunchDepth, int maxDepth, float branchingFactor);
void clearDeviceTTs();
void freeTT();
void printTTStats();
void resetCallStats();
uint64 getOomFallbackCount();
int getEffectiveLD();
void setEffectiveLD(int ld);
