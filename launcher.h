#pragma once

#include "chess.h"

// GPU memory buffer for BFS tree storage
extern void *preAllocatedBufferHost;
extern uint64 g_preAllocatedMemorySize;

// Runtime TT toggle (default: enabled, disable with -nott CLI flag)
extern bool g_useTT;

// Initialization
void initGPU(int gpu);
void initMoveGen();

// Launch depth estimation
uint32 estimateLaunchDepth(QuadBitBoard *pos, GameState *gs, uint8 rootColor, float *outBranchingFactor = nullptr);

// GPU perft
void perftLauncher(QuadBitBoard *pos, GameState *gs, uint8 rootColor, uint32 depth, int launchDepth);
uint64 perft_gpu_host_bfs(QuadBitBoard *pos, GameState *gs, uint8 rootColor, int depth, void *gpuBuffer, size_t bufferSize);

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

// Peak BFS memory from last GPU call (set in perft_kernels.cu)
extern size_t g_lastBfsPeakMemory;

// Per-GPU-call diagnostics (set in perft_kernels.cu)
#if VERBOSE_LOGGING
extern int g_lastLeafCount;
extern int g_lastBfsLevelCounts[20];
extern int g_lastBfsNumLevels;
extern int g_lastBfsDepth;
#endif
extern bool g_lastBfsOom;
