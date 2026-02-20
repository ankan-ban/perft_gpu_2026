#pragma once

#include "chess.h"

// GPU memory buffer for BFS tree storage
extern void *preAllocatedBufferHost;

// Initialization
void initGPU(int gpu);
void initMoveGen();

// Launch depth estimation
uint32 estimateLaunchDepth(QuadBitBoard *pos, GameState *gs, uint8 rootColor);

// GPU perft
void perftLauncher(QuadBitBoard *pos, GameState *gs, uint8 rootColor, uint32 depth, int launchDepth);
uint64 perft_gpu_host_bfs(QuadBitBoard *pos, GameState *gs, uint8 rootColor, int depth, void *gpuBuffer, size_t bufferSize);

// CPU perft
void perftCPU(QuadBitBoard *pos, GameState *gs, uint8 rootColor, uint32 depth);
uint64 perft_cpu_dispatch(QuadBitBoard *pos, GameState *gs, uint8 color, uint32 depth);
