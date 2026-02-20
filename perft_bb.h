#pragma once

#include "chess.h"

extern void *preAllocatedBufferHost;

void initMoveGen();
uint64 perft_bb(QuadBitBoard *pos, GameState *gs, uint8 color, uint32 depth);
uint64 perft_gpu_host_bfs(QuadBitBoard *pos, GameState *gs, uint8 rootColor, int depth, void *gpuBuffer, size_t bufferSize);
uint64 perft_cpu_dispatch(QuadBitBoard *pos, GameState *gs, uint8 color, uint32 depth);
