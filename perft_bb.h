#pragma once

#include "chess.h"

extern void *preAllocatedBufferHost;

void initMoveGen();
uint64 perft_bb(QuadBitBoard *pos, GameState *gs, uint32 depth);
uint64 perft_gpu_host_bfs(QuadBitBoard *pos, GameState *gs, int depth, void *gpuBuffer, size_t bufferSize);
