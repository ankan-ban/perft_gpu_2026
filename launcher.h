#pragma once

#include "chess.h"

void initGPU(int gpu);
uint32 estimateLaunchDepth(QuadBitBoard *pos, GameState *gs);
void perftLauncher(QuadBitBoard *pos, GameState *gs, uint32 depth, int launchDepth);
