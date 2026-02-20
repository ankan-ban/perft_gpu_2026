#pragma once

#include "chess.h"

void initGPU(int gpu);
uint32 estimateLaunchDepth(QuadBitBoard *pos, GameState *gs, uint8 rootColor);
void perftLauncher(QuadBitBoard *pos, GameState *gs, uint8 rootColor, uint32 depth, int launchDepth);
void perftCPU(QuadBitBoard *pos, GameState *gs, uint8 rootColor, uint32 depth);
