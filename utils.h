#pragma once

#include <chrono>
#include "chess.h"

// Parse FEN string directly into QuadBitBoard + GameState + color
void readFENString(const char fen[], QuadBitBoard *qbb, GameState *gs, uint8 *color);

class Timer {
public:
    void start() { mStart = std::chrono::high_resolution_clock::now(); }
    void stop()  { mStop  = std::chrono::high_resolution_clock::now(); }
    double elapsed() const {
        return std::chrono::duration<double>(mStop - mStart).count();
    }
private:
    std::chrono::high_resolution_clock::time_point mStart, mStop;
};
