// Zobrist random number generation and initialization

#include "zobrist.h"

// CPU-side Zobrist tables
ZobristRandoms zobrist1;
ZobristRandoms zobrist2;

// splitmix64: high-quality PRNG for Zobrist key generation
// Reference: http://xoshiro.di.unimi.it/splitmix64.c
static uint64 splitmix64(uint64 &state)
{
    state += 0x9e3779b97f4a7c15ULL;
    uint64 z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void fillZobristRandoms(ZobristRandoms &zr, uint64 &state)
{
    for (int color = 0; color < 2; color++)
        for (int piece = 0; piece < 6; piece++)
            for (int sq = 0; sq < 64; sq++)
                zr.pieces[color][piece][sq] = splitmix64(state);

    for (int color = 0; color < 2; color++)
        for (int side = 0; side < 2; side++)
            zr.castling[color][side] = splitmix64(state);

    for (int file = 0; file < 8; file++)
        zr.enPassant[file] = splitmix64(state);

    zr.sideToMove = splitmix64(state);
}

void initZobrist()
{
    // Fixed seeds for deterministic, reproducible hash values
    uint64 state1 = 0xBEEF1234CAFE5678ULL;
    uint64 state2 = 0xDEAD5678FACE1234ULL;

    fillZobristRandoms(zobrist1, state1);
    fillZobristRandoms(zobrist2, state2);
}
