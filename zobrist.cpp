// Zobrist random number initialization
// Uses the same pre-generated random numbers as the reference perft_gpu project
// (generated from Intel Ivy Bridge RDRAND instruction, stored in randoms.cpp)

#include "zobrist.h"
#include <string.h>

// CPU-side Zobrist tables
ZobristRandoms zobrist1;
ZobristRandoms zobrist2;

// Pre-generated random numbers (defined in randoms.cpp, from the reference perft_gpu project)
extern unsigned long long randoms[];

void initZobrist()
{
    // Use the same offsets as the reference project for the two Zobrist sets.
    // Our ZobristRandoms (781 uint64s) is slightly smaller than the reference's (782)
    // because we don't have the 'depth' field, but the first 781 fields match exactly:
    // pieces[2][6][64] (768) + castling[2][2] (4) + enPassant[8] (8) + sideToMove (1) = 781
    memcpy(&zobrist1, &randoms[1200], sizeof(ZobristRandoms));
    memcpy(&zobrist2, &randoms[333], sizeof(ZobristRandoms));
}
