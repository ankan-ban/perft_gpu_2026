#ifndef CHESS_H
#define CHESS_H

#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <assert.h>

#ifndef S_OK
#define S_OK 0
#endif

// various compile time settings
#include "switches.h"

typedef unsigned char      uint8;
typedef unsigned short     uint16;
typedef unsigned int       uint32;
typedef unsigned long long uint64;

#define HI(x) ((uint32)((x)>>32))
#define LO(x) ((uint32)(x))

#define CT_ASSERT(expr) \
int __static_assert(int static_assert_failed[(expr)?1:-1])

#define BIT(i)   (1ULL << (i))

// Terminology:
//
// file - column [A - H]
// rank - row    [1 - 8]


// piece constants
#define PAWN    1
#define KNIGHT  2
#define BISHOP  3
#define ROOK    4
#define QUEEN   5
#define KING    6

// chance (side) constants
#define WHITE   0
#define BLACK   1


// castle flags (1 and 2)
#define CASTLE_FLAG_KING_SIDE   1
#define CASTLE_FLAG_QUEEN_SIDE  2


enum eSquare
{
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
};

// Quad-bitboard representation: 4 bitboards encode piece type + color per square
// bb[0] = color bit (1 for black pieces, 0 for white/empty)
// bb[1] = piece bit 0 (set for PAWN, BISHOP, QUEEN)
// bb[2] = piece bit 1 (set for KNIGHT, BISHOP, KING)
// bb[3] = piece bit 2 (set for ROOK, QUEEN, KING)
// Piece encoding: PAWN=001, KNIGHT=010, BISHOP=011, ROOK=100, QUEEN=101, KING=110
struct QuadBitBoard
{
    uint64 bb[4];
};
CT_ASSERT(sizeof(QuadBitBoard) == 32);

// Game state stored separately for better coalescing
// 1-byte packed layout: bits [1:0]=whiteCastle, [3:2]=blackCastle, [7:4]=enPassent
// chance is removed — always passed as template param or function arg
struct GameState
{
    union {
        struct {
            uint8 whiteCastle : 2;
            uint8 blackCastle : 2;
            uint8 enPassent   : 4;   // file + 1
        };
        uint8 raw;  // for bulk castle-clear LUT operation
    };
};
CT_ASSERT(sizeof(GameState) == 1);

// a more compact move structure (16 bit)
// from http://chessprogramming.wikispaces.com/Encoding+Moves
class CMove
{
public:

    CUDA_CALLABLE_MEMBER CMove(uint8 from, uint8 to, uint8 flags)
    {
        m_Move = ((flags & 0xF) << 12) | ((to & 0x3F) << 6) | (from & 0x3F);
    }

    CUDA_CALLABLE_MEMBER CMove()
    {
        m_Move = 0;
    }

    CUDA_CALLABLE_MEMBER unsigned int getTo()    const {return (m_Move >> 6)  & 0x3F;}
    CUDA_CALLABLE_MEMBER unsigned int getFrom()  const {return (m_Move)       & 0x3F;}
    CUDA_CALLABLE_MEMBER unsigned int getFlags() const {return (m_Move >> 12) & 0x0F;}

protected:

   uint16 m_Move;

};

CT_ASSERT(sizeof(CMove) == 2);

enum eCompactMoveFlag
{
    CM_FLAG_QUIET_MOVE        = 0,

    CM_FLAG_DOUBLE_PAWN_PUSH  = 1,

    CM_FLAG_KING_CASTLE       = 2,
    CM_FLAG_QUEEN_CASTLE      = 3,

    CM_FLAG_CAPTURE           = 4,
    CM_FLAG_EP_CAPTURE        = 5,


    CM_FLAG_PROMOTION         = 8,

    CM_FLAG_KNIGHT_PROMOTION  = 8,
    CM_FLAG_BISHOP_PROMOTION  = 9,
    CM_FLAG_ROOK_PROMOTION    = 10,
    CM_FLAG_QUEEN_PROMOTION   = 11,

    CM_FLAG_KNIGHT_PROMO_CAP  = 12,
    CM_FLAG_BISHOP_PROMO_CAP  = 13,
    CM_FLAG_ROOK_PROMO_CAP    = 14,
    CM_FLAG_QUEEN_PROMO_CAP   = 15,
};

struct FancyMagicEntry
{
    union
    {
        struct {
            uint64  factor;     // the magic factor
            int     position;   // position in the main lookup table (of 97264 entries)

            int     _unused;    // padding for alignment
        };
#ifndef SKIP_CUDA_CODE
#ifdef __CUDA_ARCH__
        uint4 data;
#endif
#endif
    };
};

// Combined magic entry: merges occupancy mask + magic entry for single cache-line access.
// 32 bytes per entry = 1 cache sector. Reduces L1 traffic vs separate 8+16 byte loads.
struct CombinedMagicEntry
{
    uint64  mask;       // occupancy mask (was sqBishopAttacksMasked / sqRookAttacksMasked)
    uint64  factor;     // magic number
    int     position;   // position in the main lookup table
    int     _pad;       // padding to 32 bytes for aligned access
    uint64  _pad2;
};
CT_ASSERT(sizeof(CombinedMagicEntry) == 32);

// max no of moves possible for a given board position (this can be as large as 218 ?)
// e.g, test this FEN string "3Q4/1Q4Q1/4Q3/2Q4R/Q4Q2/3Q4/1Q4Rp/1K1BBNNk w - - 0 1"
#define MAX_MOVES 256


#endif
