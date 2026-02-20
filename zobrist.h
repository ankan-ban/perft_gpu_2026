#pragma once

#include "chess.h"

// -------------------------------------------------------------------------
// 128-bit Zobrist hash for transposition table keying
// lo: used as TT index (masked to table size)
// hi: used for verification (XOR'd with count in lockless scheme)
// -------------------------------------------------------------------------

struct Hash128
{
    uint64 lo;
    uint64 hi;
};

CT_ASSERT(sizeof(Hash128) == 16);

// Zobrist random number tables
struct ZobristRandoms
{
    uint64 pieces[2][6][64];    // [color][pieceType-1][square]
    uint64 castling[2][2];      // [color][0=kingside, 1=queenside]
    uint64 enPassant[8];        // [file 0-7]
    uint64 sideToMove;
};

// CPU-side Zobrist tables (defined in zobrist.cpp)
extern ZobristRandoms zobrist1;
extern ZobristRandoms zobrist2;

// -------------------------------------------------------------------------
// GPU-side Zobrist tables and accessor macros
// -------------------------------------------------------------------------

#ifdef __CUDACC__

__device__ static ZobristRandoms g_zobrist1;
__device__ static ZobristRandoms g_zobrist2;

#ifdef __CUDA_ARCH__
#define ZOB1(field) __ldg(&g_zobrist1.field)
#define ZOB2(field) __ldg(&g_zobrist2.field)
#else
#define ZOB1(field) zobrist1.field
#define ZOB2(field) zobrist2.field
#endif

#else // !__CUDACC__

#define ZOB1(field) zobrist1.field
#define ZOB2(field) zobrist2.field

#endif // __CUDACC__

// -------------------------------------------------------------------------
// Initialization (generates deterministic randoms, copies to GPU)
// -------------------------------------------------------------------------

void initZobrist();

// -------------------------------------------------------------------------
// Extract piece type at a square from a QuadBitBoard
// Returns 0 if empty, PAWN(1)..KING(6) if occupied
// -------------------------------------------------------------------------

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint8 getPieceAt(const QuadBitBoard *pos, uint8 sq)
{
    uint64 bit = BIT(sq);
    uint8 piece = 0;
    if (pos->bb[1] & bit) piece |= 1;
    if (pos->bb[2] & bit) piece |= 2;
    if (pos->bb[3] & bit) piece |= 4;
    return piece;
}

// -------------------------------------------------------------------------
// Compute hash from scratch (for root position and debug validation)
// -------------------------------------------------------------------------

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE Hash128 computeHash(const QuadBitBoard *pos, const GameState *gs, uint8 color)
{
    Hash128 h = {0, 0};

    // Side to move: XOR if white (convention: hash includes sideToMove when white plays)
    if (color == WHITE)
    {
        h.lo ^= ZOB1(sideToMove);
        h.hi ^= ZOB2(sideToMove);
    }

    // Castling rights
    if (gs->whiteCastle & CASTLE_FLAG_KING_SIDE)
    {
        h.lo ^= ZOB1(castling[WHITE][0]);
        h.hi ^= ZOB2(castling[WHITE][0]);
    }
    if (gs->whiteCastle & CASTLE_FLAG_QUEEN_SIDE)
    {
        h.lo ^= ZOB1(castling[WHITE][1]);
        h.hi ^= ZOB2(castling[WHITE][1]);
    }
    if (gs->blackCastle & CASTLE_FLAG_KING_SIDE)
    {
        h.lo ^= ZOB1(castling[BLACK][0]);
        h.hi ^= ZOB2(castling[BLACK][0]);
    }
    if (gs->blackCastle & CASTLE_FLAG_QUEEN_SIDE)
    {
        h.lo ^= ZOB1(castling[BLACK][1]);
        h.hi ^= ZOB2(castling[BLACK][1]);
    }

    // En passant
    if (gs->enPassent)
    {
        h.lo ^= ZOB1(enPassant[gs->enPassent - 1]);
        h.hi ^= ZOB2(enPassant[gs->enPassent - 1]);
    }

    // Pieces: iterate all 64 squares
    for (uint8 sq = 0; sq < 64; sq++)
    {
        uint8 piece = getPieceAt(pos, sq);
        if (piece == 0) continue;

        // Determine color from bb[0]: bit set = black
        uint8 pieceColor = (pos->bb[0] & BIT(sq)) ? BLACK : WHITE;
        h.lo ^= ZOB1(pieces[pieceColor][piece - 1][sq]);
        h.hi ^= ZOB2(pieces[pieceColor][piece - 1][sq]);
    }

    return h;
}

// -------------------------------------------------------------------------
// Incremental hash update after makeMove
//
// Usage pattern (caller does NOT modify makeMove itself):
//   uint8 srcPiece = getPieceAt(pos, move.getFrom());
//   uint8 capPiece = getPieceAt(pos, move.getTo());
//   uint8 oldCastleRaw = gs->raw;
//   uint8 oldEP = gs->enPassent;
//   makeMove(pos, gs, move, color);  // unchanged
//   hash = updateHashAfterMove(hash, move, color, srcPiece, capPiece,
//                              oldCastleRaw, gs->raw, oldEP, gs->enPassent);
// -------------------------------------------------------------------------

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE Hash128 updateHashAfterMove(
    Hash128 h, CMove move, uint8 chance,
    uint8 srcPiece, uint8 capPiece,
    uint8 oldCastleRaw, uint8 newCastleRaw,
    uint8 oldEP, uint8 newEP)
{
    uint8 from  = move.getFrom();
    uint8 to    = move.getTo();
    uint8 flags = move.getFlags();

    // Flip side to move
    h.lo ^= ZOB1(sideToMove);
    h.hi ^= ZOB2(sideToMove);

    // Remove source piece
    h.lo ^= ZOB1(pieces[chance][srcPiece - 1][from]);
    h.hi ^= ZOB2(pieces[chance][srcPiece - 1][from]);

    // Determine placed piece (changes on promotion)
    uint8 placedPiece = srcPiece;
    if (flags & CM_FLAG_PROMOTION)
        placedPiece = (flags & 3) + KNIGHT;

    // Place piece at destination
    h.lo ^= ZOB1(pieces[chance][placedPiece - 1][to]);
    h.hi ^= ZOB2(pieces[chance][placedPiece - 1][to]);

    // Remove captured piece (regular captures)
    if (capPiece)
    {
        h.lo ^= ZOB1(pieces[!chance][capPiece - 1][to]);
        h.hi ^= ZOB2(pieces[!chance][capPiece - 1][to]);
    }

    // En passant capture: remove the captured pawn
    if (flags == CM_FLAG_EP_CAPTURE)
    {
        uint8 epPawnSq = (chance == WHITE) ? (to - 8) : (to + 8);
        h.lo ^= ZOB1(pieces[!chance][PAWN - 1][epPawnSq]);
        h.hi ^= ZOB2(pieces[!chance][PAWN - 1][epPawnSq]);
    }

    // Castling: move the rook
    if (flags == CM_FLAG_KING_CASTLE)
    {
        uint8 rookFrom = (chance == WHITE) ? H1 : H8;
        uint8 rookTo   = (chance == WHITE) ? F1 : F8;
        h.lo ^= ZOB1(pieces[chance][ROOK - 1][rookFrom]) ^ ZOB1(pieces[chance][ROOK - 1][rookTo]);
        h.hi ^= ZOB2(pieces[chance][ROOK - 1][rookFrom]) ^ ZOB2(pieces[chance][ROOK - 1][rookTo]);
    }
    else if (flags == CM_FLAG_QUEEN_CASTLE)
    {
        uint8 rookFrom = (chance == WHITE) ? A1 : A8;
        uint8 rookTo   = (chance == WHITE) ? D1 : D8;
        h.lo ^= ZOB1(pieces[chance][ROOK - 1][rookFrom]) ^ ZOB1(pieces[chance][ROOK - 1][rookTo]);
        h.hi ^= ZOB2(pieces[chance][ROOK - 1][rookFrom]) ^ ZOB2(pieces[chance][ROOK - 1][rookTo]);
    }

    // Castle rights: XOR only changed bits (oldRaw ^ newRaw)
    uint8 castleDelta = (oldCastleRaw ^ newCastleRaw) & 0x0F;
    if (castleDelta & 1) { h.lo ^= ZOB1(castling[WHITE][0]); h.hi ^= ZOB2(castling[WHITE][0]); }
    if (castleDelta & 2) { h.lo ^= ZOB1(castling[WHITE][1]); h.hi ^= ZOB2(castling[WHITE][1]); }
    if (castleDelta & 4) { h.lo ^= ZOB1(castling[BLACK][0]); h.hi ^= ZOB2(castling[BLACK][0]); }
    if (castleDelta & 8) { h.lo ^= ZOB1(castling[BLACK][1]); h.hi ^= ZOB2(castling[BLACK][1]); }

    // En passant file: XOR out old, XOR in new
    if (oldEP)
    {
        h.lo ^= ZOB1(enPassant[oldEP - 1]);
        h.hi ^= ZOB2(enPassant[oldEP - 1]);
    }
    if (newEP)
    {
        h.lo ^= ZOB1(enPassant[newEP - 1]);
        h.hi ^= ZOB2(enPassant[newEP - 1]);
    }

    return h;
}
