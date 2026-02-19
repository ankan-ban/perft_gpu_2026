#pragma once

#include "chess.h"

#ifdef __linux__ 
    #include <x86intrin.h>
    #define CPU_FORCE_INLINE inline
#else
    #include <intrin.h>
    #define CPU_FORCE_INLINE __forceinline
#endif

#include <time.h>


// bit board constants
#define C64(constantU64) constantU64##ULL

// valid locations for pawns
#ifndef RANKS2TO7
#define RANKS2TO7 C64(0x00FFFFFFFFFFFF00)
#endif

#define RANK1     C64(0x00000000000000FF)
#define RANK2     C64(0x000000000000FF00)
#define RANK3     C64(0x0000000000FF0000)
#define RANK4     C64(0x00000000FF000000)
#define RANK5     C64(0x000000FF00000000)
#define RANK6     C64(0x0000FF0000000000)
#define RANK7     C64(0x00FF000000000000)
#define RANK8     C64(0xFF00000000000000)

#define FILEA     C64(0x0101010101010101)
#define FILEB     C64(0x0202020202020202)
#define FILEC     C64(0x0404040404040404)
#define FILED     C64(0x0808080808080808)
#define FILEE     C64(0x1010101010101010)
#define FILEF     C64(0x2020202020202020)
#define FILEG     C64(0x4040404040404040)
#define FILEH     C64(0x8080808080808080)

#define DIAGONAL_A1H8   C64(0x8040201008040201)
#define DIAGONAL_A8H1   C64(0x0102040810204080)

#define CENTRAL_SQUARES C64(0x007E7E7E7E7E7E00)

// used for castling checks
#define F1G1      C64(0x60)
#define C1D1      C64(0x0C)
#define B1D1      C64(0x0E)

// used for castling checks
#define F8G8      C64(0x6000000000000000)
#define C8D8      C64(0x0C00000000000000)
#define B8D8      C64(0x0E00000000000000)

// used to update castle flags
#define WHITE_KING_SIDE_ROOK   C64(0x0000000000000080)
#define WHITE_QUEEN_SIDE_ROOK  C64(0x0000000000000001)
#define BLACK_KING_SIDE_ROOK   C64(0x8000000000000000)
#define BLACK_QUEEN_SIDE_ROOK  C64(0x0100000000000000)
    

#define ALLSET    C64(0xFFFFFFFFFFFFFFFF)
#define BB_EMPTY  C64(0x0)

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint8 popCount(uint64 x)
{
#ifdef __CUDA_ARCH__
    return __popcll(x);
#elif defined(_WIN64)
    return (uint8)_mm_popcnt_u64(x);
#elif defined(__linux__)
    return (uint8)_mm_popcnt_u64(x);
#else
    uint32 lo = (uint32)  x;
    uint32 hi = (uint32) (x >> 32);
    return _mm_popcnt_u32(lo) + _mm_popcnt_u32(hi);
#endif
}


// return the index of first set LSB
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint8 bitScan(uint64 x)
{
#ifdef __CUDA_ARCH__
    // __ffsll(x) returns position from 1 to 64 instead of 0 to 63
    return __ffsll(x) - 1;
#elif _WIN64
   unsigned long index;
   assert (x != 0);
   _BitScanForward64(&index, x);
   return (uint8) index;   
#elif __linux__
    return __builtin_ffsll(x) - 1;
#else
    uint32 lo = (uint32)  x;
    uint32 hi = (uint32) (x >> 32);
    unsigned long id;

    if (lo)
        _BitScanForward(&id, lo);
    else
    {
        _BitScanForward(&id, hi);
        id += 32;
    }

    return (uint8) id;
#endif
}


// CPU copy of all the below global variables are defined in GlobalVars.cpp
// bit mask containing squares between two given squares
extern uint64 Between[64][64];

// bit mask containing squares in the same 'line' as two given squares
extern uint64 Line[64][64];

// squares a piece can attack in an empty board
extern uint64 RookAttacks    [64];
extern uint64 BishopAttacks  [64];
extern uint64 QueenAttacks   [64];
extern uint64 KingAttacks    [64];
extern uint64 KnightAttacks  [64];
extern uint64 pawnAttacks[2] [64];

// magic lookup tables
#define ROOK_MAGIC_BITS    12
#define BISHOP_MAGIC_BITS  9

// same as RookAttacks and BishopAttacks, but corner bits masked off
extern uint64 RookAttacksMasked   [64];
extern uint64 BishopAttacksMasked [64];

// fancy magic tables
extern uint64 fancy_magic_lookup_table[97264];
extern FancyMagicEntry bishop_magics_fancy[64];
extern FancyMagicEntry rook_magics_fancy[64];

uint64 findRookMagicForSquare  (int square, uint64 magicAttackTable[], uint64 magic = 0, uint64 *uniqueAttackTable = NULL, uint8 *byteIndices = NULL, int *numUniqueAttacks = 0);
uint64 findBishopMagicForSquare(int square, uint64 magicAttackTable[], uint64 magic = 0, uint64 *uniqueAttackTable = NULL, uint8 *byteIndices = NULL, int *numUniqueAttacks = 0);

#ifndef SKIP_CUDA_CODE

// gpu version of the data structures
// accessed for read only using __ldg() function

// bit mask containing squares between two given squares
__device__ static uint64 gBetween[64][64];

// bit mask containing squares in the same 'line' as two given squares
__device__ static uint64 gLine[64][64];

// squares a piece can attack in an empty board
__device__ static uint64 gRookAttacks    [64];
__device__ static uint64 gBishopAttacks  [64];
__device__ static uint64 gQueenAttacks   [64];
__device__ static uint64 gKingAttacks    [64];
__device__ static uint64 gKnightAttacks  [64];
__device__ static uint64 gpawnAttacks[2] [64];


// Magical Tables
// same as RookAttacks and BishopAttacks, but corner bits masked off
__device__ static uint64 gRookAttacksMasked   [64];
__device__ static uint64 gBishopAttacksMasked [64];

// fancy magics
__device__ static uint64 g_fancy_magic_lookup_table[97264];
__device__ static FancyMagicEntry g_bishop_magics_fancy[64];
__device__ static FancyMagicEntry g_rook_magics_fancy[64];

// combined magic entries (mask + magic in one struct for single cache-line access)
__device__ static CombinedMagicEntry g_bishop_combined[64];
__device__ static CombinedMagicEntry g_rook_combined[64];

// Castle clear LUT: for each square, which castle bits to clear when a piece moves from/to it.
// Bits [1:0] = whiteCastle bits to clear, bits [3:2] = blackCastle bits to clear.
// Only 6 squares have non-zero entries: A1, E1, H1, A8, E8, H8.
__device__ static uint8 g_castleClear[64];

#endif //#ifndef SKIP_CUDA_CODE

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqsInBetweenLUT(uint8 sq1, uint8 sq2)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gBetween[sq1][sq2]);
#else
    return Between[sq1][sq2];
#endif
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqsInLineLUT(uint8 sq1, uint8 sq2)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gLine[sq1][sq2]);
#else
    return Line[sq1][sq2];
#endif
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqKnightAttacks(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gKnightAttacks[sq]);
#else
    return KnightAttacks[sq];
#endif
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqKingAttacks(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gKingAttacks[sq]);
#else
    return KingAttacks[sq];
#endif
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqRookAttacks(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gRookAttacks[sq]);
#else
    return RookAttacks[sq];
#endif
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqBishopAttacks(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gBishopAttacks[sq]);
#else
    return BishopAttacks[sq];
#endif
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqBishopAttacksMasked(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gBishopAttacksMasked[sq]);
#else
    return BishopAttacksMasked[sq];
#endif
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sqRookAttacksMasked(uint8 sq)
{
#ifdef __CUDA_ARCH__
    return __ldg(&gRookAttacksMasked[sq]);
#else
    return RookAttacksMasked[sq];
#endif
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint64 sq_fancy_magic_lookup_table(int index)
{
#ifdef __CUDA_ARCH__
    return __ldg(&g_fancy_magic_lookup_table[index]);
#else
    return fancy_magic_lookup_table[index];
#endif
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE FancyMagicEntry sq_bishop_magics_fancy(int sq)
{
#ifdef __CUDA_ARCH__
    FancyMagicEntry op;
    op.data = __ldg(&(((uint4 *)g_bishop_magics_fancy)[sq]));
    return op;
#else
    return bishop_magics_fancy[sq];
#endif
}

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE FancyMagicEntry sq_rook_magics_fancy(int sq)
{
#ifdef __CUDA_ARCH__
    FancyMagicEntry op;
    op.data = __ldg(&(((uint4 *)g_rook_magics_fancy)[sq]));
    return op;
#else
    return rook_magics_fancy[sq];
#endif
}

class MoveGeneratorBitboard
{
public:

    // move the bits in the bitboard one square in the required direction

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 northOne(uint64 x)
    {
        return x << 8;
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 southOne(uint64 x)
    {
        return x >> 8;
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 eastOne(uint64 x)
    {
        return (x << 1) & (~FILEA);
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 westOne(uint64 x)
    {
        return (x >> 1) & (~FILEH);
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 northEastOne(uint64 x)
    {
        return (x << 9) & (~FILEA);
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 northWestOne(uint64 x)
    {
        return (x << 7) & (~FILEH);
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 southEastOne(uint64 x)
    {
        return (x >> 7) & (~FILEA);
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 southWestOne(uint64 x)
    {
        return (x >> 9) & (~FILEH);
    }


    // fill the board in the given direction
    // taken from http://chessprogramming.wikispaces.com/


    // gen - generator  : starting positions
    // pro - propogator : empty squares / squares not of current side

    // uses kogge-stone algorithm

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 northFill(uint64 gen, uint64 pro)
    {
        gen |= (gen << 8) & pro;
        pro &= (pro << 8);
        gen |= (gen << 16) & pro;
        pro &= (pro << 16);
        gen |= (gen << 32) & pro;

        return gen;
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 southFill(uint64 gen, uint64 pro)
    {
        gen |= (gen >> 8) & pro;
        pro &= (pro >> 8);
        gen |= (gen >> 16) & pro;
        pro &= (pro >> 16);
        gen |= (gen >> 32) & pro;

        return gen;
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 eastFill(uint64 gen, uint64 pro)
    {
        pro &= ~FILEA;

        gen |= (gen << 1) & pro;
        pro &= (pro << 1);
        gen |= (gen << 2) & pro;
        pro &= (pro << 2);
        gen |= (gen << 3) & pro;

        return gen;
    }
    
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 westFill(uint64 gen, uint64 pro)
    {
        pro &= ~FILEH;

        gen |= (gen >> 1) & pro;
        pro &= (pro >> 1);
        gen |= (gen >> 2) & pro;
        pro &= (pro >> 2);
        gen |= (gen >> 3) & pro;

        return gen;
    }


    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 northEastFill(uint64 gen, uint64 pro)
    {
        pro &= ~FILEA;

        gen |= (gen << 9) & pro;
        pro &= (pro << 9);
        gen |= (gen << 18) & pro;
        pro &= (pro << 18);
        gen |= (gen << 36) & pro;

        return gen;
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 northWestFill(uint64 gen, uint64 pro)
    {
        pro &= ~FILEH;

        gen |= (gen << 7) & pro;
        pro &= (pro << 7);
        gen |= (gen << 14) & pro;
        pro &= (pro << 14);
        gen |= (gen << 28) & pro;

        return gen;
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 southEastFill(uint64 gen, uint64 pro)
    {
        pro &= ~FILEA;

        gen |= (gen >> 7) & pro;
        pro &= (pro >> 7);
        gen |= (gen >> 14) & pro;
        pro &= (pro >> 14);
        gen |= (gen >> 28) & pro;

        return gen;
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 southWestFill(uint64 gen, uint64 pro)
    {
        pro &= ~FILEH;

        gen |= (gen >> 9) & pro;
        pro &= (pro >> 9);
        gen |= (gen >> 18) & pro;
        pro &= (pro >> 18);
        gen |= (gen >> 36) & pro;

        return gen;
    }


    // attacks in the given direction
    // need to OR with ~(pieces of side to move) to avoid killing own pieces

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 northAttacks(uint64 gen, uint64 pro)
    {
        gen |= (gen << 8) & pro;
        pro &= (pro << 8);
        gen |= (gen << 16) & pro;
        pro &= (pro << 16);
        gen |= (gen << 32) & pro;

        return gen << 8;
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 southAttacks(uint64 gen, uint64 pro)
    {
        gen |= (gen >> 8) & pro;
        pro &= (pro >> 8);
        gen |= (gen >> 16) & pro;
        pro &= (pro >> 16);
        gen |= (gen >> 32) & pro;

        return gen >> 8;
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 eastAttacks(uint64 gen, uint64 pro)
    {
        pro &= ~FILEA;

        gen |= (gen << 1) & pro;
        pro &= (pro << 1);
        gen |= (gen << 2) & pro;
        pro &= (pro << 2);
        gen |= (gen << 3) & pro;

        return (gen << 1) & (~FILEA);
    }
    
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 westAttacks(uint64 gen, uint64 pro)
    {
        pro &= ~FILEH;

        gen |= (gen >> 1) & pro;
        pro &= (pro >> 1);
        gen |= (gen >> 2) & pro;
        pro &= (pro >> 2);
        gen |= (gen >> 3) & pro;

        return (gen >> 1) & (~FILEH);
    }


    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 northEastAttacks(uint64 gen, uint64 pro)
    {
        pro &= ~FILEA;

        gen |= (gen << 9) & pro;
        pro &= (pro << 9);
        gen |= (gen << 18) & pro;
        pro &= (pro << 18);
        gen |= (gen << 36) & pro;

        return (gen << 9) & (~FILEA);
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 northWestAttacks(uint64 gen, uint64 pro)
    {
        pro &= ~FILEH;

        gen |= (gen << 7) & pro;
        pro &= (pro << 7);
        gen |= (gen << 14) & pro;
        pro &= (pro << 14);
        gen |= (gen << 28) & pro;

        return (gen << 7) & (~FILEH);
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 southEastAttacks(uint64 gen, uint64 pro)
    {
        pro &= ~FILEA;

        gen |= (gen >> 7) & pro;
        pro &= (pro >> 7);
        gen |= (gen >> 14) & pro;
        pro &= (pro >> 14);
        gen |= (gen >> 28) & pro;

        return (gen >> 7) & (~FILEA);
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 southWestAttacks(uint64 gen, uint64 pro)
    {
        pro &= ~FILEH;

        gen |= (gen >> 9) & pro;
        pro &= (pro >> 9);
        gen |= (gen >> 18) & pro;
        pro &= (pro >> 18);
        gen |= (gen >> 36) & pro;

        return (gen >> 9) & (~FILEH);
    }


    // attacks by pieces of given type
    // pro - empty squares

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 bishopAttacksKoggeStone(uint64 bishops, uint64 pro)
    {
        return northEastAttacks(bishops, pro) |
               northWestAttacks(bishops, pro) |
               southEastAttacks(bishops, pro) |
               southWestAttacks(bishops, pro) ;
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 rookAttacksKoggeStone(uint64 rooks, uint64 pro)
    {
        return northAttacks(rooks, pro) |
               southAttacks(rooks, pro) |
               eastAttacks (rooks, pro) |
               westAttacks (rooks, pro) ;
    }


    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 bishopAttacks(uint64 bishop, uint64 pro)
    {
        uint8 square = bitScan(bishop);

#ifdef __CUDA_ARCH__
#if USE_COMBINED_MAGIC_GPU == 1
        // Combined load: mask + magic from single 32-byte struct
        const CombinedMagicEntry *entry = &g_bishop_combined[square];
        uint64 mask = __ldg(&entry->mask);
        uint64 factor = __ldg(&entry->factor);
        int position = __ldg(&entry->position);
        uint64 occ = (~pro) & mask;
        int index = (factor * occ) >> (64 - BISHOP_MAGIC_BITS);
        return sq_fancy_magic_lookup_table(position + index);
#else
        uint64 occ = (~pro) & sqBishopAttacksMasked(square);
        FancyMagicEntry magicEntry = sq_bishop_magics_fancy(square);
        int index = (magicEntry.factor * occ) >> (64 - BISHOP_MAGIC_BITS);
        return sq_fancy_magic_lookup_table(magicEntry.position + index);
#endif // USE_COMBINED_MAGIC_GPU

#else // CPU path
        uint64 occ = (~pro) & sqBishopAttacksMasked(square);
        uint64 magic  = bishop_magics_fancy[square].factor;
        uint64 index = (magic * occ) >> (64 - BISHOP_MAGIC_BITS);
        uint64 *table = &fancy_magic_lookup_table[bishop_magics_fancy[square].position];
        return table[index];
#endif
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 rookAttacks(uint64 rook, uint64 pro)
    {
        uint8 square = bitScan(rook);

#ifdef __CUDA_ARCH__
#if USE_COMBINED_MAGIC_GPU == 1
        // Combined load: mask + magic from single 32-byte struct
        const CombinedMagicEntry *entry = &g_rook_combined[square];
        uint64 mask = __ldg(&entry->mask);
        uint64 factor = __ldg(&entry->factor);
        int position = __ldg(&entry->position);
        uint64 occ = (~pro) & mask;
        int index = (factor * occ) >> (64 - ROOK_MAGIC_BITS);
        return sq_fancy_magic_lookup_table(position + index);
#else
        uint64 occ = (~pro) & sqRookAttacksMasked(square);
        FancyMagicEntry magicEntry = sq_rook_magics_fancy(square);
        int index = (magicEntry.factor * occ) >> (64 - ROOK_MAGIC_BITS);
        return sq_fancy_magic_lookup_table(magicEntry.position + index);
#endif // USE_COMBINED_MAGIC_GPU

#else // CPU path
        uint64 occ = (~pro) & sqRookAttacksMasked(square);
        uint64 magic  = rook_magics_fancy[square].factor;
        uint64 index = (magic * occ) >> (64 - ROOK_MAGIC_BITS);
        uint64 *table = &fancy_magic_lookup_table[rook_magics_fancy[square].position];
        return table[index];
#endif
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 multiBishopAttacks(uint64 bishops, uint64 pro)
    {
        uint64 attacks = 0;
        while(bishops)
        {
            uint64 bishop = getOne(bishops);
            attacks |= bishopAttacks(bishop, pro);
            bishops ^= bishop;
        }

        return attacks;
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 multiRookAttacks(uint64 rooks, uint64 pro)
    {
        uint64 attacks = 0;
        while(rooks)
        {
            uint64 rook = getOne(rooks);
            attacks |= rookAttacks(rook, pro);
            rooks ^= rook;
        }

        return attacks;
    }

CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 multiKnightAttacks(uint64 knights)
{
	uint64 attacks = 0;
	while(knights)
	{
		uint64 knight = getOne(knights);
		attacks |= sqKnightAttacks(bitScan(knight));
		knights ^= knight;
	}
	return attacks;
}


    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 kingAttacks(uint64 kingSet) 
    {
        uint64 attacks = eastOne(kingSet) | westOne(kingSet);
        kingSet       |= attacks;
        attacks       |= northOne(kingSet) | southOne(kingSet);
        return attacks;
    }

    // efficient knight attack generator
    // http://chessprogramming.wikispaces.com/Knight+Pattern
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 knightAttacks(uint64 knights) {
        uint64 l1 = (knights >> 1) & C64(0x7f7f7f7f7f7f7f7f);
        uint64 l2 = (knights >> 2) & C64(0x3f3f3f3f3f3f3f3f);
        uint64 r1 = (knights << 1) & C64(0xfefefefefefefefe);
        uint64 r2 = (knights << 2) & C64(0xfcfcfcfcfcfcfcfc);
        uint64 h1 = l1 | r1;
        uint64 h2 = l2 | r2;
        return (h1<<16) | (h1>>16) | (h2<<8) | (h2>>8);
    }



    // gets one bit (the LSB) from a bitboard
    // returns a bitboard containing that bit
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 getOne(uint64 x)
    {
        return x & (-x);
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static bool isMultiple(uint64 x)
    {
        return x ^ getOne(x);
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static bool isSingular(uint64 x)
    {
        return !isMultiple(x); 
    }


    // finds the squares in between the two given squares
    // taken from 
    // http://chessprogramming.wikispaces.com/Square+Attacked+By#Legality Test-In Between-Pure Calculation
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 squaresInBetween(uint8 sq1, uint8 sq2)
    {
        const uint64 m1   = C64(0xFFFFFFFFFFFFFFFF);
        const uint64 a2a7 = C64(0x0001010101010100);
        const uint64 b2g7 = C64(0x0040201008040200);
        const uint64 h1b7 = C64(0x0002040810204080);
        uint64 btwn, line, rank, file;
     
        btwn  = (m1 << sq1) ^ (m1 << sq2);
        file  =   (sq2 & 7) - (sq1   & 7);
        rank  =  ((sq2 | 7) -  sq1) >> 3 ;
        line  =      (   (file  &  7) - 1) & a2a7; // a2a7 if same file
        line += 2 * ((   (rank  &  7) - 1) >> 58); // b1g1 if same rank
        line += (((rank - file) & 15) - 1) & b2g7; // b2g7 if same diagonal
        line += (((rank + file) & 15) - 1) & h1b7; // h1b7 if same antidiag
        line *= btwn & -btwn; // mul acts like shift by smaller square
        return line & btwn;   // return the bits on that line inbetween
    }

    // returns the 'line' containing all pieces in the same file/rank/diagonal or anti-diagonal containing sq1 and sq2
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 squaresInLine(uint8 sq1, uint8 sq2)
    {
        // TODO: try to make it branchless?
        int fileDiff  =   (sq2 & 7) - (sq1 & 7);
        int rankDiff  =  ((sq2 | 7) -  sq1) >> 3 ;

        uint8 file = sq1 & 7;
        uint8 rank = sq1 >> 3;

        if (fileDiff == 0)  // same file
        {
            return FILEA << file;
        }
        if (rankDiff == 0)  // same rank
        {
            return RANK1 << (rank * 8);
        }
        if (fileDiff - rankDiff == 0)   // same diagonal (with slope equal to a1h8)
        {
            if (rank - file >= 0)
                return DIAGONAL_A1H8 << ((rank - file) * 8);
            else
                return DIAGONAL_A1H8 >> ((file - rank) * 8);
        }
        if (fileDiff + rankDiff == 0)  // same anti-diagonal (with slope equal to a8h1)
        {
            // for a8h1, rank + file = 7
            int shiftAmount = (rank + file - 7) * 8;
            if (shiftAmount >= 0)
                return DIAGONAL_A8H1 << shiftAmount;
            else
                return DIAGONAL_A8H1 >> (-shiftAmount);
        }

        // squares not on same line
        return 0;
    }


    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 sqsInBetween(uint8 sq1, uint8 sq2)
    {
        return sqsInBetweenLUT(sq1, sq2);
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 sqsInLine(uint8 sq1, uint8 sq2)
    {
        return sqsInLineLUT(sq1, sq2);
    }

    static void init();

    CUDA_CALLABLE_MEMBER static CPU_FORCE_INLINE uint64 findPinnedPieces (uint64 myKing, uint64 myPieces, uint64 enemyBishops, uint64 enemyRooks, uint64 allPieces, uint8 kingIndex)
    {
        // check for sliding attacks to the king's square

        // It doesn't matter if we process more attackers behind the first attackers
        // They will be taken care of when we check for no. of obstructing squares between king and the attacker
        /*
        uint64 b = bishopAttacks(myKing, ~enemyPieces) & enemyBishops;
        uint64 r = rookAttacks  (myKing, ~enemyPieces) & enemyRooks;
        */

        uint64 b = sqBishopAttacks(kingIndex) & enemyBishops;
        uint64 r = sqRookAttacks  (kingIndex) & enemyRooks;

        uint64 attackers = b | r;

        // for every attacker we need to chess if there is a single obstruction between 
        // the attacker and the king, and if so - the obstructor is pinned
        uint64 pinned = BB_EMPTY;
        while (attackers)
        {
            uint64 attacker = getOne(attackers);

            // bitscan shouldn't be too expensive but it will be good to 
            // figure out a way do find obstructions without having to get square index of attacker
            uint8 attackerIndex = bitScan(attacker);    // same as bitscan on attackers

            uint64 squaresInBetween = sqsInBetween(attackerIndex, kingIndex); // same as using obstructed() function
            uint64 piecesInBetween = squaresInBetween & allPieces;
            if (isSingular(piecesInBetween))
                pinned |= piecesInBetween;

            attackers ^= attacker;  // same as &= ~attacker
        }

        return pinned;
    }

    // returns bitmask of squares in threat by enemy pieces
    // the king shouldn't ever attempt to move to a threatened square
    // TODO: maybe make this tempelated on color?
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64 findAttackedSquares(uint64 emptySquares, uint64 enemyBishops, uint64 enemyRooks, 
                                      uint64 enemyPawns, uint64 enemyKnights, uint64 enemyKing, 
                                      uint64 myKing, uint8 enemyColor)
    {
        uint64 attacked = 0;

        // 1. pawn attacks
        if (enemyColor == WHITE)
        {
            attacked |= northEastOne(enemyPawns);
            attacked |= northWestOne(enemyPawns);
        }
        else
        {
            attacked |= southEastOne(enemyPawns);
            attacked |= southWestOne(enemyPawns);
        }

        // 2. knight attacks
        attacked |= knightAttacks(enemyKnights);

        // 3. bishop attacks — squares behind king are also under threat (king can't go there)
        uint64 proWithKing = emptySquares | myKing;
		attacked |= multiBishopAttacks(enemyBishops, proWithKing);

        // 4. rook attacks
		attacked |= multiRookAttacks(enemyRooks, proWithKing);

        // 5. King attacks
        attacked |= kingAttacks(enemyKing);
        
        // TODO: 
        // 1. figure out if we really need to mask off pieces on board
        //  - actually it seems better not to.. so that we can easily check if a capture move takes the king to check
        // 2. It might be faster to use the lookup table instead of computing (esp for king and knights).. DONE!
        return attacked/*& (emptySquares)*/;
    }


    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static void updateCastleFlag(GameState *gs, uint64 dst, uint8 chance)
    {
        if (chance == WHITE)
        {
            gs->blackCastle &= ~( ((dst & BLACK_KING_SIDE_ROOK ) >> H8)      |
                                   ((dst & BLACK_QUEEN_SIDE_ROOK) >> (A8-1))) ;
        }
        else
        {
            gs->whiteCastle &= ~( ((dst & WHITE_KING_SIDE_ROOK ) >> H1) |
                                   ((dst & WHITE_QUEEN_SIDE_ROOK) << 1)) ;
        }
    }

    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static void addCompactMove(uint32 *nMoves, CMove **genMoves, uint8 from, uint8 to, uint8 flags)
    {
        CMove move(from, to, flags);
        **genMoves = move;
        (*genMoves)++;
        (*nMoves)++;
    }


    // adds promotions if at promotion square
    // or normal pawn moves if not promotion
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static void addCompactPawnMoves(uint32 *nMoves, CMove **genMoves, uint8 from, uint64 dst, uint8 flags)
    {
        uint8 to = bitScan(dst);
        // promotion
        if (dst & (RANK1 | RANK8))
        {
            addCompactMove(nMoves, genMoves, from, to, flags | CM_FLAG_KNIGHT_PROMOTION);
            addCompactMove(nMoves, genMoves, from, to, flags | CM_FLAG_BISHOP_PROMOTION);
            addCompactMove(nMoves, genMoves, from, to, flags | CM_FLAG_QUEEN_PROMOTION);
            addCompactMove(nMoves, genMoves, from, to, flags | CM_FLAG_ROOK_PROMOTION);
        }
        else
        {
            addCompactMove(nMoves, genMoves, from, to, flags);
        }
    }


    // Helper macro: derive piece bitboards from QuadBitBoard
    // After this macro, the following local variables are defined:
    //   allPawns, knights, bishopQueens, rookQueens, kings, allPieces, whitePieces, blackPieces
#define DERIVE_PIECE_BITBOARDS(pos) \
    uint64 allPieces    = (pos)->bb[1] | (pos)->bb[2] | (pos)->bb[3]; \
    uint64 blackPieces  = (pos)->bb[0]; \
    uint64 whitePieces  = allPieces & ~blackPieces; \
    uint64 allPawns     = (pos)->bb[1] & ~(pos)->bb[2] & ~(pos)->bb[3]; \
    uint64 knights      = (pos)->bb[2] & ~(pos)->bb[1] & ~(pos)->bb[3]; \
    uint64 bishopQueens = (pos)->bb[1] & ((pos)->bb[2] ^ (pos)->bb[3]); \
    uint64 rookQueens   = (pos)->bb[3] & ~(pos)->bb[2]; \
    uint64 kings        = (pos)->bb[2] & (pos)->bb[3] & ~(pos)->bb[1];


    template<uint8 chance>
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint32 generateMovesOutOfCheck (QuadBitBoard *pos, GameState *gs, CMove *genMoves,
                                           uint64 allPawns, uint64 allPieces, uint64 myPieces,
                                           uint64 enemyPieces, uint64 pinned, uint64 threatened,
                                           uint8 kingIndex)
    {
        // derive piece bitboards from quad
        uint64 knights_      = pos->bb[2] & ~pos->bb[1] & ~pos->bb[3];
        uint64 bishopQueens_ = pos->bb[1] & (pos->bb[2] ^ pos->bb[3]);
        uint64 rookQueens_   = pos->bb[3] & ~pos->bb[2];
        uint64 kings_        = pos->bb[2] & pos->bb[3] & ~pos->bb[1];

        uint32 nMoves = 0;
        uint64 king = kings_ & myPieces;

        // figure out the no. of attackers
        uint64 attackers = 0;

        // pawn attacks
        uint64 enemyPawns = allPawns & enemyPieces;
        attackers |= ((chance == WHITE) ? (northEastOne(king) | northWestOne(king)) :
                                          (southEastOne(king) | southWestOne(king)) ) & enemyPawns;

        // knight attackers — use LUT since king is a single piece (saves ~16 ALU ops vs bulk knightAttacks)
        uint64 enemyKnights = knights_ & enemyPieces;
        attackers |= sqKnightAttacks(kingIndex) & enemyKnights;

        // bishop attackers
        uint64 enemyBishops = bishopQueens_ & enemyPieces;
        attackers |= bishopAttacks(king, ~allPieces) & enemyBishops;

        // rook attackers
        uint64 enemyRooks = rookQueens_ & enemyPieces;
        attackers |= rookAttacks(king, ~allPieces) & enemyRooks;


        // A. Try king moves to get the king out of check
        uint64 kingMoves = sqKingAttacks(kingIndex);

        kingMoves &= ~(threatened | myPieces);  // king can't move to a square under threat or a square containing piece of same side
        while(kingMoves)
        {
            uint64 dst = getOne(kingMoves);

            // TODO: set capture flag correctly
            addCompactMove(&nMoves, &genMoves, kingIndex, bitScan(dst), 0);
            kingMoves ^= dst;
        }


        // B. try moves to kill/block attacking pieces
        if (isSingular(attackers))
        {
            // Find the safe squares - i.e, if a dst square of a move is any of the safe squares,
            // it will take king out of check

            // for pawn and knight attack, the only option is to kill the attacking piece
            // for bishops rooks and queens, it's the line between the attacker and the king, including the attacker
            uint64 safeSquares = attackers | sqsInBetween(kingIndex, bitScan(attackers));

            // pieces that are pinned don't have any hope of saving the king
            // TODO: Think more about it
            myPieces &= ~pinned;

            // 1. pawn moves
            uint64 myPawns = allPawns & myPieces;

            // checking rank for pawn double pushes
            uint64 checkingRankDoublePush = RANK3 << (chance * 24);           // rank 3 or rank 6

            uint64 enPassentTarget = 0;
            if (gs->enPassent)
            {
                if (chance == BLACK)
                {
                    enPassentTarget = BIT(gs->enPassent - 1) << (8 * 2);
                }
                else
                {
                    enPassentTarget = BIT(gs->enPassent - 1) << (8 * 5);
                }
            }

            // en-passent can only save the king if the piece captured is the attacker
            uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(enPassentTarget) : northOne(enPassentTarget);
            if (enPassentCapturedPiece != attackers)
                enPassentTarget = 0;

            while (myPawns)
            {
                uint64 pawn = getOne(myPawns);

                // pawn push
                uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & (~allPieces);
                if (dst)
                {
                    if (dst & safeSquares)
                    {
                        addCompactPawnMoves(&nMoves, &genMoves, bitScan(pawn), dst, 0);
                    }
                    else
                    {
                        // double push (only possible if single push was possible and single push didn't save the king)
                        dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush):
                                                   southOne(dst & checkingRankDoublePush) ) & (safeSquares) &(~allPieces);

                        if (dst)
                        {
                            addCompactMove(&nMoves, &genMoves, bitScan(pawn), bitScan(dst), CM_FLAG_DOUBLE_PAWN_PUSH);
                        }
                    }
                }

                // captures (only one of the two captures will save the king.. if at all it does)
                uint64 westCapture = (chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn);
                uint64 eastCapture = (chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn);
                dst = (westCapture | eastCapture) & enemyPieces & safeSquares;
                if (dst)
                {
                    addCompactPawnMoves(&nMoves, &genMoves, bitScan(pawn), dst, CM_FLAG_CAPTURE);
                }

                // en-passent
                dst = (westCapture | eastCapture) & enPassentTarget;
                if (dst)
                {
                    addCompactMove(&nMoves, &genMoves, bitScan(pawn), bitScan(dst), CM_FLAG_EP_CAPTURE);
                }

                myPawns ^= pawn;
            }

            // 2. knight moves
            uint64 myKnights = (knights_ & myPieces);
            while (myKnights)
            {
                uint64 knight = getOne(myKnights);
                uint64 knightMoves = sqKnightAttacks(bitScan(knight)) & safeSquares;
                while (knightMoves)
                {
                    uint64 dst = getOne(knightMoves);
                    // TODO: set capture flag correctly
                    addCompactMove(&nMoves, &genMoves, bitScan(knight), bitScan(dst), 0);
                    knightMoves ^= dst;
                }
                myKnights ^= knight;
            }
            
            // 3. bishop moves
            uint64 bishops = bishopQueens_ & myPieces;
            while (bishops)
            {
                uint64 bishop = getOne(bishops);
                uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & safeSquares;

                while (bishopMoves)
                {
                    uint64 dst = getOne(bishopMoves);
                    addCompactMove(&nMoves, &genMoves, bitScan(bishop), bitScan(dst), 0);
                    bishopMoves ^= dst;
                }
                bishops ^= bishop;
            }

            // 4. rook moves
            uint64 rooks = rookQueens_ & myPieces;
            while (rooks)
            {
                uint64 rook = getOne(rooks);
                uint64 rookMoves = rookAttacks(rook, ~allPieces) & safeSquares;

                while (rookMoves)
                {
                    uint64 dst = getOne(rookMoves);
                    // TODO: set capture flag correctly
                    addCompactMove(&nMoves, &genMoves, bitScan(rook), bitScan(dst), 0);
                    rookMoves ^= dst;
                }
                rooks ^= rook;
            }

        }   // end of if single attacker
        else
        {
            // multiple threats => only king moves possible
        }

        return nMoves;
    }


    // generates moves for the given board position
    // returns the no of moves generated
    // genMoves contains the generated moves
    template <uint8 chance>
    CUDA_CALLABLE_MEMBER static uint32 generateMoves (QuadBitBoard *pos, GameState *gs, CMove *genMoves)
    {
        uint32 nMoves = 0;

        DERIVE_PIECE_BITBOARDS(pos);

        uint64 myPieces     = (chance == WHITE) ? whitePieces : blackPieces;
        uint64 enemyPieces  = (chance == WHITE) ? blackPieces : whitePieces;

        uint64 enemyBishops = bishopQueens & enemyPieces;
        uint64 enemyRooks   = rookQueens & enemyPieces;

        uint64 myKing     = kings & myPieces;
        uint8  kingIndex  = bitScan(myKing);

        uint64 pinned     = findPinnedPieces(myKing, myPieces, enemyBishops, enemyRooks, allPieces, kingIndex);

        uint64 threatened = findAttackedSquares(~allPieces, enemyBishops, enemyRooks, allPawns & enemyPieces,
                                                knights & enemyPieces, kings & enemyPieces,
                                                myKing, !chance);

        // king is in check: call special generate function to generate only the moves that take king out of check
        if (threatened & myKing)
        {
            return generateMovesOutOfCheck<chance>(pos, gs, genMoves, allPawns, allPieces, myPieces, enemyPieces,
                                                              pinned, threatened, kingIndex);
        }


        // generate king moves
        uint64 kingMoves = sqKingAttacks(kingIndex);

        uint8 captureFlag = 0;
        kingMoves &= ~(threatened | myPieces);  // king can't move to a square under threat or a square containing piece of same side
        while (kingMoves)
        {
            uint64 dst = getOne(kingMoves);
#ifdef __CUDA_ARCH__
            captureFlag = 0;
#else
            captureFlag = (dst & enemyPieces) ? CM_FLAG_CAPTURE : 0;
#endif
            addCompactMove(&nMoves, &genMoves, kingIndex, bitScan(dst), captureFlag);
            kingMoves ^= dst;
        }

        // generate knight moves (only non-pinned knights can move)
        uint64 myKnights = (knights & myPieces) & ~pinned;
        while (myKnights)
        {
            uint64 knight = getOne(myKnights);
            uint64 knightMoves = sqKnightAttacks(bitScan(knight)) & ~myPieces;
            while (knightMoves)
            {
                uint64 dst = getOne(knightMoves);
#ifdef __CUDA_ARCH__
                captureFlag = 0;
#else
                captureFlag = (dst & enemyPieces) ? CM_FLAG_CAPTURE : 0;
#endif

                addCompactMove(&nMoves, &genMoves, bitScan(knight), bitScan(dst), captureFlag);
                knightMoves ^= dst;
            }
            myKnights ^= knight;
        }



        // generate bishop (and queen) moves
        uint64 myBishops = bishopQueens & myPieces;

        // first deal with pinned bishops
        uint64 bishops = myBishops & pinned;
        while (bishops)
        {
            uint64 bishop = getOne(bishops);
            // TODO: bishopAttacks() function uses a kogge-stone sliding move generator. Switch to magics!
            uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & ~myPieces;
            bishopMoves &= sqsInLine(bitScan(bishop), kingIndex);    // pined sliding pieces can move only along the line

            while (bishopMoves)
            {
                uint64 dst = getOne(bishopMoves);
#ifdef __CUDA_ARCH__
                captureFlag = 0;
#else
                captureFlag = (dst & enemyPieces) ? CM_FLAG_CAPTURE : 0;
#endif
                addCompactMove(&nMoves, &genMoves, bitScan(bishop), bitScan(dst), captureFlag);
                bishopMoves ^= dst;
            }
            bishops ^= bishop;
        }

        // remaining bishops/queens
        bishops = myBishops & ~pinned;
        while (bishops)
        {
            uint64 bishop = getOne(bishops);
            uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & ~myPieces;

            while (bishopMoves)
            {
                uint64 dst = getOne(bishopMoves);
#ifdef __CUDA_ARCH__
                captureFlag = 0;
#else
                captureFlag = (dst & enemyPieces) ? CM_FLAG_CAPTURE : 0;
#endif
                addCompactMove(&nMoves, &genMoves, bitScan(bishop), bitScan(dst), captureFlag);
                bishopMoves ^= dst;
            }
            bishops ^= bishop;

        }


        // rook/queen moves
        uint64 myRooks = rookQueens & myPieces;

        // first deal with pinned rooks
        uint64 rooks = myRooks & pinned;
        while (rooks)
        {
            uint64 rook = getOne(rooks);
            uint64 rookMoves = rookAttacks(rook, ~allPieces) & ~myPieces;
            rookMoves &= sqsInLine(bitScan(rook), kingIndex);

            while (rookMoves)
            {
                uint64 dst = getOne(rookMoves);
#ifdef __CUDA_ARCH__
                captureFlag = 0;
#else
                captureFlag = (dst & enemyPieces) ? CM_FLAG_CAPTURE : 0;
#endif
                addCompactMove(&nMoves, &genMoves, bitScan(rook), bitScan(dst), captureFlag);
                rookMoves ^= dst;
            }
            rooks ^= rook;
        }

        // remaining rooks/queens
        rooks = myRooks & ~pinned;
        while (rooks)
        {
            uint64 rook = getOne(rooks);
            uint64 rookMoves = rookAttacks(rook, ~allPieces) & ~myPieces;

            while (rookMoves)
            {
                uint64 dst = getOne(rookMoves);
#ifdef __CUDA_ARCH__
                captureFlag = 0;
#else
                captureFlag = (dst & enemyPieces) ? CM_FLAG_CAPTURE : 0;
#endif
                addCompactMove(&nMoves, &genMoves, bitScan(rook), bitScan(dst), captureFlag);
                rookMoves ^= dst;
            }
            rooks ^= rook;
        }


        uint64 myPawns = allPawns & myPieces;

        // generate en-passent moves
        uint64 enPassentTarget = 0;
        if (gs->enPassent)
        {
            if (chance == BLACK)
            {
                enPassentTarget = BIT(gs->enPassent - 1) << (8 * 2);
            }
            else
            {
                enPassentTarget = BIT(gs->enPassent - 1) << (8 * 5);
            }
        }

        if (enPassentTarget)
        {
            uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(enPassentTarget) : northOne(enPassentTarget);

            uint64 epSources = (eastOne(enPassentCapturedPiece) | westOne(enPassentCapturedPiece)) & myPawns;

            while (epSources)
            {
                uint64 pawn = getOne(epSources);
                if (pawn & pinned)
                {
                    uint64 line = sqsInLine(bitScan(pawn), kingIndex);
                    if (enPassentTarget & line)
                    {
                        addCompactMove(&nMoves, &genMoves, bitScan(pawn), bitScan(enPassentTarget), CM_FLAG_EP_CAPTURE);
                    }
                }
                else
                {
                    // Check if removing both pawns reveals a rook attack on king along the rank.
                    // Mask to king's rank only — vertical attacks are irrelevant here.
                    uint64 modifiedOcc = allPieces ^ enPassentCapturedPiece ^ pawn;
                    uint64 rankMask = RANK1 << ((kingIndex >> 3) * 8);
                    uint64 causesCheck = rookAttacks(myKing, ~modifiedOcc) & enemyRooks & rankMask;
                    if (!causesCheck)
                    {
                        addCompactMove(&nMoves, &genMoves, bitScan(pawn), bitScan(enPassentTarget), CM_FLAG_EP_CAPTURE);
                    }
                }
                epSources ^= pawn;
            }
        }

        // 1. pawn moves

        // checking rank for pawn double pushes
        uint64 checkingRankDoublePush = RANK3 << (chance * 24);           // rank 3 or rank 6

        // first deal with pinned pawns
        uint64 pinnedPawns = myPawns & pinned;

        while (pinnedPawns)
        {
            uint64 pawn = getOne(pinnedPawns);
            uint8 pawnIndex = bitScan(pawn);    // same as bitscan on pinnedPawns

            // the direction of the pin (mask containing all squares in the line joining the king and the current piece)
            uint64 line = sqsInLine(pawnIndex, kingIndex);

            // pawn push
            uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & line & (~allPieces);
            if (dst) 
            {
                addCompactMove(&nMoves, &genMoves, pawnIndex, bitScan(dst), 0);

                // double push (only possible if single push was possible)
                dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): 
                                           southOne(dst & checkingRankDoublePush) ) & (~allPieces);
                if (dst) 
                {
                    addCompactMove(&nMoves, &genMoves, pawnIndex, bitScan(dst), CM_FLAG_DOUBLE_PAWN_PUSH);
                }
            }

            // captures
            // (either of them will be valid - if at all)
            dst  = ((chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn)) & line;
            dst |= ((chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn)) & line;
            
            if (dst & enemyPieces) 
            {
                addCompactPawnMoves(&nMoves, &genMoves, pawnIndex, dst, CM_FLAG_CAPTURE);
            }

            pinnedPawns ^= pawn;  // same as &= ~pawn (but only when we know that the first set contain the element we want to clear)
        }

        myPawns = myPawns & ~pinned;

        while (myPawns)
        {
            uint64 pawn = getOne(myPawns);

            // pawn push
            uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & (~allPieces);
            if (dst) 
            {
                addCompactPawnMoves(&nMoves, &genMoves, bitScan(pawn), dst, 0);

                // double push (only possible if single push was possible)
                dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush): 
                                           southOne(dst & checkingRankDoublePush) ) & (~allPieces);

                if (dst) addCompactPawnMoves(&nMoves, &genMoves, bitScan(pawn), dst, CM_FLAG_DOUBLE_PAWN_PUSH);
            }

            // captures
            uint64 westCapture = (chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn);
            dst = westCapture & enemyPieces;
            if (dst) addCompactPawnMoves(&nMoves, &genMoves, bitScan(pawn), dst, CM_FLAG_CAPTURE);

            uint64 eastCapture = (chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn);
            dst = eastCapture & enemyPieces;
            if (dst) addCompactPawnMoves(&nMoves, &genMoves, bitScan(pawn), dst, CM_FLAG_CAPTURE);

            myPawns ^= pawn;
        }

        // generate castling moves
        if (chance == WHITE)
        {
            if ((gs->whiteCastle & CASTLE_FLAG_KING_SIDE) &&
                !(F1G1 & allPieces) &&
                !(F1G1 & threatened))
            {
                addCompactMove(&nMoves, &genMoves, E1, G1, CM_FLAG_KING_CASTLE);
            }
            if ((gs->whiteCastle & CASTLE_FLAG_QUEEN_SIDE) &&
                !(B1D1 & allPieces) &&
                !(C1D1 & threatened))
            {
                addCompactMove(&nMoves, &genMoves, E1, C1, CM_FLAG_QUEEN_CASTLE);
            }
        }
        else
        {
            if ((gs->blackCastle & CASTLE_FLAG_KING_SIDE) &&
                !(F8G8 & allPieces) &&
                !(F8G8 & threatened))
            {
                addCompactMove(&nMoves, &genMoves, E8, G8, CM_FLAG_KING_CASTLE);
            }
            if ((gs->blackCastle & CASTLE_FLAG_QUEEN_SIDE) &&
                !(B8D8 & allPieces) &&
                !(C8D8 & threatened))
            {
                addCompactMove(&nMoves, &genMoves, E8, C8, CM_FLAG_QUEEN_CASTLE);
            }
        }

        return nMoves;
    }

    template<uint8 chance>
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint32 countMovesOutOfCheck (QuadBitBoard *pos, GameState *gs,
                                           uint64 allPawns, uint64 allPieces, uint64 myPieces,
                                           uint64 enemyPieces, uint64 pinned, uint64 threatened,
                                           uint8 kingIndex)
    {
        // derive piece bitboards from quad
        uint64 knights_      = pos->bb[2] & ~pos->bb[1] & ~pos->bb[3];
        uint64 bishopQueens_ = pos->bb[1] & (pos->bb[2] ^ pos->bb[3]);
        uint64 rookQueens_   = pos->bb[3] & ~pos->bb[2];
        uint64 kings_        = pos->bb[2] & pos->bb[3] & ~pos->bb[1];

        uint32 nMoves = 0;
        uint64 king = kings_ & myPieces;

        // figure out the no. of attackers
        uint64 attackers = 0;

        // pawn attacks
        uint64 enemyPawns = allPawns & enemyPieces;
        attackers |= ((chance == WHITE) ? (northEastOne(king) | northWestOne(king)) :
                                          (southEastOne(king) | southWestOne(king)) ) & enemyPawns;

        // knight attackers — use LUT since king is a single piece (saves ~16 ALU ops vs bulk knightAttacks)
        uint64 enemyKnights = knights_ & enemyPieces;
        attackers |= sqKnightAttacks(kingIndex) & enemyKnights;

        // bishop attackers
        uint64 enemyBishops = bishopQueens_ & enemyPieces;
        attackers |= bishopAttacks(king, ~allPieces) & enemyBishops;

        // rook attackers
        uint64 enemyRooks = rookQueens_ & enemyPieces;
        attackers |= rookAttacks(king, ~allPieces) & enemyRooks;


        // A. Try king moves to get the king out of check
        uint64 kingMoves = sqKingAttacks(kingIndex);

        kingMoves &= ~(threatened | myPieces);  // king can't move to a square under threat or a square containing piece of same side
        nMoves += popCount(kingMoves);

        // B. try moves to kill/block attacking pieces
        if (isSingular(attackers))
        {
            // Find the safe squares - i.e, if a dst square of a move is any of the safe squares,
            // it will take king out of check

            // for pawn and knight attack, the only option is to kill the attacking piece
            // for bishops rooks and queens, it's the line between the attacker and the king, including the attacker
            uint64 safeSquares = attackers | sqsInBetween(kingIndex, bitScan(attackers));

            // pieces that are pinned don't have any hope of saving the king
            // TODO: Think more about it
            myPieces &= ~pinned;

            // 1. pawn moves
            uint64 myPawns = allPawns & myPieces;

            // checking rank for pawn double pushes
            uint64 checkingRankDoublePush = RANK3 << (chance * 24);           // rank 3 or rank 6

            uint64 enPassentTarget = 0;
            if (gs->enPassent)
            {
                if (chance == BLACK)
                {
                    enPassentTarget = BIT(gs->enPassent - 1) << (8 * 2);
                }
                else
                {
                    enPassentTarget = BIT(gs->enPassent - 1) << (8 * 5);
                }
            }

            // en-passent can only save the king if the piece captured is the attacker
            uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(enPassentTarget) : northOne(enPassentTarget);
            if (enPassentCapturedPiece != attackers)
                enPassentTarget = 0;

            while (myPawns)
            {
                uint64 pawn = getOne(myPawns);

                // pawn push
                uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & (~allPieces);
                if (dst)
                {
                    if (dst & safeSquares)
                    {
                        if (dst & (RANK1 | RANK8))
                            nMoves += 4;
                        else
                            nMoves++;
                    }
                    else
                    {
                        dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush):
                                                   southOne(dst & checkingRankDoublePush) ) & (safeSquares) &(~allPieces);
                        if (dst)
                        {
                            nMoves++;
                        }
                    }
                }

                // captures
                uint64 westCapture = (chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn);
                uint64 eastCapture = (chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn);
                dst = (westCapture | eastCapture) & enemyPieces & safeSquares;
                if (dst)
                {
                    if (dst & (RANK1 | RANK8))
                        nMoves += 4;
                    else
                        nMoves++;
                }

                // en-passent
                dst = (westCapture | eastCapture) & enPassentTarget;
                if (dst)
                {
                    nMoves++;
                }

                myPawns ^= pawn;
            }

            // 2. knight moves
            uint64 myKnights = (knights_ & myPieces);
            while (myKnights)
            {
                uint64 knight = getOne(myKnights);
                uint64 knightMoves = sqKnightAttacks(bitScan(knight)) & safeSquares;
                nMoves += popCount(knightMoves);
                myKnights ^= knight;
            }
            
            // 3. bishop moves
            uint64 bishops = bishopQueens_ & myPieces;
            while (bishops)
            {
                uint64 bishop = getOne(bishops);
                uint64 bishopMoves = bishopAttacks(bishop, ~allPieces) & safeSquares;

                nMoves += popCount(bishopMoves);
                bishops ^= bishop;
            }

            // 4. rook moves
            uint64 rooks = rookQueens_ & myPieces;
            while (rooks)
            {
                uint64 rook = getOne(rooks);
                uint64 rookMoves = rookAttacks(rook, ~allPieces) & safeSquares;

                nMoves += popCount(rookMoves);
                rooks ^= rook;
            }

        }   // end of if single attacker
        else
        {
            // multiple threats => only king moves possible
        }

        return nMoves;
    }



    // count moves for the given board position
    // returns the no of moves generated
    template <uint8 chance>
    CUDA_CALLABLE_MEMBER static uint32 countMoves (QuadBitBoard *pos, GameState *gs)
    {
        uint32 nMoves = 0;

        DERIVE_PIECE_BITBOARDS(pos);

        uint64 myPieces     = (chance == WHITE) ? whitePieces : blackPieces;
        uint64 enemyPieces  = (chance == WHITE) ? blackPieces : whitePieces;

        uint64 enemyBishops = bishopQueens & enemyPieces;
        uint64 enemyRooks   = rookQueens & enemyPieces;

        uint64 myKing     = kings & myPieces;
        uint8  kingIndex  = bitScan(myKing);
        uint64 emptySquares = ~allPieces;

        uint64 pinned     = findPinnedPieces(myKing, myPieces, enemyBishops, enemyRooks, allPieces, kingIndex);

        uint64 threatened = findAttackedSquares(emptySquares, enemyBishops, enemyRooks, allPawns & enemyPieces,
                                                knights & enemyPieces, kings & enemyPieces,
                                                myKing, !chance);

        // king is in check
        if (threatened & myKing)
        {
            return countMovesOutOfCheck<chance>(pos, gs, allPawns, allPieces, myPieces, enemyPieces,
                                                              pinned, threatened, kingIndex);
        }

        uint64 myPawns = allPawns & myPieces;

        // 0. generate en-passent moves first
        uint64 enPassentTarget = 0;
        if (gs->enPassent)
        {
            if (chance == BLACK)
            {
                enPassentTarget = BIT(gs->enPassent - 1) << (8 * 2);
            }
            else
            {
                enPassentTarget = BIT(gs->enPassent - 1) << (8 * 5);
            }
        }

        if (enPassentTarget)
        {
            uint64 enPassentCapturedPiece = (chance == WHITE) ? southOne(enPassentTarget) : northOne(enPassentTarget);

            uint64 epSources = (eastOne(enPassentCapturedPiece) | westOne(enPassentCapturedPiece)) & myPawns;

            while (epSources)
            {
                uint64 pawn = getOne(epSources);
                if (pawn & pinned)
                {
                    uint64 line = sqsInLine(bitScan(pawn), kingIndex);
                    if (enPassentTarget & line)
                    {
                        nMoves++;
                    }
                }
                else
                {
                    // Check if removing both pawns reveals a rook attack on king along the rank.
                    // Use magic rook lookup from king's square with modified occupancy
                    // instead of Kogge-Stone east/west attacks (~14 ALU ops → 1 magic lookup).
                    // Mask to king's rank only — vertical attacks are irrelevant here.
                    uint64 modifiedOcc = allPieces ^ enPassentCapturedPiece ^ pawn;
                    uint64 rankMask = RANK1 << ((kingIndex >> 3) * 8);
                    uint64 causesCheck = rookAttacks(myKing, ~modifiedOcc) & enemyRooks & rankMask;
                    if (!causesCheck)
                    {
                        nMoves++;
                    }
                }
                epSources ^= pawn;
            }
        }

        // 1. pawn moves

        // checking rank for pawn double pushes
        uint64 checkingRankDoublePush = RANK3 << (chance * 24);           // rank 3 or rank 6

        // first deal with pinned pawns
        uint64 pinnedPawns = myPawns & pinned;

        while (pinnedPawns)
        {
            uint64 pawn = getOne(pinnedPawns);
            uint8 pawnIndex = bitScan(pawn);    // same as bitscan on pinnedPawns

            // the direction of the pin (mask containing all squares in the line joining the king and the current piece)
            uint64 line = sqsInLine(pawnIndex, kingIndex);

            // pawn push
            uint64 dst = ((chance == WHITE) ? northOne(pawn) : southOne(pawn)) & line & emptySquares;
            if (dst)
            {
                nMoves++;

                // double push (only possible if single push was possible)
                dst = ((chance == WHITE) ? northOne(dst & checkingRankDoublePush):
                                           southOne(dst & checkingRankDoublePush) ) & emptySquares;
                if (dst) 
                {
                    nMoves++;
                }
            }

            // captures
            // (either of them will be valid - if at all)
            dst  = ((chance == WHITE) ? northWestOne(pawn) : southWestOne(pawn)) & line;
            dst |= ((chance == WHITE) ? northEastOne(pawn) : southEastOne(pawn)) & line;
            
            if (dst & enemyPieces) 
            {
                if (dst & (RANK1 | RANK8))
                    nMoves += 4;    // promotion
                else
                    nMoves++;
            }

            pinnedPawns ^= pawn;  // same as &= ~pawn (but only when we know that the first set contain the element we want to clear)
        }

        myPawns = myPawns & ~pinned;

        // pawn push (accumulate all promotions to save POPCs)
        uint64 allPromotions = 0;
        uint64 dsts = ((chance == WHITE) ? northOne(myPawns) : southOne(myPawns)) & emptySquares;
        uint64 promos = dsts & (RANK1 | RANK8);
        allPromotions = promos;
        nMoves += popCount(dsts ^ promos);  // non-promotion pushes

        // double push (never promotions — rank 4/5 destinations)
        dsts = ((chance == WHITE) ? northOne(dsts & checkingRankDoublePush):
                                    southOne(dsts & checkingRankDoublePush) ) & emptySquares;
        nMoves += popCount(dsts);

        // captures
        dsts = ((chance == WHITE) ? northWestOne(myPawns) : southWestOne(myPawns)) & enemyPieces;
        promos = dsts & (RANK1 | RANK8);
        allPromotions |= promos;
        nMoves += popCount(dsts ^ promos);  // non-promotion captures

        dsts = ((chance == WHITE) ? northEastOne(myPawns) : southEastOne(myPawns)) & enemyPieces;
        promos = dsts & (RANK1 | RANK8);
        allPromotions |= promos;
        nMoves += popCount(dsts ^ promos);  // non-promotion captures

        // all promotions: 4 moves each (N/B/R/Q)
        nMoves += 4 * popCount(allPromotions);

        // generate castling moves
        if (chance == WHITE)
        {
            if ((gs->whiteCastle & CASTLE_FLAG_KING_SIDE) &&
                !(F1G1 & allPieces) && !(F1G1 & threatened))
                nMoves++;
            if ((gs->whiteCastle & CASTLE_FLAG_QUEEN_SIDE) &&
                !(B1D1 & allPieces) && !(C1D1 & threatened))
                nMoves++;
        }
        else
        {
            if ((gs->blackCastle & CASTLE_FLAG_KING_SIDE) &&
                !(F8G8 & allPieces) && !(F8G8 & threatened))
                nMoves++;
            if ((gs->blackCastle & CASTLE_FLAG_QUEEN_SIDE) &&
                !(B8D8 & allPieces) && !(C8D8 & threatened))
                nMoves++;
        }

        // generate king moves
        uint64 kingMoves = sqKingAttacks(kingIndex);

        kingMoves &= ~(threatened | myPieces);
        nMoves += popCount(kingMoves);

        // generate knight moves (only non-pinned knights can move)
        uint64 myKnights = (knights & myPieces) & ~pinned;
        while (myKnights)
        {
            uint64 knight = getOne(myKnights);
            uint64 knightMoves = sqKnightAttacks(bitScan(knight)) & ~myPieces;
            nMoves += popCount(knightMoves);
            myKnights ^= knight;
        }


        // generate bishop (and queen) moves
        uint64 myBishops = bishopQueens & myPieces;

        // first deal with pinned bishops
        uint64 bishops = myBishops & pinned;
        while (bishops)
        {
            uint64 bishop = getOne(bishops);
            uint64 bishopMoves = bishopAttacks(bishop, emptySquares) & ~myPieces;
            bishopMoves &= sqsInLine(bitScan(bishop), kingIndex);

            nMoves += popCount(bishopMoves);
            bishops ^= bishop;
        }

        // remaining bishops/queens
        bishops = myBishops & ~pinned;
        while (bishops)
        {
            uint64 bishop = getOne(bishops);
            uint64 bishopMoves = bishopAttacks(bishop, emptySquares) & ~myPieces;

            nMoves += popCount(bishopMoves);
            bishops ^= bishop;
        }

        // rook/queen moves
        uint64 myRooks = rookQueens & myPieces;

        // first deal with pinned rooks
        uint64 rooks = myRooks & pinned;
        while (rooks)
        {
            uint64 rook = getOne(rooks);
            uint64 rookMoves = rookAttacks(rook, emptySquares) & ~myPieces;
            rookMoves &= sqsInLine(bitScan(rook), kingIndex);    // pined sliding pieces can move only along the line

            nMoves += popCount(rookMoves);
            rooks ^= rook;
        }

        // remaining rooks/queens
        rooks = myRooks & ~pinned;
        while (rooks)
        {
            uint64 rook = getOne(rooks);
            uint64 rookMoves = rookAttacks(rook, emptySquares) & ~myPieces;

            nMoves += popCount(rookMoves);
            rooks ^= rook;
        }

        return nMoves;
    }

    template<uint8 chance>
    CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static void makeMove (QuadBitBoard *pos, GameState *gs, CMove move)
    {
        uint64 src = BIT(move.getFrom());
        uint64 dst = BIT(move.getTo());

        // figure out the source piece from quad encoding
        // bb[1]=piece bit 0, bb[2]=piece bit 1, bb[3]=piece bit 2
        // PAWN=1(001), KNIGHT=2(010), BISHOP=3(011), ROOK=4(100), QUEEN=5(101), KING=6(110)
        uint32 piece = 0;
        if (pos->bb[1] & src) piece |= 1;
        if (pos->bb[2] & src) piece |= 2;
        if (pos->bb[3] & src) piece |= 4;

        // promote the pawn (if this was promotion move)
        // Branchless: promotions have bit 3 set (flags >= 8), piece type in bits [1:0]+2
        if (move.getFlags() & CM_FLAG_PROMOTION)
            piece = (move.getFlags() & 3) + KNIGHT;

        // clear source and destination from all bitboards (4 ops instead of 10)
        uint64 clearMask = ~(src | dst);
        pos->bb[0] &= clearMask;
        pos->bb[1] &= clearMask;
        pos->bb[2] &= clearMask;
        pos->bb[3] &= clearMask;

        // place piece at destination
        if (chance == BLACK) pos->bb[0] |= dst;   // color bit
        if (piece & 1) pos->bb[1] |= dst;
        if (piece & 2) pos->bb[2] |= dst;
        if (piece & 4) pos->bb[3] |= dst;

        // en-passent capture: clear the captured pawn
        if (move.getFlags() == CM_FLAG_EP_CAPTURE)
        {
            uint64 epCapture = (chance == WHITE) ? southOne(dst) : northOne(dst);
            // captured pawn: bb[0] set if enemy (always true), bb[1] set (pawn bit)
            // bb[2] and bb[3] guaranteed 0 for pawn
            pos->bb[0] &= ~epCapture;
            pos->bb[1] &= ~epCapture;
        }

        // castling: move the rook
        if (chance == WHITE)
        {
            if (move.getFlags() == CM_FLAG_KING_CASTLE)
            {
                // white king side castle: rook H1->F1, white rook = bb[3] only
                pos->bb[3] = (pos->bb[3] ^ BIT(H1)) | BIT(F1);
            }
            else if (move.getFlags() == CM_FLAG_QUEEN_CASTLE)
            {
                // white queen side castle: rook A1->D1
                pos->bb[3] = (pos->bb[3] ^ BIT(A1)) | BIT(D1);
            }
        }
        else
        {
            if (move.getFlags() == CM_FLAG_KING_CASTLE)
            {
                // black king side castle: rook H8->F8, black rook = bb[3] + bb[0]
                pos->bb[3] = (pos->bb[3] ^ BIT(H8)) | BIT(F8);
                pos->bb[0] = (pos->bb[0] ^ BIT(H8)) | BIT(F8);
            }
            else if (move.getFlags() == CM_FLAG_QUEEN_CASTLE)
            {
                // black queen side castle: rook A8->D8
                pos->bb[3] = (pos->bb[3] ^ BIT(A8)) | BIT(D8);
                pos->bb[0] = (pos->bb[0] ^ BIT(A8)) | BIT(D8);
            }
        }

        // update game state
        gs->chance = !chance;
        gs->enPassent = 0;

#ifdef __CUDA_ARCH__
        // Castle flag update via LUT: handles king moves, rook moves, and captures on rook squares
        // in a single branchless sequence (2 loads + 1 OR + 2 AND-NOT into bitfields)
        {
            uint8 castleClear = __ldg(&g_castleClear[move.getFrom()]) | __ldg(&g_castleClear[move.getTo()]);
            gs->whiteCastle &= ~castleClear;
            gs->blackCastle &= ~(castleClear >> 2);
        }
#else
        // CPU path: original logic
        updateCastleFlag(gs, dst, chance);
        if (piece == KING)
        {
            if (chance == WHITE)
                gs->whiteCastle = 0;
            else
                gs->blackCastle = 0;
        }
        if (piece == ROOK)
        {
            updateCastleFlag(gs, src, !chance);
        }
#endif

        if (move.getFlags() == CM_FLAG_DOUBLE_PAWN_PUSH)
        {
            // Always set the EP flag — countMoves will check if any EP source
            // pawns actually exist. This avoids re-deriving piece bitboards
            // (~25 SASS instructions) in makeMove's predicated double-push path.
            gs->enPassent = (move.getFrom() & 7) + 1;
        }
    }

};

// Free function wrapper for generateMoves (dispatches on template chance)
CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE uint32 generateMoves(QuadBitBoard *pos, GameState *gs, uint8 color, CMove *genMoves)
{
    if (color == BLACK)
    {
        return MoveGeneratorBitboard::generateMoves<BLACK>(pos, gs, genMoves);
    }
    else
    {
        return MoveGeneratorBitboard::generateMoves<WHITE>(pos, gs, genMoves);
    }
}


