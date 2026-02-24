#pragma once

#include "chess.h"
#include "zobrist.h"

// -------------------------------------------------------------------------
// Transposition table entry (16 bytes, uniform for all depths)
//
// XOR lockless scheme (Hyatt/Crafty):
//   Store: verification = hash.hi ^ count
//   Probe: if ((verification ^ count) == hash.hi) → hit
//   If a concurrent write corrupts the entry, the XOR check fails
//   with overwhelming probability → treated as miss. Always correct.
// -------------------------------------------------------------------------

struct TTEntry
{
    uint64 verification;   // hash.hi XOR count
    uint64 count;          // perft value
};

CT_ASSERT(sizeof(TTEntry) == 16);

// Per-depth transposition table descriptor
struct TTTable
{
    TTEntry *entries;       // pointer to entry array (device or host)
    uint64 mask;            // indexMask = numEntries - 1 (size is power of 2)
};

// -------------------------------------------------------------------------
// Device-side TT probe/store (used in GPU kernels)
// -------------------------------------------------------------------------

#ifdef __CUDACC__

__device__ __forceinline__ bool ttProbe(const TTTable &tt, Hash128 hash, uint64 *outCount)
{
    if (!tt.entries) return false;
    uint64 idx = hash.lo & tt.mask;

    // 128-bit atomic load via PTX to read both fields consistently
    uint64 storedVerif, storedCount;
    const TTEntry *ptr = &tt.entries[idx];
    asm volatile("ld.global.v2.u64 {%0,%1}, [%2];"
        : "=l"(storedVerif), "=l"(storedCount)
        : "l"(ptr));

    // Match the reference's lockless scheme: XOR both hash parts with count.
    // verification = (hash.hi ^ hash.lo) ^ count. Probe checks against (hash.hi ^ hash.lo).
    uint64 expectedKey = hash.hi ^ hash.lo;
    if ((storedVerif ^ storedCount) == expectedKey)
    {
        *outCount = storedCount;
        return true;
    }
    return false;
}

__device__ __forceinline__ void ttStore(const TTTable &tt, Hash128 hash, uint64 count)
{
    if (!tt.entries) return;
    uint64 idx = hash.lo & tt.mask;

    // Match reference: XOR both hash.hi and hash.lo into verification
    uint64 verification = (hash.hi ^ hash.lo) ^ count;
    TTEntry *ptr = &tt.entries[idx];
    asm volatile("st.global.v2.u64 [%0], {%1,%2};"
        :: "l"(ptr), "l"(verification), "l"(count));
}

#endif // __CUDACC__

// -------------------------------------------------------------------------
// Host-side TT probe/store (used in CPU perft path)
// -------------------------------------------------------------------------

inline bool ttProbeHost(const TTTable &tt, Hash128 hash, uint64 *outCount)
{
    if (!tt.entries) return false;
    uint64 idx = hash.lo & tt.mask;
    uint64 storedCount = tt.entries[idx].count;
    uint64 storedVerif = tt.entries[idx].verification;
    uint64 expectedKey = hash.hi ^ hash.lo;
    if ((storedVerif ^ storedCount) == expectedKey)
    {
        *outCount = storedCount;
        return true;
    }
    return false;
}

inline void ttStoreHost(const TTTable &tt, Hash128 hash, uint64 count)
{
    if (!tt.entries) return;
    uint64 idx = hash.lo & tt.mask;
    tt.entries[idx].count = count;
    tt.entries[idx].verification = (hash.hi ^ hash.lo) ^ count;
}

// -------------------------------------------------------------------------
// Lossless chained hash table for host CPU (launch depth).
// No entry is ever evicted — guarantees every GPU BFS result is reusable.
// Chain entries are allocated from a flat pool (bump allocator).
// Thread-safe store via InterlockedIncrement + CAS for chain prepend.
// -------------------------------------------------------------------------

struct LosslessEntry
{
    uint64 hashKey;     // hash.hi ^ hash.lo for verification
    uint64 count;       // perft value
    int32_t next;       // next entry index (-1 = end of chain)
    int32_t _pad;
};

CT_ASSERT(sizeof(LosslessEntry) == 24);

struct LosslessTT
{
    int32_t *buckets;           // bucket heads (-1 = empty)
    LosslessEntry *pool;        // entry pool
    uint64 bucketMask;          // numBuckets - 1
    int32_t nextFree;           // next free entry
    int32_t poolCapacity;       // max entries in pool
};

inline bool losslessProbe(const LosslessTT &tt, Hash128 hash, uint64 *outCount)
{
    if (!tt.buckets) return false;
    uint64 bucket = hash.lo & tt.bucketMask;
    uint64 key = hash.hi ^ hash.lo;
    int32_t idx = tt.buckets[bucket];
    while (idx >= 0)
    {
        if (tt.pool[idx].hashKey == key)
        {
            *outCount = tt.pool[idx].count;
            return true;
        }
        idx = tt.pool[idx].next;
    }
    return false;
}

inline void losslessStore(LosslessTT &tt, Hash128 hash, uint64 count)
{
    if (!tt.buckets) return;
    // Atomic bump allocator — safe for concurrent stores from multiple threads
#ifdef _MSC_VER
    long newIdx = _InterlockedIncrement((volatile long *)&tt.nextFree) - 1;
#else
    int32_t newIdx = __atomic_fetch_add(&tt.nextFree, 1, __ATOMIC_SEQ_CST);
#endif
    if (newIdx >= tt.poolCapacity) return;
    uint64 bucket = hash.lo & tt.bucketMask;
    tt.pool[newIdx].hashKey = hash.hi ^ hash.lo;
    tt.pool[newIdx].count = count;
    // CAS loop to prepend to bucket chain
#ifdef _MSC_VER
    long oldHead;
    do {
        oldHead = *(volatile long *)&tt.buckets[bucket];
        tt.pool[newIdx].next = (int32_t)oldHead;
    } while (_InterlockedCompareExchange((volatile long *)&tt.buckets[bucket], (long)newIdx, oldHead) != oldHead);
#else
    int32_t oldHead;
    do {
        oldHead = __atomic_load_n(&tt.buckets[bucket], __ATOMIC_SEQ_CST);
        tt.pool[newIdx].next = oldHead;
    } while (!__atomic_compare_exchange_n(&tt.buckets[bucket], &oldHead, newIdx, true, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
#endif
}

// -------------------------------------------------------------------------
// TT management (allocation, deallocation)
// -------------------------------------------------------------------------

#define MAX_TT_DEPTH 32

// Global TT arrays (defined in launcher.cu)
extern TTTable deviceTTs[MAX_TT_DEPTH];   // GPU memory, indexed by remaining depth
extern LosslessTT hostLosslessTTs[MAX_TT_DEPTH];  // lossless chained tables, indexed by remaining depth

// Initialize TTs based on launch depth and max depth
void initTT(int launchDepth, int maxLaunchDepth, int maxDepth, float branchingFactor);

// Free all TT memory
void freeTT();
