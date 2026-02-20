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
    uint64 storedCount = __ldg(&tt.entries[idx].count);
    uint64 storedVerif = __ldg(&tt.entries[idx].verification);
    if ((storedVerif ^ storedCount) == hash.hi)
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
    tt.entries[idx].count = count;
    tt.entries[idx].verification = hash.hi ^ count;
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
    if ((storedVerif ^ storedCount) == hash.hi)
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
    tt.entries[idx].verification = hash.hi ^ count;
}

// -------------------------------------------------------------------------
// TT management (allocation, deallocation)
// -------------------------------------------------------------------------

#define MAX_TT_DEPTH 32

// Global TT arrays (defined in launcher.cu)
extern TTTable deviceTTs[MAX_TT_DEPTH];   // GPU memory, indexed by remaining depth
extern TTTable hostTTs[MAX_TT_DEPTH];     // pinned host memory, indexed by remaining depth

// Initialize TTs based on launch depth and max depth
void initTT(int launchDepth, int maxDepth);

// Free all TT memory
void freeTT();
