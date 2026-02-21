# perft_gpu Development Notes

## Git
- Push to `https://github.com/ankan-ban/perft_gpu_2026` (NOT perft_gpu)
- Never add `Co-Authored-By` lines to commit messages
- Always do clean rebuilds (`--clean-first`) after editing headers - NVCC doesn't reliably track transitive header dependencies

## Branch Status
- **`transposition-tables`** (main working branch, **dirty — uncommitted OOM fix + logging work**):
  - Dedup generation counter (+0.7%, eliminates memset)
  - Lossless CPU hash at launch depth (neutral, kept for deep perft)
  - HOST_TT_BUDGET_MB = 65536
  - **OOM fallback fix**: all BFS OOM paths signal g_lastBfsOom, caller falls back to CPU recursion
  - **Mid-iteration LD decrease**: g_effectiveLD drops on first OOM, prevents cascading corruption
  - **Dynamic LD increase DISABLED**: root perft(N) memory doesn't predict worst-case perft(N+1)
  - **Device TTs**: allocated for depths 3 through LD-1 (TT[4] at 1024M, others at 512M for LD=8)
  - **All host TTs lossless**: LosslessTT at all CPU depths, proportional sizing by branchingFactor^(maxDepth-d), 4M entry floor
  - **VERBOSE_LOGGING switch** (switches.h, off by default): call size/time histograms, per-BFS-level stats, progress reporting
  - **WIP**: moving ALL logging behind VERBOSE_LOGGING — basic call stats + TT stats still print when off. Need to finish wrapping them. Code does NOT currently build cleanly with VERBOSE_LOGGING=0 due to partially-complete refactor.
  - **Baseline at LD=8**: perft 10 = 3.55s, 19,554B nps | perft 11 = 43.5s, 48,204B nps | perft 12 = 618s, 101,767B nps | perft 13 = 9503s, 208,476B nps
- **`async-streams`** (WIP side branch, branched from transposition-tables):
  - All of the above PLUS CUDA streams + background thread for GPU BFS overlap
  - 2 streams, split buffer (8GB each), pinned readback memory, stream-local sync
  - Currently regresses ~3.7% due to per-call std::thread create/join overhead
  - **Next session TODO**: thread pool (create once at init), separate full-size 16GB buffers,
    profile with Nsight Systems to identify regression source

## Architecture
- Host-driven BFS with CUDA kernels per phase per level
- QuadBitBoard (4x uint64 = 32 bytes) + GameState (1 byte packed bitfields: whiteCastle:2 + blackCastle:2 + enPassent:4)
- Color (chance) passed explicitly — not stored in GameState
- Bump allocator from single pre-allocated GPU buffer (no per-level cudaMalloc)
- CUB InclusiveSum (in-place), merge-path interval expand, no CDP
- Fused 2-level leaf kernel: last 2 levels handled per-thread (no BFS expansion needed)
- Dynamic launch depth estimation accounts for 2-level leaf memory savings
- Combined magic entry struct: mask+factor co-located for single cache-line access
- Pure CPU path available via `-cpu` flag (template-specialized, countMoves at leaf)
- **Transposition tables**: 128-bit Zobrist hash, separate TT per depth, XOR lockless 16-byte entries
  - Device TTs (depths 3..launchDepth-1): probed in GPU BFS downsweep, stored via upsweep
  - Host TTs (depths launchDepth..maxDepth): probed/stored in CPU recursion
  - Upsweep: after leaf kernel, reduces per-position counts back through BFS levels
  - Leaf TT: fused kernel probes/stores via HASH_IN_LEAF_KERNEL switch

## File Structure
- `perft.cu` — main entry point, CLI parsing, `-cpu` flag handling, LD adjustment
- `launcher.cu` / `launcher.h` — GPU init, launch depth estimation, GPU+CPU perft launchers, CPU perft, TT allocation, OOM fallback, diagnostics
- `perft_kernels.cu` — GPU kernels, device helpers, host-driven BFS, upsweep, move generator init, OOM signaling
- `MoveGeneratorBitboard.h` — move generation logic (templates + magics), ~1950 lines
- `chess.h` — core data structures (QuadBitBoard, GameState, CMove, magic entries)
- `switches.h` — compile-time flags (BLOCK_SIZE, MIN_BLOCKS_PER_MP, USE_TT, HASH_IN_LEAF_KERNEL, VERBOSE_LOGGING, etc.)
- `zobrist.h` / `zobrist.cpp` — 128-bit Zobrist hashing (Hash128, ZobristRandoms, computeHash, updateHashAfterMove)
- `tt.h` — transposition table (TTEntry, TTTable, probe/store functions)
- `uint128.h` — simple 128-bit integer for deep perft accumulation
- `utils.h` / `util.cpp` — FEN parsing, board display, Timer class
- `GlobalVars.cpp` — CPU bitboard lookup tables + hardcoded magic entries
- `Magics.cpp` — magic number discovery routines

## Performance
- RTX 6000 Pro Blackwell (with TT, LD=8, lossless host TTs + TT[4] 2x):
  - Startpos perft 9: 0.29s | perft 10: 3.55s, 19,554B nps
  - perft 11: 43.5s, 48,204B nps | perft 12: 618s, 101,767B nps
  - perft 13: 9503s (2h 38m), 208,476B nps
  - GPU at 590-600W, 88-89°C, 93-97% util throughout perft 13 (confirmed via nvidia-smi)
- RTX 6000 Pro Blackwell (no TT baseline): Startpos perft 9 ~1027B nps (2.38s)
- RTX 4090: Startpos perft 9 ~729B nps (3.34s), Position 2 perft 7 ~1103B nps (0.34s)

## Benchmarking
For TT performance benchmarking, only test **startpos perft 10 and perft 11** — shallower depths are too fast to measure meaningfully.
```
./build/Release/perft_gpu.exe "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" 10
./build/Release/perft_gpu.exe "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" 11
```
- **Run benchmarks SERIALLY** — never run multiple GPU benchmarks in parallel (corrupts measurements)
- Always do a clean rebuild (`--clean-first`) before benchmarking if any code changed
- Baseline (LD=8, lossless host TTs + TT[4] 2x): perft 10 ~19,554B nps (3.55s), perft 11 ~48,204B nps (43.5s), perft 12 ~101,767B nps (618s)

## OOM Handling (FIXED)

### Previous Bug
- BFS OOM returned 0 or partial results (perft_2level_from_board_kernel fallback was wrong)
- Wrong values stored in TTs, poisoning ALL future lookups
- Cascading corruption: wrong TT entry → less pruning → larger BFS → more OOMs
- Perft 12 gave 37.5T instead of correct 62.9T (40% undercount)

### Fix
- All BFS OOM paths set `g_lastBfsOom = true` + `cudaDeviceSynchronize()` before returning 0
- Fixed OOM paths: d_curBoards/d_curStates/d_curHashes/d_moveCounts/d_ttHitCounts, d_newIndices/d_newMoves, d_leafCounts, d_parentCounts, partition overflow
- Caller (`perft_cpu_recurse`) detects OOM via `g_lastBfsOom`, falls back to CPU recursion
- CPU recursion recurses one level deeper, sub-calls retry GPU at depth-1 (smaller BFS, fits in memory)
- `g_effectiveLD` decreased mid-iteration on first OOM — prevents all future OOMs in that iteration
- Removed the old `perft_2level_from_board_kernel` OOM fallback (was returning wrong results)
- Removed noisy "OOM: need X bytes..." printf from GpuBumpAllocator (confusing, not actionable)

## Launch Depth (LD=8, no dynamic increase)

### Why LD increase is disabled
- Root perft(N) memory does NOT predict worst-case perft(N+1) from arbitrary positions
- Perft(8) from root: 793MB (4.8% of 16GB) — looks safe to increase
- But perft(9) from some non-root positions: >16GB → OOM
- LD=9 at perft 12: **~75% slower** than LD=8 due to OOMs + TT dilution (9 TTs at 256M each vs 5 at 512M)

### Current behavior
- LD fixed at memory-estimated initial value (8 for startpos)
- LD can still DECREASE on OOM mid-iteration (via g_effectiveLD)
- Manual LD override via 3rd CLI argument still works
- Device TTs allocated for depths 3 through LD-1 only (5 TTs at 512M each for LD=8)
- Post-iteration: only decreases LD if OOM fallbacks occurred, never increases

## Verbose Logging (VERBOSE_LOGGING switch, WIP)

### What's behind the switch (when VERBOSE_LOGGING=1)
- GPU BFS call stats: count, total/avg/min/max time, OOM fallback count
- Host TT probe stats: hits/misses/hit rate
- Peak BFS memory tracking
- Call size histogram (by leaf position count: 0, 1-100, ..., 1M+)
- Call time histogram (<0.01ms through 100ms+)
- Per-BFS-level position count tracking (avg expansion ratios)
- Progress reporting every 10s (calls/sec, host TT hit%, avg call time, OOM count)
- TT fill level stats (lossless + lossy occupancy)
- Per-GPU-call leaf count + BFS level counts in perft_kernels.cu

### WIP status
- **Code does NOT build with VERBOSE_LOGGING=0** — partially refactored
- Need to finish wrapping: some stats variables (g_gpuCallCount, etc.) and their collection code are only partially behind the switch
- All the verbose histogram/progress code IS properly wrapped
- OOM handling (g_lastBfsOom, g_oomFallbackCount, g_effectiveLD) stays unconditional — not logging

### Perft 12 diagnostics (from VERBOSE_LOGGING=1 run)
- 101,239 GPU calls, avg 6.5ms, min 0.015ms, max 164ms
- 29.4% of calls had 0 leaf positions (all TT hits, ~0.17ms each, ~5s total = 0.8%)
- 57.2% had 1M+ leaf positions (avg 11ms, bulk of compute time)
- Host TT hit rate: 38.5% (lossless at LD=8)
- GPU at 580-605W throughout (confirmed via nvidia-smi every 5s for full 11 min run)
- 0 OOM fallbacks, peak BFS memory 18.3%
- BFS expansion ratios: 12-18x per level (device TTs provide moderate pruning)

## Kernel Bottleneck Analysis (Nsight Compute)

**NOTE**: These profiles are from the pre-TT codebase. The TT adds hash computation (~8 __ldg per move) + TT probe to the leaf kernel and makemove kernel. Re-profiling needed to check register count, occupancy, and pipe utilization changes.

### fused_2level_leaf_kernel — Blackwell (504K blocks, pos2 perft 7, auto depth 7) [PRE-TT]
- Compute (SM) throughput: 72.6%, Memory throughput: 45.6%
- ALU pipe: 52.4% utilization, 191B instructions (71% of total)
- IPC: 2.90 active, issue slots busy 72.2%
- 56 registers with __launch_bounds__(384, 3), 75% theoretical / 65.4% achieved occupancy
- L1 global loads: 65.9B sectors, 98.7% hit rate
- L1 local loads: 1.07B sectors, 40.9% hit rate (CMove[218] array)
- L1 local stores: 4.9B sectors (CMove array + register spills)
- Warp stalls: not_selected 2.51, wait 2.27, long_scoreboard 1.82, math_throttle 1.56
- Active threads/warp: 25.0/32 (78%), branch efficiency 88.3%
- Uncoalesced global access: 15% excess sectors (scatter pattern from indices[])
- Local memory waste: 0.5 bytes/sector on loads, 1.1 bytes/sector on stores (CMove is uint16)

### fused_2level_leaf_kernel — Blackwell WITH TT (504K blocks, pos2 perft 7, auto depth 7)
- Compute (SM) throughput: 66.8% (was 72.6%, -5.8pp), Memory throughput: 35.3% (was 45.6%, -10.3pp)
- ALU pipe: 48.0% (was 52.4%, -4.4pp), 222B instructions (was 191B, +16%)
- 56 registers — unchanged despite hash computation, occupancy preserved
- L1 global loads: 35.6B sectors (was 65.9B, -46%) — TT hits skip subtrees
- L1 local loads: 706M sectors (was 1.07B, -34%) — fewer CMove arrays for TT-pruned positions
- L1 local stores: 2.97B sectors (was 4.9B, -39%)
- Stall analysis: long_scoreboard and wait dominate (similar pattern to pre-TT)
- **Key takeaway**: 56 regs preserved. TT adds ~16% instructions but saves ~46% global traffic.
  Net negative for pos2@depth7 (low TT hit rate), but net positive for startpos@depth9 (high hit rate)

### RTX 4090 comparison
- ALU heavy pipe: 85% SM utilization (vs 52.4% on Blackwell — wider pipes spread load)

### BFS overhead (with TT)
- Nsight Systems trace: leaf kernel = 88.7% of GPU time (was 99.4% pre-TT)
- TT prunes subtrees → BFS/upsweep/dedup become relatively more visible
- Upsweep 4.2%, makemove 2.7%, dedup 1.7%, generate 1.5%, interval expand 0.9%
- GPU idle gaps: 293ms (6.5%) — CPU recursion between GPU BFS calls

## Transposition Table Design

### Data Structures
- `Hash128`: two uint64 (lo, hi). lo indexes TT (masked), hi verifies (XOR'd with count)
- `TTEntry`: 16 bytes uniform `{ uint64 verification; uint64 count; }`. XOR lockless: `verification = hash.hi ^ count`, probe checks `(verification ^ count) == hash.hi`
- `TTTable`: `{ TTEntry *entries; uint64 mask; }` — mask = numEntries - 1 (power of 2)
- `ZobristRandoms`: `pieces[2][6][64]`, `castling[2][2]`, `enPassant[8]`, `sideToMove` — two sets for 128-bit hash

### Hash Computation
- `computeHash()`: from-scratch, iterates all 64 squares. Used for root position only
- `updateHashAfterMove()`: incremental, ~8 `__ldg` Zobrist lookups per move (LSU pipe)
- Caller pattern: derive srcPiece/capPiece from QBB BEFORE makeMove, read new castle/EP AFTER makeMove
- makeMove itself is NEVER modified — hash update is a separate function
- Castle rights delta: `oldCastleRaw ^ newCastleRaw`, only XOR changed bits
- Zobrist tables generated by splitmix64 with fixed seeds (deterministic across runs)
- GPU: `__device__` globals with `__ldg` access. CPU: regular globals. `ZOB1()`/`ZOB2()` macros dispatch

### TT Memory Layout
- Separate TT per remaining depth (no depth field in entries)
- Device TTs: depths 3 through launchDepth-1 (GPU `cudaMalloc`, separate from BFS buffer)
- Host TTs: depths launchDepth through maxDepth (`malloc`, not pinned)
- Budget: `DEVICE_TT_BUDGET_MB` / `HOST_TT_BUDGET_MB` divided equally among TTs per category
- TTs persist across depth iterations (perft(d) from a position is depth-invariant)

### GPU BFS Integration
- **Downsweep**: `makemove_and_count_moves_kernel` loads parent hash, computes child hash via `updateHashAfterMove`, probes device TT. TT hits: moveCount=0 (no expansion), ttHitCounts[i]=count. TT misses: normal countMoves
- **Hash flow**: `Hash128 *d_curHashes` allocated alongside boards/states at each BFS level. `d_prevHashes` advances each iteration. Root hash computed on CPU and copied to GPU
- **Per-level state saved** in `BFSLevelSave`: indicesToParent, hashes, ttHitCounts, count, remainingDepth
- **Upsweep** (after leaf): `reduce_by_parent` kernel (atomicAdd children→parent counts) + `add_tt_hits_and_store` kernel (add TT hit counts + store in device TT). Runs from leaf back to root level
- **Result**: comes from upsweep total (sum of root children counts), NOT the leaf global atomic counter (which misses TT-hit subtrees pruned during downsweep)
- **Leaf TT** (`HASH_IN_LEAF_KERNEL`): fused 2-level leaf kernel probes/stores TT at remaining depth 2. Guarded by compile-time switch

### CPU Path Integration
- `perft_cpu_recurse`: probes `hostTTs[depth]` before recursing/launching GPU, stores after
- Most impactful optimization: `hostTTs[launchDepth]` caches entire GPU BFS calls
- `perft_cpu` (template, -cpu mode): same probe/store pattern with host TTs

### Unique Positions per Ply (OEIS A083276, startpos)
Critical for TT sizing — unique positions grow ~9-10x per ply vs ~28-30x for total paths:
| Ply | Unique Positions | Total (perft) | Transposition Ratio |
|-----|-----------------|---------------|---------------------|
| 5 | 822,518 | 4,865,609 | 5.9x |
| 6 | 9,417,681 | 119,060,324 | 12.6x |
| 7 | 96,400,068 | 3,195,901,860 | 33.2x |
| 8 | 988,187,354 | 84,998,978,956 | 86x |
| 9 | 9,183,421,888 | 2,439,530,234,167 | 266x |
| 10 | 85,375,278,064 | 69,352,859,712,417 | 812x |
| 11 | 726,155,461,002 | 2,097,651,003,696,806 | 2,889x |

### GPU BFS Call Scaling (startpos, LD=8, lossless host TTs + TT[4] 2x)
| Perft | GPU Calls | Avg ms | Max ms | Host TT Hit% | Total Time |
|-------|-----------|--------|--------|--------------|------------|
| 9 | 20 | 15.3 | 25.4 | 0% | 0.29s |
| 10 | 400 | 8.9 | 26.4 | 0% | 3.55s |
| 11 | 7,602 | 5.9 | 56.4 | 13.9% | 43.5s |
| 12 | 101,239 | 6.5 | 164.0 | 38.5% | 618s |
| 13 | ~1.3M | ~6.5 | — | ~50%+ | 9,503s |
- GPU calls track OEIS unique positions at abs depth N-8 (remaining depth = LD)
- ~29% of perft 12 calls have 0 leaf positions (all device TT hits, ~0.17ms each)

### Known TT Behavior
- Startpos perft 9: +569% speedup with upsweep TT store (6865B vs 1027B nps)
- Pos2 perft 7: +165% speedup with upsweep TT store (4846B vs 1825B nps)
- TT benefit scales with depth: deeper perft = more transpositions = more TT hits
- The iterative depth loop (perft 1, 2, ..., N) naturally warms TTs for later iterations
- Upsweep TT store is critical: caches per-position results across GPU BFS calls

## Committed Optimizations

### Fused 2-level leaf kernel [+9-17%]
- Each thread: makeMove → generateMoves → loop(makeMove+countMoves)
- Eliminates last BFS level's massive move/index arrays
- Better cache locality: siblings processed together
- Enabled perft 10 startpos (previously OOM)

### Dynamic launch depth [+5-6%]
- Memory budget multiplied by branching factor (2-level leaf saves one expansion)
- Higher launch depth = fewer CPU calls
- Dynamic OOM fallback: CPU recursion retry at depth-1

### __launch_bounds__(384, 3) on leaf kernel [+4-9%]
- Forces 56 registers (from 70), increases occupancy 50% → 75%

### Combined magic entry struct [+3.5%]
- MagicMaskFactor: mask+factor co-located (16 bytes, single cache-line)
- Separate position array (4 bytes per square)
- Reduces L1 traffic vs separate mask + FancyMagicEntry loads
- Controllable via USE_COMBINED_MAGIC_GPU flag

### Branchless promotion in makeMove [+1%]
- `if (flags & 8) piece = (flags & 3) + 2` replaces 4 if-else-if chains

### Simplified double-pawn-push EP check [~1%]
- Always set EP flag for double pushes; countMoves handles the source check
- Eliminates ~25 SASS instructions in predicated path

### Bulk SWAR pawn evasions in countMovesOutOfCheck [+1-2%]
- Replaced per-pawn while loop with bulk shift-and-mask pawn move generation
- Same pattern as countMoves: northOne(myPawns), popCount on masked results
- Eliminates per-pawn branches on a cold path (~4-5% of positions are in check)

### Transposition tables [+569% startpos perft 9, +165% pos2 perft 7]
- 128-bit Zobrist hash (splitmix64, two ZobristRandoms sets), separate updateHashAfterMove()
- 16-byte TTEntry with XOR lockless verification, separate TT per remaining depth
- GPU: hash flow through BFS + TT probe in downsweep + upsweep to store per-position counts
- CPU: probe/store host TTs in recursive path, caches entire GPU BFS calls
- Leaf TT: fused kernel probes/stores at depth 2 (HASH_IN_LEAF_KERNEL switch)
- Upsweep result is authoritative (leaf global atomic misses TT-hit subtrees)
- BFS early-exit fix: when nextLevelCount=0, set currentCount=0 to prevent leaf double-counting

### OOM fallback [correctness fix]
- All BFS OOM paths signal g_lastBfsOom, caller falls back to CPU recursion at depth-1
- Mid-iteration LD decrease (g_effectiveLD) prevents cascading OOMs
- Fixed critical bug: previous OOM handling returned 0, corrupting TTs and causing 40% undercount

### Code cleanup [cleanliness]
- Removed 13 compile-time flags, ~2.6MB unused tables
- 9 active flags in switches.h (BLOCK_SIZE, LIMIT_REGISTER_USE, MIN_BLOCKS_PER_MP, PREALLOCATED_MEMORY_SIZE, USE_COMBINED_MAGIC_GPU, USE_TT, HASH_IN_LEAF_KERNEL, DEVICE/HOST_TT_BUDGET_MB, VERBOSE_LOGGING)

## Rejected Optimizations (tried and measured)

### getOne/bitScan elimination [REJECTED — instability]
- NVCC already optimizes `bitScan(getOne(x))` → `bitScan(x)` internally
- Manual change caused catastrophic instability (some runs 18x slower)

### `uint8 flags` local variable [REJECTED — 18x regression]
- Disrupts register allocation in deeply-inlined makeMove+countMoves chain

### Checkmask/pinmask (Gigantua-style) [REJECTED — 5-9% regression]
- Non-fused: +3.3% instructions, only -6.5% divergence = net negative
- Fused single-pass: still -5.3%. Extra instructions outweigh divergence savings
- Works on CPU (branch prediction), not GPU (predication handles divergence differently)

### QBB sliding attacks [REJECTED — 74% regression]
- Pure ALU, zero tables — catastrophic: overloads saturated ALU heavy pipe
- Magic bitboards essential: offload work to underutilized LSU pipe

### 3-level leaf [REJECTED — 26% regression]
- Two CMove[218] arrays = 1.1KB/thread stack overwhelms L1

### __any_sync warp skips [REJECTED — 3.4% regression]
- VOTE intrinsic overhead exceeds savings from skipping predicated code

### Software popcount [REJECTED — 24% regression]
- 12 regular ALU ops is 4x worse throughput than 1 hardware POPC

### LUT knight/king in findAttackedSquares [REJECTED — 2.3% regression / neutral]
- LSU latency (~30 cycles) exceeds bulk ALU pipeline (1 cycle each, pipelined)

### Multi-position batching [ANALYZED — not worth it]
- Launch overhead is only 0.03% of total time

### Smarter bump allocator [ANALYZED — not worth it]
- Memory isn't the bottleneck (1.3GB used of 16GB)

### Non-bitboard representations [ANALYZED — fundamentally worse]
- Bitboards: 1 POPC counts 64 squares. Alternatives need 8-30 per-square checks
- LOP3.LUT IS the SIMD for bitboards — already optimal
- SIMD video instructions (VADD4/DP4A) operate on packed bytes, not individual bits

### Blackwell v4.u64 / ulonglong4 for combined magic [REJECTED — no gain / -1.9%]
- Single 256-bit load vs multiple smaller loads — no measurable difference or slight regression
- ulonglong4 cast loses __ldg read-only cache hint, __ldg with separate fields already coalesces via L1

### X-ray pin calculator [REJECTED — 7-9% regression]
- Replace sqBishopAttacks/sqRookAttacks LUT + isSingular loop with 4 magic lookups (normal + x-ray)
- Intended to reduce warp divergence from variable-length attacker loop
- 4 extra memory-dependent magic lookups cost far more than the loop they eliminate
- Current approach: 2 cheap LUT lookups + short loop (0-2 iters) with isSingular (x & (x-1), 2 ops)

### Kogge-Stone in findAttackedSquares [REJECTED — 10-21% regression]
- Replace multiBishopAttacks/multiRookAttacks loops with Kogge-Stone parallel prefix
- Same mechanism as QBB rejection: pure ALU overloads saturated ALU heavy pipe
- Even though findAttackedSquares is called only 1× per position (vs ~30× for move gen),
  the ~64 ALU ops still measurably hurt vs a few magic lookups on the underutilized LSU pipe

### Piece-in-CMove (uint32 CMove with piece type) [REJECTED — 14-23% regression]
- Encoded 3-bit piece type in CMove (uint16→uint32) to skip 6-op bitboard probe in makeMove
- Saved ~12 ALU ops/makeMove (piece reconstruction + promotion check) × 31 calls = ~370 ops/thread
- But childMoves[218] doubled from 436B to 872B per thread → doubled L1TEX local memory traffic
- Same mechanism as 3-level leaf: per-thread stack growth overwhelms L1 at this occupancy level
- Startpos perft 9: 538B nps (was 699B), pos2 perft 7: 941B nps (was 1095B)
- Expanded board (32B→64B) analyzed and rejected without testing: maintenance cost ≥ derivation cost

### Iterator generateMoves (forEachMove callback) [REJECTED — 20-39% regression]
- Replaced CMove[218] array with callback: generate one move, immediately makeMove+countMoves
- Eliminated all CMove array traffic but caused catastrophic register spills
- Local loads exploded 20.7x (1.07B → 22.2B sectors): at every callback call site, NVCC must
  spill all live move-generation state (pinned, threatened, bitboards) across makeMove+countMoves
- With 56-register cap, interleaving generation and processing is far worse than two sequential phases
- Same mechanism as 3-level leaf / Piece-in-CMove: per-thread stack growth overwhelms L1

### Batched per-piece generateMoves (CMove[16] buffer) [REJECTED — 3.5-21% regression]
- Generate moves one piece at a time into small buffer, process batch, reuse buffer
- Better than iterator (generation state can die between batches) but common state
  (pinned, threatened, etc.) must stay live across all processBatch calls → still spills
- Check evasion path used CMove[64] fallback (rare, few evasion moves)
- Startpos: 991B nps (-3.5%), pos2: 1440B nps (-21%)

### Cached inversions in DERIVE_PIECE_BITBOARDS [REJECTED — neutral]
- Pre-compute ~b1, ~b2, ~b3 into local variables instead of inline inversions
- LOP3.LUT already fuses `b1 & ~b2 & ~b3` into a single instruction
- Pre-computing inversions PREVENTS LOP3 fusion, splitting 1 op into 2+

### Branchless castling accumulation [REJECTED — neutral]
- Replace `if (cond) nMoves++` with `nMoves += (cond)` in countMoves
- `chance` is a template parameter → zero divergence by construction
- Generates identical SASS — predicated increment is already branchless

### FGMC / Fused Generate-Make-Count [REJECTED — 19-37% regression]
- Re-test of Iterator approach with template-specialized makeMove per piece type
- Confirmed identical failure: ~120 bytes of generation state spills at ~30 countMoves call sites
- Startpos: 839B nps (-19%), pos2: 1152B nps (-37%)
- Template specialization made zero difference vs generic iterator rejection

### Template-specialized makeMove with switch dispatch [REJECTED — 2.1% regression]
- Extract piece type from board, switch-dispatch to makeMoveKnownPiece<chance, pieceType>
- Eliminates ~6 ops piece extraction + ~10-18 ops dead predicated code per call
- But 6-way switch causes warp divergence: all 6 specializations execute, 5/6 predicated off
- Code bloat from 6 makeMove copies hurts instruction cache
- Net: dispatch overhead exceeds dead-code savings

### Bulk pawn generation with reverse derivation in generateMoves [REJECTED — neutral/mixed]
- Replace per-pawn while loop with bulk shifts, iterate destinations, derive source by reverse offset
- Same pattern as bulk countMoves (which helped), but applied to move generation
- Startpos: -1.0%, pos2: +0.9% — mixed results, net neutral
- Asymmetric offsets (northWest=+7, southWest=-9) are error-prone
- Per-pawn loop is already efficient for generateMoves since it needs from/to pairs anyway

### BFS early-exit double-counting bug [FIXED]
- When all positions at a BFS level had TT hits, nextLevelCount=0 caused early loop break
- The break happened BEFORE d_prevBoards was updated and currentCount was reset
- Result: leaf kernel processed stale positions (from previous level) using root indices (all zeros)
- This added phantom perft(2) counts to position 0 on top of its correct TT hit value
- Fix: set currentCount=0 when breaking on nextLevelCount=0, preventing leaf kernel from running
- The ~10K monotonic increase per call matched perft(3) of the GPU BFS call's root position

### Unfused bfsMinLevel=2 with TT+dedup [REJECTED — 52% regression startpos, OOM pos2]
- Changed bfsMinLevel from 3 to 2 to get dedup at one more level
- Replaces fused 2-level leaf with simple 1-level leaf
- Startpos: 3240B nps (was 6760B), pos2: OOM/hung at perft(7)
- Same mechanism as 3-level leaf: massive BFS arrays at level 2 overwhelm memory
- The fused kernel's memory savings are essential — can't replace with BFS expansion

### Block size / blocks-per-SM tuning [CONFIRMED — 384/3 is optimal]
- Tested: 256/3, 256/4, 256/5, 320/3, 320/4, 352/3, 352/4, 384/2, 384/3, 384/4, 512/2, 512/3, 416/3, 448/3
- 512+ and 416+ block sizes fail to build (register budget)
- MIN_BLOCKS_PER_MP=4 catastrophic (-51%): forces ≤42 registers → massive spilling
- MIN_BLOCKS_PER_MP=2 loses 14%: not enough warps to hide latency
- Smaller block sizes (256, 320, 352) consistently worse than 384
- Current 384/3 (56 regs, 75% occupancy) is the sweet spot

### Dynamic LD increase 8→9 [REJECTED — 75% regression at perft 12]
- Root perft(8) uses 4.8% memory → heuristic increased LD to 9
- At LD=9, some perft(9) calls from non-root positions exceed 16GB → OOM
- Device TTs spread across 9 depths (256M each) instead of 5 (512M each) → less effective
- Per-call time 13x larger at LD=9 (120ms vs 9ms) but only 9x fewer calls → net slower
- OOMs corrupted TTs causing cascading performance degradation
- Fix: disabled LD increase entirely, LD can only decrease on OOM

## Key Lessons

### GPU optimization
- NVCC is extremely sensitive to source-level patterns — "optimizing" code can trigger regressions
- Never introduce `uint8` local variables in hot GPU paths
- LOP3.LUT (7341 uses) already fully exploits 3-input boolean fusion — can't improve
- Magic bitboards are essential: trade ALU for LSU, balancing pipe utilization
- Hardware POPC/FLO are irreplaceable even on the bottlenecked heavy pipe
- At 85% ALU heavy utilization, the kernel is near the hardware ceiling

### Methodology
- **Always try experiments even when theory says they won't help**
- Combined magic struct "shouldn't help" (loads already parallel) but gave +3.5%
- Simplified EP check seemed marginal but helped on startpos
- Theory predicts, experiments decide — GPU behavior is hard to model precisely
- The only thing that can beat AI is human willingness to try "stupid" ideas
- Never give up trying — there's always one more thing to optimize

### OOM handling
- **Never return wrong results from GPU OOM** — they poison TTs and cause cascading failures
- Signal OOM to caller, let it retry at smaller depth or fall back to CPU
- Add cudaDeviceSynchronize before OOM returns to prevent stale kernel interference
- Root perft(N) memory does NOT predict worst-case from arbitrary positions — don't use it for LD increase

## Nsight Systems Profile (startpos perft 10, with TT)

### Kernel time breakdown
| Kernel | Calls | Total (ms) | % of GPU |
|--------|-------|-----------|----------|
| fused_2level_leaf_kernel | 425 | 3734.9 | 88.7% |
| makemove_and_count_moves_kernel | 2116 | 114.7 | 2.7% |
| Upsweep (reduce+store+redirects) | 6774 | 177.7 | 4.2% |
| Dedup (write+check) | 4232 | 69.6 | 1.7% |
| generate_moves_kernel | 2116 | 63.5 | 1.5% |
| mp_interval_expand_kernel | 2116 | 36.6 | 0.9% |
| CUB prefix sum | 4232 | 7.9 | 0.2% |
| Memset | 5517 | 78.8 | (overlaps) |

### Key metrics
- ~425 GPU BFS calls per perft(10), each ~9.7ms average
- GPU wall clock 4504ms, kernel time 4210ms, GPU idle 293ms (6.5%)
- 91% of leaf time in calls >10ms (207 calls)
- Leaf kernel duration: min 0.08ms, avg 8.8ms, max 27.3ms

## Current Architecture (streams + threading, WIP)

### Async GPU BFS (in progress — currently regresses 3.7%)
- 2 CUDA streams, 2 buffer halves (8GB each from 16GB preallocated)
- perft_gpu_host_bfs uses explicit stream: all kernels, memsets, CUB use stream param
- cudaMemcpy → cudaMemcpyAsync + cudaStreamSynchronize (pinned readback memory)
- cudaDeviceSynchronize → cudaStreamSynchronize (stream-local sync)
- Background thread submits GPU BFS on stream 1 while main thread works on stream 0
- Lossless TT store is thread-safe (InterlockedIncrement + CAS for chain prepend)
- **Current regression cause**: thread create/join overhead (~30µs × 425 calls ≈ 12ms),
  plus half-buffer may affect bump allocator patterns. Need profiling.

### Next steps for async (next session)
- Create thread pool once at init (avoid per-call thread create/join overhead)
- Try with full 16GB buffer for each stream (allocate 2 separate buffers)
- Profile with Nsight Systems to identify where the regression comes from
- Consider: the GPU may already be fully saturated (88% leaf kernel), so overlapping
  two calls may just slow both down via SM contention

## Committed TT Optimizations (beyond base TT)

### Dedup generation counter (eliminates dedup memset) [+0.7%]
- Pack 32-bit generation into upper bits of DedupEntry.index1
- Bump generation each BFS level; stale entries ignored (wrong generation)
- Eliminates 54ms of cudaMemset for >100MB dedup tables
- write_dedup_table and check_duplicates take generation parameter

### Lossless CPU hash at launch depth [neutral but kept]
- LosslessEntry: 24 bytes (hashKey, count, next chain pointer)
- LosslessTT: bucket array (int32 heads, -1=empty) + entry pool (bump allocator)
- Thread-safe store: _InterlockedIncrement for pool alloc, CAS loop for chain prepend
- Probe walks chain comparing hashKey = hash.hi ^ hash.lo
- At perft 10-11, neutral vs lossy (pool capacity ~85M vs 128M lossy entries)
- Expected to help at deeper perft (12+) where more unique positions accumulate

### HOST_TT_BUDGET_MB increase [2048 → 65536, neutral but kept]
- User has 90GB+ free system memory, no reason to be conservative
- Host TTs are massively oversized even at perft 13 (822K entries needed vs 512M available)
- Not a bottleneck — kept generous to avoid any edge cases at very deep perft

## Rejected TT Optimizations (tried and measured)

### HASH_IN_LEAF_KERNEL [REJECTED — 35% regression]
- Adding hash compute + TT probe/store to fused 2-level leaf kernel
- Hash overhead (~8 __ldg + ALU) runs for ALL threads, depth-2 TT hit rate too low
- Tested with 8GB equal-share, 512MB fixed shallow TTs — always net loss
- Reference avoids this: BFS goes unfused to depth 2 (normal BFS level, not leaf kernel)
- The fused 2-level leaf architecture is incompatible with efficient leaf-level hashing

### Shallow 8-byte TT entries [REJECTED — only useful with HASH_IN_LEAF_KERNEL]
- ShallowTTEntry: 8 bytes, [63:24]=40-bit hash verification, [23:0]=24-bit count
- Doubles capacity vs 16-byte TTEntry for same memory at depth 2
- Moot since leaf TT is disabled; depth 2 has no TT without HASH_IN_LEAF_KERNEL

### Segmented upsweep reduction [REJECTED — 1% regression]
- Replace atomicAdd scatter (reduce_by_parent) with per-parent sequential sum
- Used inclusive prefix sums (saved during forward pass) as segment boundaries
- Fused with add_tt_hits_and_store, eliminated parentCounts memset
- Fewer threads (1 per parent vs 1 per child) → reduced GPU occupancy
- atomicAdd contention at 30:1 is NOT a bottleneck on Blackwell hardware

### CUDA streams without threading [REJECTED — 1.6% regression]
- All operations use explicit stream (not default), cudaStreamSynchronize
- Overhead of cudaMemcpyAsync staging + stream sync exceeds savings from avoiding global sync
- No true overlap possible with single-threaded CPU — each call still serializes

## Potential Optimization Ideas (untested)

### Non-uniform device TT sizing
- OEIS data shows unique positions per depth vary enormously (9M at depth 6 vs 85B at depth 10)
- Current equal-budget-per-depth wastes VRAM on depths with few unique positions
- Allocate proportional to log(unique_positions) or use a simple large/small split

### Multi-root batching
- Accumulate multiple root positions, launch single GPU BFS processing all of them
- Would keep GPU saturated regardless of individual subtree size
- Requires architectural changes: BFS must track multiple independent root trees
- Composes with async-streams: batch on stream A while batch B runs

### Dual-entry replacement (from reference perft_gpu)
- Two entries per slot: deepest-ever + most-recent
- Halves effective collision rate, 32 bytes per slot

## Next Session TODO
1. **Finish VERBOSE_LOGGING refactor**: code doesn't build with VERBOSE_LOGGING=0 — need to finish wrapping all stats variables and collection code behind the switch
2. **Performance**: perft 12 at 662s, target < 600s
   - Multi-root batching: accumulate positions, launch single BFS for all
   - Non-uniform device TT sizing (proportional to unique positions per depth)
   - Profile with Nsight Systems: where is time spent during perft 12?
3. Async GPU BFS with thread pool (WIP, on side branch `async-streams`)
4. Consider larger PREALLOCATED_MEMORY_SIZE if LD=9 would help (need >26GB for some calls)

## Perft 16 Feasibility Estimate
- Growth factor with TT: ~12-13x per depth (vs ~30x raw branching factor)
- Estimated perft 16 time: ~6-7 months with current setup
- Key bottlenecks for deep perft:
  - Host TT capacity: billions of unique positions at LD=8, need massive tables
  - uint128 overflow: perft(14) ≈ 6.2×10^19 exceeds uint64 max, need uint128 accumulation throughout
  - TT effectiveness degrades: fixed capacity vs exponentially growing position count
