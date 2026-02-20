# perft_gpu Development Notes

## Git
- Push to `https://github.com/ankan-ban/perft_gpu_2026` (NOT perft_gpu)
- Never add `Co-Authored-By` lines to commit messages
- Always do clean rebuilds (`--clean-first`) after editing headers - NVCC doesn't reliably track transitive header dependencies

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
  - Device TTs (depths 2..launchDepth-1): probed in GPU BFS downsweep, stored via upsweep
  - Host TTs (depths launchDepth..maxDepth): probed/stored in CPU recursion
  - Upsweep: after leaf kernel, reduces per-position counts back through BFS levels
  - Leaf TT: fused kernel probes/stores via HASH_IN_LEAF_KERNEL switch

## File Structure
- `perft.cu` — main entry point, CLI parsing, `-cpu` flag handling
- `launcher.cu` / `launcher.h` — GPU init, launch depth estimation, GPU+CPU perft launchers, CPU perft, TT allocation
- `perft_kernels.cu` — GPU kernels, device helpers, host-driven BFS, upsweep, move generator init
- `MoveGeneratorBitboard.h` — move generation logic (templates + magics), ~1950 lines
- `chess.h` — core data structures (QuadBitBoard, GameState, CMove, magic entries)
- `switches.h` — compile-time flags (BLOCK_SIZE, MIN_BLOCKS_PER_MP, USE_TT, HASH_IN_LEAF_KERNEL, etc.)
- `zobrist.h` / `zobrist.cpp` — 128-bit Zobrist hashing (Hash128, ZobristRandoms, computeHash, updateHashAfterMove)
- `tt.h` — transposition table (TTEntry, TTTable, probe/store functions)
- `uint128.h` — simple 128-bit integer for deep perft accumulation
- `utils.h` / `util.cpp` — FEN parsing, board display, Timer class
- `GlobalVars.cpp` — CPU bitboard lookup tables + hardcoded magic entries
- `Magics.cpp` — magic number discovery routines

## Performance
- RTX 6000 Pro Blackwell (with TT + upsweep store): Startpos perft 9 ~6865B nps (0.36s), Position 2 perft 7 ~4846B nps (0.077s)
- RTX 6000 Pro Blackwell (no TT baseline): Startpos perft 9 ~1027B nps (2.38s), Position 2 perft 7 ~1825B nps (0.205s)
- RTX 4090: Startpos perft 9 ~729B nps (3.34s), Position 2 perft 7 ~1103B nps (0.34s)
- RTX 4090: Startpos perft 10 ~639B nps (108s) — previously impossible (OOM)

## Benchmarking
When asked to benchmark, follow this protocol:
- **Startpos perft 9**: single iteration, report nps and time
  ```
  ./build/Release/perft_gpu.exe "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" 9
  ```
- **Position 2 perft 7**: 10 iterations, report **median** nps. Let the app auto-detect launch depth.
  ```
  for i in $(seq 1 10); do ./build/Release/perft_gpu.exe "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" 7 2>&1 | grep "Perft(07)"; done
  ```
- **Run benchmarks SERIALLY** — never run multiple GPU benchmarks in parallel (corrupts measurements)
- Always do a clean rebuild (`--clean-first`) before benchmarking if any code changed
- Compare against baseline numbers above and report delta %

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

### BFS overhead is negligible
- Nsight Systems trace: leaf kernel = 99.4% of GPU time
- All BFS kernels + interval expand + prefix sum = 0.6%
- CPU overhead = ~0.7% (kernel launches + cudaMemcpy)

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
- Device TTs: depths 2 through launchDepth-1 (GPU `cudaMalloc`, separate from BFS buffer)
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

### TT Sizing (current defaults)
- Device TTs: 2GB total budget → ~16M entries per depth at launchDepth=8 (6 depths × 256MB each)
- Host TTs: 2GB total budget → 64M entries per depth for CPU levels
- Entries: 16 bytes each, power-of-2 table sizes

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
- Dynamic OOM fallback: perft_2level_from_board_kernel

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

### Code cleanup [cleanliness]
- Removed 13 compile-time flags, ~2.6MB unused tables
- 8 active flags remain in switches.h (BLOCK_SIZE, LIMIT_REGISTER_USE, MIN_BLOCKS_PER_MP, PREALLOCATED_MEMORY_SIZE, USE_COMBINED_MAGIC_GPU, USE_TT, HASH_IN_LEAF_KERNEL, DEVICE/HOST_TT_BUDGET_MB)

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

## Potential TT Optimization Ideas (untested)

### TT sizing tuning
- Current: equal size per depth. Deeper depths see more unique positions → might benefit from larger TTs
- Try: scale TT size by expected position count per depth (exponential growth toward leaves)
- Try: single shared TT across depths (add depth field to entry) to maximize utilization

### Register pressure with TT
- Hash computation adds ~4-5 registers. Nsight Compute profile needed to check if still at 56 cap
- If register pressure increased: try reducing hash to 64-bit for shallow device TTs (depths 2-3)
- Could split hash update into template-specialized version (chance=template) to reduce branch code

### TT-aware launch depth
- With TT, CPU-side TT hits at launchDepth save entire GPU calls → may want lower launch depth
- Current estimator doesn't account for TT hit rate — could adapt dynamically based on observed hits
- The hash arrays add ~24 bytes/position per BFS level → effective memory budget reduced

### BFS-level deduplication (alternative to TT)
- Instead of TT, sort positions by hash at each BFS level and merge duplicates
- Track multiplicity: same position from different parents gets counted once with weight
- Eliminates ALL duplicate work (not just cached subset), but requires sort + compact at each level
- Could combine with TT: deduplicate within a GPU call, TT across GPU calls

### Upsweep optimization
- Current: atomicAdd per child in reduce_by_parent → contention ~30 per parent
- Alternative: segmented reduction (indices are sorted from interval expand)
- Alternative: use CUB's DeviceSegmentedReduce with segment boundaries from prefix sums

### Host TT with pinned memory
- Current: host TTs use regular malloc
- Pinned memory (cudaHostAlloc) could enable zero-copy GPU access for CPU-level TTs
- Useful if GPU BFS levels overlap with host TT depths (e.g., variable launch depth)

### HASH_IN_LEAF_KERNEL impact
- Currently enabled. Need to benchmark with it disabled (set to 0) to measure leaf TT benefit vs overhead
- Leaf TT adds hash computation + probe to the hottest kernel — register pressure concern
- Nsight Compute re-profiling needed to check register count, occupancy, stall breakdown

### Shallow hash entries (from reference perft_gpu)
- Reference uses 8-byte entries at shallow depths (perft values fit in 24 bits, pack into hash key)
- Doubles effective table capacity for leaves where most positions live
- Our current 16-byte uniform entries waste space at depths 1-4 where counts are small
- Could use: `ShallowTTEntry { uint64 hashHi_xor_count24; }` at depths 2-4

### Dual-entry replacement (from reference perft_gpu)
- Reference stores two entries per slot: deepest-ever + most-recent
- Prevents valuable deep entries from being evicted by shallow ones
- 32 bytes per slot but halves effective collision rate
- Most impactful at depths with many unique positions competing for limited slots

### BFS-level duplicate detection (from reference perft_gpu)
- Separate small hash table to find exact duplicates within a single BFS level
- Before interval expand: if two parent positions generate the same child, keep only one
- Completely eliminates redundant computation (vs TT which only catches cached subset)
- Reference used this at depth 1 (FIND_DUPLICATES_IN_BFS flag)

### Lossless CPU hash tables (from reference perft_gpu)
- Reference uses chained hash tables (linked lists) for CPU levels — no entry is ever lost
- More memory hungry but guarantees every computed result is reusable
- Most impactful at the launch boundary depth where each TT hit saves an entire GPU BFS call
- Allocates chain entries in large chunks (128M entries per chunk)
