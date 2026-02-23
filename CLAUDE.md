# perft_gpu Development Notes

## Git
- Push to `https://github.com/ankan-ban/perft_gpu_2026` (NOT perft_gpu)
- Never add `Co-Authored-By` lines to commit messages
- Always do clean rebuilds (`--clean-first`) after editing headers - NVCC doesn't reliably track transitive header dependencies
- **Never commit or push until the user has manually reviewed/tested and given explicit go-ahead**

## Branches
- **`master`** — main branch with TT support (runtime toggle via `-nott`)
- **`simple_no_tt`** — archived pre-TT codebase snapshot

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
- **Transposition tables** (runtime toggle via `g_useTT` / `-nott` CLI flag):
  - 128-bit Zobrist hash, separate TT per depth, XOR lockless 16-byte entries
  - Device TTs (depths 3..launchDepth-1): probed in GPU BFS downsweep, stored via upsweep
  - Host TTs (depths launchDepth..maxDepth): lossless chained tables, probed/stored in CPU recursion
  - `makemove_and_count_moves_kernel<bool useTT>` templated for zero overhead when TT disabled
  - Upsweep result is authoritative (leaf global atomic misses TT-hit subtrees)

## File Structure
- `perft.cu` — main entry point, CLI parsing, flag handling
- `launcher.cu` / `launcher.h` — GPU init, launch depth estimation, GPU+CPU perft launchers, TT allocation, OOM fallback
- `perft_kernels.cu` — GPU kernels, host-driven BFS, upsweep, move generator init
- `MoveGeneratorBitboard.h` — move generation logic (templates + magics), ~1950 lines
- `chess.h` — core data structures (QuadBitBoard, GameState, CMove, magic entries)
- `switches.h` — compile-time flags (BLOCK_SIZE, MIN_BLOCKS_PER_MP, HASH_IN_LEAF_KERNEL, VERBOSE_LOGGING, etc.)
- `zobrist.h` / `zobrist.cpp` — 128-bit Zobrist hashing
- `tt.h` — transposition table (TTEntry, TTTable, LosslessTT, probe/store functions)
- `uint128.h` — simple 128-bit integer for deep perft accumulation
- `utils.h` / `util.cpp` — FEN parsing, board display, Timer class

## CLI Flags
```
perft_gpu <fen> <depth> [-nott] [-dtt <MB>] [-htt <MB>] [-ld <N>] [-cpu]
```

## Performance Baselines
- **RTX 6000 Pro Blackwell (TCC, TT enabled, LD=8)**: perft 10: 3.39s (20,428B nps) | perft 11: 41.15s (50,979B nps) | perft 12: 584.6s (107,516B nps) | perft 13: 9,126s (217,074B nps)
- **RTX 6000 Pro Blackwell (no TT)**: perft 9: 2.05s (~1,189B nps)
- **RTX 4090 (TT enabled)**: perft 9: 0.47s (~5,156B nps) | perft 10: 5.71s (~12,149B nps) | perft 11: 78.95s (~26,569B nps)
- **RTX 4090 (no TT)**: perft 9: 3.25s (~750B nps) | perft 10: 95.0s (~730B nps)

## Benchmarking
- Only test **startpos perft 10+** — shallower depths are too fast to measure meaningfully
- **Run benchmarks SERIALLY** — never run multiple GPU benchmarks in parallel
- Always do a clean rebuild (`--clean-first`) before benchmarking if any code changed

## Key Design Decisions
- **LD never increases dynamically** — root perft(N) memory does NOT predict worst-case perft(N+1) from arbitrary positions. LD can only decrease on OOM.
- **OOM fallback** — all BFS OOM paths signal `g_lastBfsOom`, caller falls back to CPU recursion at depth-1. Never return wrong results from OOM — they poison TTs.
- **Block size 384, MIN_BLOCKS_PER_MP=3** — extensively tested, optimal for 56 registers / 75% occupancy
- **HASH_IN_LEAF_KERNEL disabled** — 35% regression; fused 2-level leaf architecture incompatible with efficient leaf-level hashing
- **VERBOSE_LOGGING** (switches.h) — WIP, code does NOT build with VERBOSE_LOGGING=0

## GPU Optimization Lessons
- NVCC is extremely sensitive to source-level patterns — "optimizing" code can trigger regressions
- Never introduce `uint8` local variables in hot GPU paths (disrupts register allocation)
- Magic bitboards are essential: trade ALU for LSU, balancing pipe utilization
- Per-thread stack growth overwhelms L1 (rejected: 3-level leaf, piece-in-CMove, iterator generateMoves, FGMC)
- Pure ALU approaches overload saturated ALU heavy pipe (rejected: QBB sliding attacks, Kogge-Stone, software popcount)
- Multi-stream rejected: GPU already 93-97% utilized, SM contention causes 2-9% regression worsening with depth

## Rejected Optimizations (do not re-try)
| Optimization | Result | Root cause |
|---|---|---|
| getOne/bitScan elimination | Instability (18x slower) | NVCC already optimizes internally |
| uint8 flags local variable | -18x | Disrupts register allocation |
| Checkmask/pinmask (Gigantua) | -5-9% | Extra instructions > divergence savings on GPU |
| QBB sliding attacks | -74% | Overloads saturated ALU pipe |
| 3-level leaf | -26% | 1.1KB/thread stack overwhelms L1 |
| __any_sync warp skips | -3.4% | VOTE intrinsic overhead > savings |
| Software popcount | -24% | Hardware POPC irreplaceable |
| LUT knight/king in findAttackedSquares | -2.3% | LSU latency > bulk ALU |
| v4.u64/ulonglong4 combined magic | -1.9% | Loses __ldg cache hint |
| X-ray pin calculator | -7-9% | 4 extra magic lookups > short loop |
| Kogge-Stone findAttackedSquares | -10-21% | Pure ALU overloads heavy pipe |
| Piece-in-CMove (uint32) | -14-23% | CMove array doubled, L1 overwhelmed |
| Iterator generateMoves | -20-39% | Register spills at ~30 callback sites |
| FGMC (fused generate-make-count) | -19-37% | Same as iterator |
| Batched per-piece generateMoves | -3.5-21% | State spills across batches |
| Template-specialized makeMove dispatch | -2.1% | 6-way switch causes warp divergence |
| Unfused bfsMinLevel=2 | -52% / OOM | Massive BFS arrays at level 2 |
| Dynamic LD increase 8->9 | -75% at perft 12 | OOMs from non-root positions + TT dilution |
| Multi-stream (2 workers) | -2-9% | SM contention, worsens with depth |
| HASH_IN_LEAF_KERNEL | -35% | Hash overhead for all threads, low hit rate at depth 2 |
| Segmented upsweep | -1% | Fewer threads, atomicAdd not a bottleneck |
| CUDA streams (no threading) | -1.6% | Stream sync overhead > savings |
| Cached ~b inversions | Neutral | Prevents LOP3.LUT fusion |
| Branchless castling | Neutral | Already branchless via predication |
| Bulk pawn gen in generateMoves | Neutral | Per-pawn loop already efficient for from/to pairs |
