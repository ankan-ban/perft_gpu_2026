# perft_gpu Development Notes

See `README.md` for project overview, usage, CLI flags, file overview, build instructions, and current performance numbers.

## Git
- Push to `https://github.com/ankan-ban/perft_gpu_2026` (NOT perft_gpu)
- Never add `Co-Authored-By` lines to commit messages
- Always do clean rebuilds (`--clean-first`) after editing headers - NVCC doesn't reliably track transitive header dependencies
- **Never commit or push until the user has manually reviewed/tested and given explicit go-ahead**

## Branches
- **`master`** — main branch with TT support (runtime toggle via `-nott`)
- **`simple_no_tt`** — archived pre-TT codebase snapshot

## Architecture notes (not in README)
- `GameState` is 1 byte packed bitfields: whiteCastle:2 + blackCastle:2 + enPassent:4
- Color (chance) passed explicitly — not stored in GameState
- Combined magic entry struct: mask+factor co-located for single cache-line access
- Pure CPU path (`-cpu`) is template-specialized, countMoves at leaf
- **Transposition tables** (runtime toggle via `g_useTT` / `-nott` CLI flag):
  - 128-bit Zobrist hash, separate TT per remaining depth
  - **Depth 2 (leaf TT)**: 16B XOR-locked `TTEntry`, probed inside `fused_2level_leaf_kernel` (gated by `HASH_IN_LEAF_KERNEL`)
  - **Depth 3 (shallow TT)**: 8B packed entry `{hash.lo[63:24] | count[23:0]}` — half the entry size, 2× capacity per byte (gated by `USE_SHALLOW_TT_DEPTH3`). 24-bit count safe (218³ ≈ 10.4M < 2²⁴). Probe is `ld.global.u64`, store is `st.global.u64`. Drops `hash.hi` cross-check (collision resistance 128-bit → 64-bit, still negligible at TT sizes ≤ 2³²).
  - **Depths 4..launchDepth-1**: 16B XOR-locked `TTEntry` — probed in GPU BFS downsweep, stored via upsweep
  - **Host TTs (depths launchDepth..maxDepth)**: lossless chained tables, probed/stored in CPU recursion
  - `makemove_and_count_moves_kernel<bool useTT, typename TTT>` and `add_tt_hits_and_store<typename TTT>` are templated on TT type; overload-resolved `ttProbe`/`ttStore` pick the right path per depth. Zero overhead when TT disabled.
  - Upsweep result is authoritative (leaf global atomic misses TT-hit subtrees)

## Benchmarking
- Only test **startpos perft 10+** — shallower depths are too fast to measure meaningfully
- **Run benchmarks SERIALLY** — never run multiple GPU benchmarks in parallel
- Always do a clean rebuild (`--clean-first`) before benchmarking if any code changed

## Key Design Decisions
- **LD never increases dynamically** — root perft(N) memory does NOT predict worst-case perft(N+1) from arbitrary positions. LD can only decrease on OOM.
- **OOM fallback** — all BFS OOM paths signal `g_lastBfsOom`, caller falls back to CPU recursion at depth-1. Never return wrong results from OOM — they poison TTs.
- **Block size 416, MIN_BLOCKS_PER_MP=3** — extensively tested on Blackwell (RTX PRO 6000); was 384 on older chips
- **HASH_IN_LEAF_KERNEL ENABLED** — +7% on startpos perft 10 (2.14s → 2.00s, 3-run avg, Blackwell). The historical -35% rejection predates current code: leaf TT[2] now allocates 32 GB (auto-budget) and hit rate is high enough to pay for the extra hash update per leaf thread. Skipping it leaves 32 GB of VRAM idle — the other TTs cap out at 8 GB per slot regardless.
- **VERBOSE_LOGGING** (switches.h) — gates call-size/time histograms, host TT hit-rate counters, per-BFS-level position counts, progress reporting. Both `=0` (default, lean release) and `=1` (diagnostic) build cleanly.

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
| Segmented upsweep | -1% | Fewer threads, atomicAdd not a bottleneck |
| CUDA streams (no threading) | -1.6% | Stream sync overhead > savings |
| Cached ~b inversions | Neutral | Prevents LOP3.LUT fusion |
| Branchless castling | Neutral | Already branchless via predication |
| Bulk pawn gen in generateMoves | Neutral | Per-pawn loop already efficient for from/to pairs |
| Multithreaded CPU recursion (16 threads) | Neutral (+1.4%) | GPU calls serialized by mutex; not enough CPU work to overlap at perft 11 |
| Batched GPU launches (depth LD+1) | -2.3% at perft 11 | Reduces ~18K GPU calls to ~600; modest win, slight regression at perft 10 |
| 2-way set-associative device TT | Neutral (+0.8%) | Device TTs already large enough; low collision rate; extra probe load overhead |
| Parent non-slider attack cache (per-element dispatch) | -20% at perft 7 pos2 -nott | Branch divergence in inner loop: avg active threads/warp 25.77 -> 21.37, local memory spilling +63%, executed instructions +19%. Cache savings of ~15 ops × 40% slider rate (= 6 ops/child) dwarfed by dispatch overhead. |
| Parent non-slider attack cache (in-place partition) | -9% at perft 7 pos2 -nott | Partition pre-pass eliminates per-element branch but introduces its own divergence + local memory writes for the swap; two-loop structure adds register pressure. Net cache savings too small to justify any overhead on GPU. |
| BLSR in generateMovesToSink outer slider/knight loops | -3.0% at perft 7 pos2 -nott | NVCC's existing register allocation for the outer-piece-loop interacts with the inner emit loop's live `bishop`/`rook` register. Swapping XOR -> BLSR specifically here disrupts that schedule (countMoves outer loops are fine because they have no nested inner loop). |
| BLSR in multi*Attacks / findPinnedPieces / countMovesOutOfCheck slider / EP+pinnedPawns in countMoves | Neutral | Either NVCC already does the transform internally for these specific loops, or these loops have too little leverage (in-check path is ~3-5% of leaves; EP-able positions are rare). |

## Accepted optimizations (with measurement notes)

- **Shallow 8B TT entry at remaining-depth 3** (`tt.h` `ShallowTTTable`, `USE_SHALLOW_TT_DEPTH3` switch). Packs `{hash.lo[63:24] | count[23:0]}` into a single uint64, halving entry size vs the 16B XOR-locked `TTEntry`. In the same 8 GB byte budget the table holds 1024M entries instead of 512M — lower load factor + halved probe/store bandwidth (single `ld.global.u64`/`st.global.u64`). Win: -2.8% on startpos perft 10 (2.089s → 2.031s, 3-run avg, Blackwell). 24-bit count is safe at depth 3 (218³ ≈ 10.4M < 2²⁴); store skipped if count overflows (never silently truncates). Toggleable via `USE_SHALLOW_TT_DEPTH3=0` to fall back to 16B `TTEntry`. NOT safe to apply at depth ≥ 4 (218⁴ ≈ 2.3B > 2²⁴).
- **HASH_IN_LEAF_KERNEL (leaf TT at remaining-depth 2)** (`switches.h:HASH_IN_LEAF_KERNEL=1`). The fused 2-level leaf kernel probes/stores `deviceTTs[2]` (16B XOR-locked, auto-sized to ~32 GB on Blackwell). Win: +7% on startpos perft 10 (2.14s → 2.00s, 3-run avg). Hit rate at depth 2 is high enough to offset the per-leaf-thread hash update + probe cost. The leaf TT alloc takes ~32 GB of an ~78 GB device-TT budget; disabling it leaves that VRAM unused (other TTs cap at 8 GB/slot regardless), so the choice is "use it or waste it".
- **BLSR (`bb &= bb - 1`) replacing `bb ^= piece` in countMoves outer piece loops** (`MoveGeneratorBitboard.h:1876-1940`, knight + pinned/unpinned bishop + pinned/unpinned rook). Win: -1.5% on pos2 perft 7 -nott. NCU: warp cycles per issued inst 12.45 -> 12.42, SM throughput 68.41 -> 68.59%. `getOne`+`bitScanOne` are preserved (full `getOne/bitScan elimination` regresses 18x because NVCC relies on those primitives).
- **BLSR replacing `Moves ^= dst` in generateMovesToSink/generateMovesOutOfCheck inner destination-iteration emit loops** (10 sites total). Win: an additional -0.3% on top of the prior change. NCU: executed instructions drop 247.76B -> 247.32B (-0.18%); savings compound across the BFS.
- Cumulative effect: ~1.5% wall-clock improvement across pos2 perft 7 (both -nott and TT) and startpos perft 10 TT. Confirmed via direct A/B against the prior commit.
- **BLSR scope rule for future ports**: BLSR helps in piece loops that have NO nested inner emit loop (countMoves: just popCount per piece) and in destination-iteration inner emit loops. BLSR HURTS in piece loops that contain a nested inner emit loop (generateMovesToSink outer slider) — the outer-loop `piece` register must stay live for the inner loop, and switching the outer's clear step disrupts NVCC's joint register schedule.

## NCU baseline notes (fused_2level_leaf_kernel, useTT=0, pos2 perft 7 dominant launch)
- SM throughput: ~68.6%; IPC active: 2.75; achieved occupancy: 71% (theoretical 81.25%, 48 regs/thread, 3 blocks/SM)
- Local memory spilling: 1.5B requests over the launch — comes from the per-thread CMove[80] array. Can't easily shrink (FUSED_CHILD_MOVES_CAP=80 is the safe upper bound; iterator/FGMC alternatives are on the rejected list).
- Stall reason breakdown (per issue active): wait 267% > long_scoreboard 227% > not_selected 222% > math_pipe_throttle 134% > short_scoreboard 97% > no_instruction 84%. ALU pipe (math_pipe_throttle) is the saturated resource; "wait" reflects fixed-latency math op chains.
- L1 hit 95.83%, L2 hit 99.35%, DRAM 11 GB/s (40% of bandwidth) — NOT bandwidth-limited.
