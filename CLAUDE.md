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
| Multithreaded CPU recursion (16 threads) | Neutral (+1.4%) | GPU calls serialized by mutex; not enough CPU work to overlap at perft 11 |
| Batched GPU launches (depth LD+1) | -2.3% at perft 11 | Reduces ~18K GPU calls to ~600; modest win, slight regression at perft 10 |
| 2-way set-associative device TT | Neutral (+0.8%) | Device TTs already large enough; low collision rate; extra probe load overhead |
| Parent non-slider attack cache (per-element dispatch) | -20% at perft 7 pos2 -nott | Branch divergence in inner loop: avg active threads/warp 25.77 -> 21.37, local memory spilling +63%, executed instructions +19%. Cache savings of ~15 ops × 40% slider rate (= 6 ops/child) dwarfed by dispatch overhead. |
| Parent non-slider attack cache (in-place partition) | -9% at perft 7 pos2 -nott | Partition pre-pass eliminates per-element branch but introduces its own divergence + local memory writes for the swap; two-loop structure adds register pressure. Net cache savings too small to justify any overhead on GPU. |
| BLSR in generateMovesToSink outer slider/knight loops | -3.0% at perft 7 pos2 -nott | NVCC's existing register allocation for the outer-piece-loop interacts with the inner emit loop's live `bishop`/`rook` register. Swapping XOR -> BLSR specifically here disrupts that schedule (countMoves outer loops are fine because they have no nested inner loop). |
| BLSR in multi*Attacks / findPinnedPieces / countMovesOutOfCheck slider / EP+pinnedPawns in countMoves | Neutral | Either NVCC already does the transform internally for these specific loops, or these loops have too little leverage (in-check path is ~3-5% of leaves; EP-able positions are rare). |

## Accepted optimizations (with measurement notes)

- **BLSR (`bb &= bb - 1`) replacing `bb ^= piece` in countMoves outer piece loops** (`MoveGeneratorBitboard.h:1876-1940`, knight + pinned/unpinned bishop + pinned/unpinned rook). Win: -1.5% on pos2 perft 7 -nott. NCU: warp cycles per issued inst 12.45 -> 12.42, SM throughput 68.41 -> 68.59%. `getOne`+`bitScanOne` are preserved (full `getOne/bitScan elimination` regresses 18x because NVCC relies on those primitives).
- **BLSR replacing `Moves ^= dst` in generateMovesToSink/generateMovesOutOfCheck inner destination-iteration emit loops** (10 sites total). Win: an additional -0.3% on top of the prior change. NCU: executed instructions drop 247.76B -> 247.32B (-0.18%); savings compound across the BFS.
- Cumulative effect: ~1.5% wall-clock improvement across pos2 perft 7 (both -nott and TT) and startpos perft 10 TT. Confirmed via direct A/B against the prior commit.
- **BLSR scope rule for future ports**: BLSR helps in piece loops that have NO nested inner emit loop (countMoves: just popCount per piece) and in destination-iteration inner emit loops. BLSR HURTS in piece loops that contain a nested inner emit loop (generateMovesToSink outer slider) — the outer-loop `piece` register must stay live for the inner loop, and switching the outer's clear step disrupts NVCC's joint register schedule.

## NCU baseline notes (fused_2level_leaf_kernel, useTT=0, pos2 perft 7 dominant launch)
- SM throughput: ~68.6%; IPC active: 2.75; achieved occupancy: 71% (theoretical 81.25%, 48 regs/thread, 3 blocks/SM)
- Local memory spilling: 1.5B requests over the launch — comes from the per-thread CMove[80] array. Can't easily shrink (FUSED_CHILD_MOVES_CAP=80 is the safe upper bound; iterator/FGMC alternatives are on the rejected list).
- Stall reason breakdown (per issue active): wait 267% > long_scoreboard 227% > not_selected 222% > math_pipe_throttle 134% > short_scoreboard 97% > no_instruction 84%. ALU pipe (math_pipe_throttle) is the saturated resource; "wait" reflects fixed-latency math op chains.
- L1 hit 95.83%, L2 hit 99.35%, DRAM 11 GB/s (40% of bandwidth) — NOT bandwidth-limited.

## Candidate Optimization: Parent Non-Slider Attack Cache [TESTED, REJECTED — see rejected table above]

Originally proposed based on `f240b19` in `perft_cpu_2026`. Tested with two implementations (per-element dispatch and in-place partition); both regressed. See the rejected table entries for the failure analysis. The section below is preserved for historical reference; **do not re-attempt this approach.**

### Theoretical basis

`findAttackedSquares` (MoveGeneratorBitboard.h:835) computes 5 components: pawn shifts, knight LUT, bishop magics, rook magics, king LUT. For any position R reached from Q by a move M:
- **Enemy slider attacks in R** depend on R's occupancy → must recompute.
- **Enemy pawn/knight/king attack bitboards in R** depend only on enemy P/N/K piece locations. The enemy (= side that just moved) is the side that played M.
- **Invariant**: if M did not change enemy P/N/K positions, then `nonSliderAttacks(R) == nonSliderAttacks(Q)`.

A safe sufficient condition: M is a slider move (Bishop/Rook/Queen) AND M is not a promotion (promotions turn a pawn into a slider — changes enemy P bitboard). Castling fails the test (king is non-slider, also moves a rook). Pawn/knight/king moves fail. EP fails. Slider captures of *our* pieces are fine (our pieces don't enter enemy non-slider attacks).

A more permissive runtime test would be `(Q.allPawns & Q.enemy) == (R.allPawns & R.enemy)` and same for knights/kings, but the static "slider-only-non-promotion" classification is free at generate time and likely captures the bulk of the win.

### Where to insert

| Location | What changes |
|---|---|
| `perft_kernels.cu:382` `fusedCountChildren<childColor>` | Compute `enemyNSAtk = pawnAttacks(enemyPawns) \| knightAttacks(enemyKnights) \| kingAttacks(enemyKing)` of `pos` ONCE before the child loop. Iterate `[0, nSliders)` with the fast path, `[nSliders, nChildMoves)` with the existing slow path. |
| `MoveGeneratorBitboard.h:1557` `generateMoves` | Change signature to return `{nMoves, nSliderMoves}` (e.g., pack as `uint32 = (nSliders<<16) \| nMoves`). Reorder emission in `generateMovesToSink` so slider non-promotion moves come first, then everything else. |
| `MoveGeneratorBitboard.h:1161` `generateMovesToSink` | Today's order is pawns → king → knights → bishops → rooks (1190 comment). Reorder to: bishop-non-promo → rook-non-promo → queen-non-promo → pawns → knights → king → castling → promotions. Record `nSliders` at the boundary. |
| `MoveGeneratorBitboard.h:1700` `countMoves` | Add overload `countMoves<chance, useCachedNS>(pos, gs, cachedNSAtk)`. When `useCachedNS`, the call to `findAttackedSquares` at line 1719 is replaced with `cachedNSAtk \| multiBishopAttacks(enemyBishops, proWithKing) \| multiRookAttacks(enemyRooks, proWithKing)`. The pinned/check logic stays identical. |

CMove encoding stays unchanged — all 4 flag bits are used (chess.h:120-144), so no room for a piece-type bit. Reordering the *emission* in `generateMovesToSink` is the carrier for slider-vs-non-slider classification.

### Why variant B (emission reorder) over variant A (encode in CMove)

Variant A (expand CMove or use a parallel piece-type array) is in the rejected list: "Piece-in-CMove (uint32) −14-23%" — doubling per-thread CMove array overwhelmed L1. Variant B keeps CMove at 2 bytes, doesn't grow the `childMoves[80]` array, and turns the classification into a *partition index* rather than per-element metadata.

### Implementation steps

1. **Add a new helper in MoveGeneratorBitboard.h**:
   ```cpp
   CUDA_CALLABLE_MEMBER CPU_FORCE_INLINE static uint64
   findNonSliderAttacks(uint64 enemyPawns, uint64 enemyKnights,
                        uint64 enemyKing, uint8 enemyColor);
   ```
   Implementation = first/second/fifth blocks of `findAttackedSquares` only. Place adjacent to that function.

2. **Reorder `generateMovesToSink`** so slider non-promotions emit first. Track count. The existing per-piece loops (bishops/rooks/queens at ~1352/1374) move to the top; pawns/king/knights move below. Castling (a king move that also moves a rook) stays on the slow side.

3. **Change `generateMoves` return** to a packed `(nSliders, nMoves)` pair. Update its single caller in `perft_kernels.cu:386`. The other consumer (top-level CPU recursion in `launcher.cu:706`) needs the count too — easiest to ignore `nSliders` there and call `countMoves` without cache. Make the cache parameter optional via overload.

4. **Templated `countMoves<chance, bool useCachedNS>`**: when `useCachedNS=true`, skip the non-slider attack computation in `findAttackedSquares`. Templating (not runtime flag) keeps NVCC's existing register allocation pattern intact — both specializations compile separately.

5. **Update `fusedCountChildren`** to split the loop:
   ```cpp
   uint64 enemyNSAtk = findNonSliderAttacks(...);  // from pos
   // ... existing generateMoves call returns {nSliders, nTotal} ...
   for (i = 0; i < nSliders; i++) { ... countMoves<...,true>(..., enemyNSAtk); }
   for (; i < nTotal; i++)        { ... countMoves<...,false>(...); }
   ```

6. **Verify correctness BEFORE benchmarking**:
   - Run `-cpu` perft on startpos through depth 6 — must match known counts byte-for-byte.
   - Run GPU `-nott` perft 8 on startpos AND on Kiwipete (CPW pos2: `r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1`) and pos3/pos4/pos5 from the standard CPW test set — must match references.
   - Why both with and without TT: TT correctness requires perft counts to be stable; a per-leaf bug would corrupt entries and cascade.

### Risks and abort criteria

- **Register pressure**: extra uint64 carried into `countMoves` + extra live value in `fusedCountChildren`. Current cap is 56 regs at 75% occupancy (BLOCK_SIZE=384, MIN_BLOCKS_PER_MP=3). Check `--ptxas-options=-v` for register count. If it spills above 56, occupancy drops to 50% → likely regression. Mitigation: NVCC may recompute the cache locally if it's cheaper than spilling — but the templated path forces use of the parameter, so spills are likely. Consider passing only the pawn+knight portion and recomputing king attacks (cheap LUT) inside countMoves.
- **NVCC codegen pathology**: project history shows that adding even a single local variable can wreck the leaf kernel (re: uint8 flags, −18×). Inspect SASS via `cuobjdump --dump-sass` on `_Z*fused_2level_leaf_kernel*` before and after. Diff for new spill instructions (`LDL`/`STL` to local memory).
- **Loop split overhead**: two back-to-back for-loops may prevent some loop-invariant hoisting NVCC currently does on the single loop. If both lower-bound and upper-bound are computed correctly, NVCC should still fuse the prologue. Worst case: combine into one loop with a runtime `i < nSliders` branch — NVCC may predicate.
- **Emission-order side effects**: the comment at MoveGeneratorBitboard.h:1190 explicitly notes "Generate pawn moves first. This changes fused child replay order without adding a second replay scan." → there is *prior tuning* of emission order. Pawn-first may itself be a memory-locality win because consecutive pawn children share QBB cache structure. Reordering to slider-first may regress this independently of the cache benefit. **This is the highest-risk concern.** Measure with both orderings.
- **Castling classification**: a castling move's `from` is the king square — it must NOT be on the slider-fast path even though a rook also moves. Castling has its own flags (`CM_FLAG_KING_CASTLE`/`CM_FLAG_QUEEN_CASTLE`) — keep it in the slow segment.
- **Promotions**: bishop/rook/queen promotions move a pawn → change enemy P bitboard if next ply is the enemy. They must be slow path. Promotion flags (CM_FLAG_PROMOTION = bit 3) are separable.

**Abort if**: register count climbs above 60, SASS shows new local memory traffic in the leaf inner loop, or startpos perft 10 regresses by >0.5% with the reordering alone (no cache use) — that means the emission-order tuning was load-bearing.

### Measurement protocol

1. Clean rebuild after each change: `cmake --build build --clean-first --config Release`.
2. **Sanity**: `perft_gpu <startpos> 7 -nott` matches known answer.
3. **Order-only A/B**: implement the emission reorder WITHOUT the cache. Measure perft 10 and perft 11. If regression > 0.5%, the emission order is load-bearing — try keeping pawns first and grouping sliders separately at the end of emission (and adjusting indices accordingly).
4. **Cache enabled**: enable the templated `useCachedNS=true` path. Measure perft 10 and perft 11 startpos, no TT and with TT. Three runs each, take min.
5. **Compare**: target is ≥1% improvement at perft 11 (TT-enabled). Anything less and the risk/code-complexity ratio doesn't justify keeping it.
6. **Profile**: if neutral or marginal, profile with Nsight Compute on the fused leaf kernel — look at SM_INST_EXECUTED_PIPE_ALU and LSU pipe utilization deltas. If ALU dropped without LSU rising, the cache is working at the instruction level even if wall-time is hidden by latency.

### What success looks like
A net 1-3% improvement at perft 11/12, no SASS spills, correctness across the CPW test set. Update the table above and the Rejected Optimizations table appropriately; if accepted, document the slider-first emission order as a constraint future refactors must preserve.
