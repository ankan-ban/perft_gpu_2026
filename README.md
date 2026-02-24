# perft_gpu

A GPU-accelerated chess [perft](https://www.chessprogramming.org/Perft) calculator using CUDA.

This is a heavily optimized rewrite of the [original perft_gpu](https://github.com/ankan-ban/perft_gpu) project, stripped down to single-GPU mode with a host-driven BFS approach. The focus is on raw move generation throughput, with optional transposition tables for deep perft.

## Usage

```
perft_gpu <fen> <depth> [-nott] [-dtt <MB>] [-htt <MB>] [-ld <N>] [-cpu]
```

| Flag | Description |
|---|---|
| `<fen>` | FEN string for the position to analyze |
| `<depth>` | Maximum perft depth |
| `-nott` | Disable transposition tables (raw move generation throughput) |
| `-dtt <MB>` | Override device TT memory budget in MB (default: auto, 95% of free VRAM) |
| `-htt <MB>` | Override host TT memory budget in MB (default: auto, 90% of system RAM) |
| `-ld <N>` | Manual BFS/CPU switchover depth (auto-detected if omitted) |
| `-cpu` | Pure CPU mode (no GPU, useful for debugging/comparison) |

Transposition tables are enabled by default. They provide massive speedups at deeper depths by caching previously computed subtree counts.

### Examples

```bash
# Starting position, depth 10, with transposition tables (default)
perft_gpu "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" 10

# Kiwipete position, depth 7, without transposition tables
perft_gpu "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -" 7 -nott

# With explicit launch depth and custom TT budgets
perft_gpu "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" 12 -ld 8 -dtt 8192 -htt 32768

# Pure CPU mode (no GPU)
perft_gpu "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" 7 -cpu
```

The program prints cumulative perft results for depth 1 through the specified maximum depth.

## Performance

### NVIDIA RTX 6000 Pro (Blackwell, 96GB VRAM, TCC mode)

Without transposition tables:

| Position | Depth | Nodes | Time | Speed |
|---|---|---|---|---|
| Starting position | 9 | 2,439,530,234,167 | 2.05s | ~1,189 billion nps |
| [Position 2](https://www.chessprogramming.org/Perft_Results#Position_2) (Kiwipete) | 7 | 374,190,009,323 | 0.19s | ~1,965 billion nps |

With transposition tables enabled, lossless host TTs, launch depth 8:

| Position | Depth | Nodes | Time | Speed |
|---|---|---|---|---|
| Starting position | 9 | 2,439,530,234,167 | 0.29s | ~8,308 billion nps |
| Starting position | 10 | 69,352,859,712,417 | 3.39s | ~20,428 billion nps |
| Starting position | 11 | 2,097,651,003,696,806 | 41.15s | ~50,979 billion nps |
| Starting position | 12 | 62,854,969,236,701,747 | 584.6s | ~107,516 billion nps |
| Starting position | 13 | 1,981,066,775,000,396,239 | 9,126s | ~217,074 billion nps |

### NVIDIA RTX 4090

Without transposition tables:

| Position | Depth | Nodes | Time | Speed |
|---|---|---|---|---|
| Starting position | 9 | 2,439,530,234,167 | 3.25s | ~750 billion nps |
| Starting position | 10 | 69,352,859,712,417 | 95.0s | ~730 billion nps |
| [Position 2](https://www.chessprogramming.org/Perft_Results#Position_2) (Kiwipete) | 7 | 374,190,009,323 | 0.311s | ~1,204 billion nps |
| [Position 3](https://www.chessprogramming.org/Perft_Results#Position_3) | 10 | 860,322,381,070 | 1.82s | ~474 billion nps |

With transposition tables enabled, lossless host TTs, launch depth 8:

| Position | Depth | Nodes | Time | Speed |
|---|---|---|---|---|
| Starting position | 9 | 2,439,530,234,167 | 0.47s | ~5,156 billion nps |
| Starting position | 10 | 69,352,859,712,417 | 5.71s | ~12,149 billion nps |
| Starting position | 11 | 2,097,651,003,696,806 | 78.95s | ~26,569 billion nps |

### NVIDIA DGX Spark (GB10 Grace Blackwell, 128GB unified LPDDR5x)

Without transposition tables:

| Position | Depth | Nodes | Time | Speed |
|---|---|---|---|---|
| Starting position | 8 | 84,998,978,956 | 0.356s | ~239 billion nps |
| Starting position | 9 | 2,439,530,234,167 | 9.38s | ~260 billion nps |
| [Position 2](https://www.chessprogramming.org/Perft_Results#Position_2) (Kiwipete) | 7 | 374,190,009,323 | 0.86s | ~435 billion nps |

With transposition tables enabled, lossless host TTs, launch depth 8:

| Position | Depth | Nodes | Time | Speed |
|---|---|---|---|---|
| Starting position | 8 | 84,998,978,956 | 0.112s | ~759 billion nps |
| Starting position | 9 | 2,439,530,234,167 | 1.38s | ~1,763 billion nps |
| Starting position | 10 | 69,352,859,712,417 | 16.71s | ~4,151 billion nps |
| Starting position | 11 | 2,097,651,003,696,806 | 214.88s | ~9,762 billion nps |

## How it works

The perft tree is explored using breadth-first search driven from the host. At each BFS level, CUDA kernels expand all positions at the current depth in parallel:

1. **Make moves + count children** - Apply each move and count the resulting legal moves
2. **Prefix scan** (CUB InclusiveSum) - Compute output offsets for the next level
3. **Merge-path interval expand** - Load-balanced mapping from child indices back to parent indices, ensuring all CTAs do equal work regardless of how many moves each position has
4. **Generate moves** - Write out child moves to the next level's buffer using shared memory staging for coalesced writes

The last two levels of the tree are handled by a **fused 2-level leaf kernel**: each thread makes a move, generates all child moves locally, then for each child makes the move and counts grandchildren. This eliminates the most memory-intensive BFS expansion and enables deeper perft without running out of GPU memory.

### Key design choices

- **Quad-bitboard representation** - Each position is stored as 4 x uint64 (32 bytes) encoding piece type + color per square, with game state in a separate 1-byte packed struct. This gives power-of-2 struct size for GPU coalescing.
- **Host-driven BFS** with one kernel launch per phase per level (no CUDA Dynamic Parallelism)
- **Fused 2-level leaf kernel** - Handles the last 2 tree levels per-thread, eliminating massive memory allocations for the deepest BFS level and improving cache locality
- **Merge-path interval expand** (inspired by [moderngpu](https://github.com/moderngpu/moderngpu)) for perfect load balancing across CTAs
- **Bump allocator** for GPU memory - a single pre-allocated buffer avoids per-level `cudaMalloc`/`cudaFree` overhead
- **Dynamic launch depth** - Automatically estimates optimal CPU/GPU split based on branching factor and available memory
- **Bitboard move generation** using magic bitboards for sliding pieces

## Building

Requirements:
- CUDA Toolkit (tested with CUDA 13.1)
- CMake 3.18+
- GPU with compute capability 8.0+ (e.g., RTX 3090, RTX 4090, RTX 6000 Blackwell)

```bash
cmake -B build
cmake --build build --config Release
```

To target a different GPU architecture, edit `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt`.

## File overview

| File | Description |
|---|---|
| `perft.cu` | Entry point, CLI parsing, flag handling |
| `launcher.cu` / `launcher.h` | GPU init, launch depth estimation, GPU+CPU perft launchers, TT allocation, OOM fallback |
| `perft_kernels.cu` | GPU kernels, host-driven BFS, upsweep, move generator init |
| `MoveGeneratorBitboard.h` | Bitboard-based legal move generation (~1950 lines) |
| `chess.h` | Core data structures (`QuadBitBoard`, `GameState`, `CMove`, magic entries) |
| `switches.h` | Compile-time flags (`BLOCK_SIZE`, `MIN_BLOCKS_PER_MP`, TT budgets, etc.) |
| `tt.h` | Transposition table structures (`TTEntry`, `TTTable`, `LosslessTT`, probe/store) |
| `zobrist.h` / `zobrist.cpp` | 128-bit Zobrist hashing |
| `uint128.h` | 128-bit unsigned integer for deep perft accumulation |
| `utils.h` / `util.cpp` | FEN parsing, board display, Timer class |
| `GlobalVars.cpp` | Magic tables, attack tables |
| `Magics.cpp` | Magic number generation for sliding piece attacks |
| `randoms.cpp` | Pre-generated random numbers for Zobrist hashing |

## License

MIT
