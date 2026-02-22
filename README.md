# perft_gpu

A GPU-accelerated chess [perft](https://www.chessprogramming.org/Perft) calculator using CUDA.

This is a heavily optimized rewrite of the [original perft_gpu](https://github.com/ankan-ban/perft_gpu) project, stripped down to single-GPU mode with a host-driven BFS approach. The focus is on raw move generation throughput, with optional transposition tables for deep perft.

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

### NVIDIA RTX 4090 (no transposition tables)

| Position | Depth | Nodes | Time | Speed |
|---|---|---|---|---|
| Starting position | 9 | 2,439,530,234,167 | 3.34s | ~729 billion nps |
| Starting position | 10 | 69,352,859,712,417 | 97.4s | ~712 billion nps |
| [Position 2](https://www.chessprogramming.org/Perft_Results#Position_2) (Kiwipete) | 7 | 374,190,009,323 | 0.34s | ~1,103 billion nps |
| [Position 3](https://www.chessprogramming.org/Perft_Results#Position_3) | 7 | 178,633,661 | - | - |

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
- CUDA Toolkit (tested with CUDA 12.x)
- CMake 3.18+
- GPU with compute capability 8.0+ (e.g., RTX 3090, RTX 4090, RTX 6000 Blackwell)

```bash
cmake -B build
cmake --build build --config Release
```

To target a different GPU architecture, edit `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt`.

## Usage

```bash
# Default: starting position, depth 10
./build/Release/perft_gpu

# Custom FEN and depth
./build/Release/perft_gpu "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" 9

# With explicit launch depth (controls BFS/DFS switchover)
./build/Release/perft_gpu "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -" 7 6
```

The program prints cumulative perft results for depth 1 through the specified maximum depth.

## File overview

| File | Description |
|---|---|
| `perft.cu` | Entry point, GPU init, FEN parsing, main loop |
| `perft_kernels.cu` | GPU kernels, host-driven BFS, fused 2-level leaf |
| `launcher.cu` | GPU initialization, launch depth estimation, CPU perft fallback |
| `MoveGeneratorBitboard.h` | Bitboard-based legal move generation |
| `chess.h` | Board representation (`QuadBitBoard`, `GameState`), move structures |
| `switches.h` | Compile-time constants (`BLOCK_SIZE`, memory sizes, lookup table options) |
| `utils.h` / `util.cpp` | Board display, FEN parsing, timer utilities |
| `GlobalVars.cpp` | Magic tables, attack tables |
| `Magics.cpp` | Magic number generation for sliding piece attacks |

## License

MIT
