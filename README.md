# perft_gpu

A GPU-accelerated chess [perft](https://www.chessprogramming.org/Perft) calculator using CUDA.

This is a heavily optimized rewrite of the [original perft_gpu](https://github.com/ankan-ban/perft_gpu) project, stripped down to single-GPU mode with a host-driven BFS approach and no transposition tables. The focus is on raw move generation throughput.

## Performance

Measured on a single NVIDIA A100 GPU:

| Position | Depth | Nodes | Speed |
|---|---|---|---|
| Starting position | 9 | 2,439,530,234,167 | ~250 billion nps |
| [Position 2](https://www.chessprogramming.org/Perft_Results#Position_2) (Kiwipete) | 7 | 374,190,009,323 | ~616 billion nps |
| [Position 3](https://www.chessprogramming.org/Perft_Results#Position_3) | 7 | 178,633,661 | - |

## How it works

The perft tree is explored using breadth-first search driven from the host. At each BFS level, CUDA kernels expand all positions at the current depth in parallel:

1. **Make moves + count children** - Apply each move and count the resulting legal moves
2. **Prefix scan** (CUB InclusiveSum) - Compute output offsets for the next level
3. **Merge-path interval expand** - Load-balanced mapping from child indices back to parent indices, ensuring all CTAs do equal work regardless of how many moves each position has
4. **Generate moves** - Write out child moves to the next level's buffer

This loop repeats until a configurable "launch depth", after which the remaining depth is counted on-GPU without materializing further levels.

### Key design choices

- **Host-driven BFS** with one kernel launch per phase per level (no CUDA Dynamic Parallelism)
- **Merge-path interval expand** (inspired by [moderngpu](https://github.com/moderngpu/moderngpu)) for perfect load balancing across CTAs
- **Bump allocator** for GPU memory - a single 3 GB pre-allocated buffer avoids per-level `cudaMalloc`/`cudaFree` overhead
- **Bitboard move generation** using magic bitboards for sliding pieces

## Building

Requirements:
- CUDA Toolkit (tested with CUDA 12.x)
- CMake 3.18+
- GPU with compute capability 8.0+ (e.g., A100, RTX 3090)

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
| `perft_bb.h` | Host-driven BFS implementation, GPU kernels, merge-path interval expand |
| `launcher.h` | GPU initialization, launch depth estimation, CPU perft fallback |
| `MoveGeneratorBitboard.h` | Bitboard-based legal move generation (~3400 lines) |
| `chess.h` | Board representation (`HexaBitBoardPosition`), move structures |
| `switches.h` | Compile-time constants (`BLOCK_SIZE`, `IE_VT`, memory sizes) |
| `utils.h` / `util.cpp` | Board display, FEN parsing, timer utilities |
| `GlobalVars.cpp` | Magic tables, attack tables |
| `Magics.cpp` | Magic number generation for sliding piece attacks |

## License

MIT
