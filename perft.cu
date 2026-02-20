#include <cuda_runtime.h>
#include <string.h>
#include "utils.h"
#include "launcher.h"

int main(int argc, char *argv[])
{
    BoardPosition testBoard;

    // Check for -cpu flag anywhere in args
    bool cpuMode = false;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-cpu") == 0)
        {
            cpuMode = true;
            // Shift remaining args down to remove -cpu
            for (int j = i; j < argc - 1; j++)
                argv[j] = argv[j + 1];
            argc--;
            i--;
        }
    }

    if (!cpuMode)
    {
        int totalGPUs;
        cudaGetDeviceCount(&totalGPUs);
        printf("No of GPUs detected: %d\n", totalGPUs);

        if (totalGPUs == 0)
        {
            printf("No CUDA GPUs found. Exiting.\n");
            return 1;
        }

        // Initialize single GPU (device 0)
        initGPU(0);
    }

    initMoveGen();

    char fen[1024] = "";
    int maxDepth = 10;

    if (argc >= 3)
    {
        strcpy(fen, argv[1]);
        maxDepth = atoi(argv[2]);
    }
    else
    {
        printf("\nUsage: perft_gpu <fen> <depth> [<launchdepth>] [-cpu]\n");
        printf("\nAs no parameters were provided... running default test\n");
    }

    if (strlen(fen) > 5)
    {
        Utils::readFENString(fen, &testBoard);
    }
    else
    {
        Utils::readFENString("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &testBoard);
    }
    Utils::dispBoard(&testBoard);

    QuadBitBoard testBB;
    GameState testGS;
    uint8 rootColor = testBoard.chance;
    Utils::board088ToQuadBB(&testBB, &testGS, &testBoard);

    if (cpuMode)
    {
        printf("CPU mode\n");
        fflush(stdout);

        for (int depth = 1; depth <= maxDepth; depth++)
        {
            perftCPU(&testBB, &testGS, rootColor, depth);
            fflush(stdout);
        }
    }
    else
    {
        uint32 launchDepth = estimateLaunchDepth(&testBB, &testGS, rootColor);
        launchDepth = min(launchDepth, (uint32)11);

        // for best performance without GPU hash
        if (launchDepth < 6)
            launchDepth = 6;

        if (argc >= 4)
        {
            launchDepth = atoi(argv[3]);
        }

        if ((uint32)maxDepth < launchDepth)
        {
            launchDepth = maxDepth;
        }

        printf("Launch depth: %d\n", launchDepth);
        fflush(stdout);

        for (int depth = 1; depth <= maxDepth; depth++)
        {
            perftLauncher(&testBB, &testGS, rootColor, depth, launchDepth);
            fflush(stdout);
        }

        cudaFree(preAllocatedBufferHost);
        cudaDeviceReset();
    }

    return 0;
}
