#include <cuda_runtime.h>
#include <string.h>
#include "utils.h"
#include "launcher.h"

// TT budget overrides (defined in launcher.cu)
extern int g_deviceTTBudgetMB;
extern int g_hostTTBudgetMB;

// Launch depth override (-ld flag, 0 = auto)
static int g_launchDepthOverride = 0;

int main(int argc, char *argv[])
{
    // Parse flags from anywhere in args
    bool cpuMode = false;
    for (int i = 1; i < argc; i++)
    {
        bool consumed = false;
        if (strcmp(argv[i], "-cpu") == 0)
        {
            cpuMode = true;
            consumed = true;
        }
        else if (strcmp(argv[i], "-nott") == 0)
        {
            g_useTT = false;
            consumed = true;
        }
        else if (strcmp(argv[i], "-ld") == 0 && i + 1 < argc)
        {
            g_launchDepthOverride = atoi(argv[i + 1]);
            for (int j = i; j < argc - 2; j++) argv[j] = argv[j + 2];
            argc -= 2; i--; continue;
        }
        else if (strcmp(argv[i], "-dtt") == 0 && i + 1 < argc)
        {
            g_deviceTTBudgetMB = atoi(argv[i + 1]);
            // Remove both -dtt and value
            for (int j = i; j < argc - 2; j++) argv[j] = argv[j + 2];
            argc -= 2; i--; continue;
        }
        else if (strcmp(argv[i], "-htt") == 0 && i + 1 < argc)
        {
            g_hostTTBudgetMB = atoi(argv[i + 1]);
            for (int j = i; j < argc - 2; j++) argv[j] = argv[j + 2];
            argc -= 2; i--; continue;
        }

        if (consumed)
        {
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
        printf("\nUsage: perft_gpu <fen> <depth> [-nott] [-dtt <MB>] [-htt <MB>] [-ld <N>] [-cpu]\n");
        printf("\nAs no parameters were provided... running default test\n");
    }

    QuadBitBoard testBB;
    GameState testGS;
    uint8 rootColor;

    if (strlen(fen) > 5)
        readFENString(fen, &testBB, &testGS, &rootColor);
    else
        readFENString("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &testBB, &testGS, &rootColor);

    if (cpuMode)
    {
        printf("CPU mode\n");
        fflush(stdout);

        // CPU mode: all TTs are host-side, starting from depth 2
        initTT(2, 2, maxDepth, 30.0f);

        for (int depth = 1; depth <= maxDepth; depth++)
        {
            perftCPU(&testBB, &testGS, rootColor, depth);
            printTTStats();
            fflush(stdout);
        }

        freeTT();
    }
    else
    {
        float branchingFactor = 30.0f;
        uint32 launchDepth = estimateLaunchDepth(&testBB, &testGS, rootColor, &branchingFactor);
        launchDepth = min(launchDepth, (uint32)11);

        // for best performance without GPU hash
        if (launchDepth < 6)
            launchDepth = 6;

        if (g_launchDepthOverride > 0)
        {
            launchDepth = g_launchDepthOverride;
        }

        if ((uint32)maxDepth < launchDepth)
        {
            launchDepth = maxDepth;
        }

        // LD is fixed (never increases dynamically — root memory doesn't predict worst-case).
        // Can still decrease on OOM during an iteration.
        int currentLD = (int)launchDepth;

        printf("Launch depth: %d\n", currentLD);
        fflush(stdout);

        // Device TTs for depths 3 through launchDepth-1 (exact LD range, no overallocation)
        initTT((int)launchDepth, (int)launchDepth, maxDepth, branchingFactor);

        for (int depth = 1; depth <= maxDepth; depth++)
        {
            int effectiveLD = min(currentLD, depth);
            resetCallStats();
            setEffectiveLD(effectiveLD);
            perftLauncher(&testBB, &testGS, rootColor, depth, effectiveLD);
            printTTStats();
            fflush(stdout);

            // Adjust LD: only decrease on OOM, never increase.
            uint64 ooms = getOomFallbackCount();
            if (ooms > 0 && depth >= (int)launchDepth)
            {
                int newLD = getEffectiveLD();
                if (newLD != currentLD)
                {
                    printf("  >> Launch depth adjusted: %d -> %d (OOM fallbacks: %llu)\n",
                           currentLD, newLD, (unsigned long long)ooms);
                    currentLD = newLD;
                }
            }
        }

        freeTT();

        cudaFree(preAllocatedBufferHost);
        cudaDeviceReset();
    }

    return 0;
}
