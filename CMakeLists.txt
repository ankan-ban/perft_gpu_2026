cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES "80;120" CACHE STRING "CUDA architectures" FORCE)
project(perft_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(perft_gpu
    perft.cu
    perft_kernels.cu
    launcher.cu
    util.cpp
    GlobalVars.cpp
    Magics.cpp
    zobrist.cpp
    randoms.cpp
)

target_include_directories(perft_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(perft_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
)

# No separable compilation needed (no CDP)
set_target_properties(perft_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
set_target_properties(perft_gpu PROPERTIES CUDA_ARCHITECTURES "80;120")
