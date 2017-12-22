cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER g++)

set(CUFLAGS
        --optimize 2  -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -D_FORCE_INLINES -std=c++11
)

set(IEEE_COMPLIANCE
        -ftz=false -prec-div=true -prec-sqrt=true
)

set(CUDA_NVCC_FLAGS
${CUFLAGS} ${IEEE_COMPLIANCE}
)


set(SATISFY_GPU_SOURCES
        src/solver.cu
        src/gpu_parse_file.cu
    )

set(CUME_SRC
        src/CuME/cume_base.cu
        src/CuME/cume_cpu_timer.cpp
        src/CuME/cume_devices.cu
        src/CuME/cume_gpu_timer.cu
        src/CuME/cume_kernel.cu
    )

set(SATISFY_CPU_SOURCES
        src/cpu_timer.cpp
        src/cpu_timer.h
        src/Solver.cpp
        src/cpu_test.cpp
        )


add_library(File
        src/Utilities/file.cpp
        )

add_executable(Cpu_satisfy ${SATISFY_CPU_SOURCES})

target_link_libraries(Cpu_satisfy File)

cuda_add_library(Cume ${CUME_SRC})

cuda_add_executable(Gpu_satisfy ${SATISFY_GPU_SOURCES})

target_link_libraries(Gpu_satisfy Cume File)