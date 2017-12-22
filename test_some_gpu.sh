#!/usr/bin/env bash


nvprof --unified-memory-profiling off ./cmake-build-debug/Gpu_satisfy pb_pigeons_2_2.txt 2> data/gpu_2_2
nvprof --unified-memory-profiling off ./cmake-build-debug/Gpu_satisfy pb_pigeons_8_8.txt 2> data/gpu_8_8
nvprof --unified-memory-profiling off ./cmake-build-debug/Gpu_satisfy pb_pigeons_25_25.txt 2> data/gpu_25_25
nvprof --unified-memory-profiling off ./cmake-build-debug/Gpu_satisfy pb_pigeons_45_50.txt 2> data/gpu_45_50
nvprof --unified-memory-profiling off ./cmake-build-debug/Gpu_satisfy pb_pigeons_2_1.txt 2> data/gpu_2_1
nvprof --unified-memory-profiling off ./cmake-build-debug/Gpu_satisfy pb_pigeons_2_1.txt 2> data/gpu_4_3
nvprof --unified-memory-profiling off ./cmake-build-debug/Gpu_satisfy pb_pigeons_5_4.txt 2> data/gpu_5_4
#nvprof --unified-memory-profiling off ./cmake-build-debug/Gpu_satisfy pb_pigeons_9_8.txt 2> data/gpu_6_5