// ==================================================================
// CUME V2017
// Copyright 2015-2017 Jean-Michel RICHER
// License: BSD License
// Please see the README file for documentation
// ==================================================================

#ifndef CUME_H
#define CUME_H

// note: 
// - use __CUDACC__ to verify that nvcc is the compiler
// - use __CUDA_ARCH__ to get architecture (
//		-- 100 = compute_10
//		-- 110 = compute_11
//		-- 200 = compute_20
//  )
// for example:
// #if __CUDA_ARCH__ >= 200
// #else
// #endif

#include "cume_base.h"
#include "cume_devices.h"
#include "cume_kernel.h"
#include "cume_array.h"
#include "cume_pinned_array.h"
#include "cume_zero_copy_array.h"
#include "cume_matrix.h"
#include "cume_cpu_timer.h"
#include "cume_gpu_timer.h"

#endif

