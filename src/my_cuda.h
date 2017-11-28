/*
 * my_cuda.h
 *
 *  Created on: Sep 15, 2017
 *      Author: richer
 */

#ifndef MY_CUDA_H_
#define MY_CUDA_H_

#include <cuda.h>

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define cuda_check(value) {     \
		cudaError_t err = value; \
		if (err != cudaSuccess) {       \
			cerr << endl; \
			cerr << "============================================\n"; \
			cerr << "Error: " << cudaGetErrorString(err) << " at line "; \
			cerr << __LINE__ << " in file " <<  __FILE__;   \
			cerr <<  endl; \
			exit(EXIT_FAILURE); \
		} \
}

#define cuda_check_kernel() { \
		cudaError_t err = cudaGetLastError(); \
		if (err != cudaSuccess)  { \
			cerr << endl; \
			cerr << "============================================\n"; \
			cerr << "Kernel Error: " << cudaGetErrorString(err) << " at line "; \
			cerr << __LINE__ << " in file " <<  __FILE__;   \
			cerr <<  endl; \
			exit(EXIT_FAILURE); \
		} \
}


#endif /* MY_CUDA_H_ */
