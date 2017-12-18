#include "cume_gpu_timer.h"

using namespace cume;

GPUTimer::GPUTimer() {
	cudaEventCreate(&t_start);
	cudaEventCreate(&t_stop);
}

/**
 * destructor
 */
GPUTimer::~GPUTimer() {
	cudaEventDestroy(t_start);
	cudaEventDestroy(t_stop);
}

/**
 * start timer
 */
void GPUTimer::start() {
	cudaEventRecord(t_start, 0);
}

/**
 * stop timer
 */
void GPUTimer::stop() {
	cudaEventRecord(t_stop, 0);
	cudaEventSynchronize(t_stop);
}	

/**
 * print timer difference in milliseconds
 */
ostream& GPUTimer::print(ostream& out) {
	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, t_start, t_stop);
	out.setf(ios::fixed);
	out.precision(2);
	out << elapsed_time << "ms";
	return out;
}
