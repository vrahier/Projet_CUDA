#include "gpu_timer.h"


GPUTimer::GPUTimer() {
	cudaEventCreate(&t_start);
	cudaEventCreate(&t_stop);
}

GPUTimer::~GPUTimer() {
	cudaEventDestroy(t_start);
	cudaEventDestroy(t_stop);
}

void GPUTimer::start() {
	cudaEventRecord(t_start, NULL);
}

void GPUTimer::stop() {
	cudaEventRecord(t_stop, NULL);
	cudaEventSynchronize(t_stop);
}

ostream& GPUTimer::print(ostream& out) {
	float milli_seconds = 0.0f;
	cudaEventElapsedTime(&milli_seconds, t_start, t_stop);
	out << fixed;
	out << milli_seconds;
	return out;
}

