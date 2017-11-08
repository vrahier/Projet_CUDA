/*
 * gpu_timer.h
 *
 *  Created on: Sep 15, 2017
 *      Author: richer
 */

#ifndef GPU_TIMER_H_
#define GPU_TIMER_H_

#include <cuda.h>
#include <iostream>
using namespace std;


class GPUTimer {
public:
	GPUTimer();
	~GPUTimer();

	void start();
	void stop();

	ostream& print(ostream& out);

	friend ostream& operator<<(ostream& out, GPUTimer& g) {
		return g.print(out);
	}

private:
	cudaEvent_t t_start, t_stop;
};


#endif /* GPU_TIMER_H_ */
