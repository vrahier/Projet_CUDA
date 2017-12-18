// ==================================================================
// CUME V2017
// Copyright 2015-2017 Jean-Michel RICHER
// License: BSD License
// Please see the README file for documentation
// ==================================================================

#ifndef CUME_GPU_TIMER_H
#define CUME_GPU_TIMER_H

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Simplification of the usage of cudaEvents
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#include <cuda.h>
#include <iostream>
using namespace std;
#include <time.h>

namespace cume {

class GPUTimer {
protected:
	cudaEvent_t t_start, t_stop;

public:
	/**
	 * constructor with no arguments
	 */ 
	GPUTimer();
	
	/**
	 * destructor
	 */
	~GPUTimer();
	
	/**
	 * start timer
	 */
	void start();
	/**
	 * stop timer
	 */
	void stop();
	
	/**
	 * print timer difference in milliseconds
	 */
	ostream& print(ostream& out);

	friend ostream& operator<<(ostream& out, GPUTimer& obj) {
		return obj.print(out);	
	}

};


} // end of namespace

#endif

