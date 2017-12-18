// ==================================================================
// CUME V2017
// Copyright 2015-2017 Jean-Michel RICHER
// License: BSD License
// Please see the README file for documentation
// ==================================================================

#ifndef CUME_CPU_TIMER_H
#define CUME_CPU_TIMER_H

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Simplification of the usage of timer for CPU
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#include <stdint.h>
#include <iostream>
#include <math.h>
#include <chrono>
#include <iomanip>
using namespace std;

namespace cume {

/**
 * class used to measure performances of algorithms.
 * <ul>
 * 	<li>use start() method to start timer</li>
 * 	<li>use stop() method to stop</li>
 * 	<li>use get_seconds() method to obtain the number of seconds
 * 	between start and stop</li>
 * 	</ul>
 */
class CPUTimer {
public:
	typedef std::chrono::time_point<std::chrono::high_resolution_clock> Time;
	typedef std::chrono::milliseconds millis;

	/**
	 * default constructor
	 */
	CPUTimer();

	/**
	 * start chrono and record clocks
	 */
	void start();

	/**
	 * stop chrono and record clocks
	 */
	void stop();


	long int get_milli_seconds();

	/**
	 *
	 */
	ostream& print(ostream& out);

	/**
	 *
	 */
	friend ostream& operator<<(ostream& out, CPUTimer& c) {
		return c.print(out);
	}

private:
	Time _start, _stop;
};

} // end of namespace

#endif

