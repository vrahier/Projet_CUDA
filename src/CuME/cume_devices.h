// ==================================================================
// CUME V2017
// Copyright 2015-2017 Jean-Michel RICHER
// License: BSD License
// Please see the README file for documentation
// ==================================================================

#ifndef CUME_DEVICES_H
#define CUME_DEVICES_H

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Definition of a class to handle GPU (device) characteristics
// features
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#include <cuda.h>
#include "cume_base.h"

namespace cume {

/**
 * The Devices class is implemented as a singleton design
 * pattern and helps get information and select the devices
 */
class Devices {
protected:
	/**
	 * number of devices on computer
	 */
	int devices_count;
        
	/**
	 * array of information about each device
	 */
	cudaDeviceProp *devices;

	static Devices *instance;
		
public:
	enum {
		DEVICE_0 = 0,
		DEVICE_1 = 1,
		DEVICE_2 = 2,
		DEVICE_3 = 3
	};

	static Devices& get_instance();
		
	~Devices() ;
		
	/**
	 * @return number of devices
	 */
	int get_count() {
		return devices_count;
	}
	
	void select(int device_id);
    
	cudaDeviceProp *get_device_properties(int device_id);
	
	void memory_report(ostream& out);
	
    ostream& print(ostream& out);
    
    /**
     * display information for each device on stream
     */
    friend ostream& operator<<(ostream& out, Devices& obj) {
    	return obj.print(out);
    }

protected:	
	Devices();

};

} // end of namespace

#endif

