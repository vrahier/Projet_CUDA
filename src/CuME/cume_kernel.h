// ==================================================================
// CUME V2017
// Copyright 2015-2017 Jean-Michel RICHER
// License: BSD License
// Please see the README file for documentation
// ==================================================================

#ifndef CUME_KERNEL_H 
#define CUME_KERNEL_H 

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// The cume_kernel.h header file contains template functions and 
// classes defined in order to simplify the use of kernel calls and 
// their parameters (grid, block, shared memory and stream).
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#include <cuda.h>
#include <cstdarg>
#include "cume_base.h"
#include "cume_devices.h"

namespace cume {

// ------------------------------------------------------------------
// definition of a macro instruction that runs a kernel with given
// parameters but with No Resource (NR)
// - name is the name of the kernel
// - kc is an instance of the KernelConfig class
// ------------------------------------------------------------------
#define kernel_call_no_resource(name, kc, ...) \
	cume::GPUTimer kernel_timer; \
	cume::Devices::get_instance().select(kc.get_device_id()); \
	if (kc.is_timer_needed()) { \
		kernel_timer.start(); \
	} \
	name <<< kc.get_grid(), kc.get_block(), kc.get_shared(), kc.get_stream() >>>\
            (__VA_ARGS__); \
    cume_check_kernel(); \
    if (kc.is_timer_needed()) { \
		kernel_timer.stop(); \
		cout << "kernel execution time = " << kernel_timer << endl; \
	}                                           

// ------------------------------------------------------------------
// definition of a macro instruction that runs a kernel with given
// parameters but with Resource (WR)
// - name is the name of the kernel
// - kc is an instance of the KernelConfig class
// ------------------------------------------------------------------
#define kernel_call(name, kc, ...) \
	cume::GPUTimer kernel_timer; \
	cume::Devices::get_instance().select(kc.get_device_id()); \
	if (kc.is_timer_needed()) { \
		kernel_timer.start(); \
	} \
	name <<< kc.get_grid(), kc.get_block(), kc.get_shared(), kc.get_stream() >>>\
            (kc.get_resource(), __VA_ARGS__); \
    cume_check_kernel(); \
    if (kc.is_timer_needed()) { \
		kernel_timer.stop(); \
		cout << "kernel execution time = " << kernel_timer << endl; \
	}
	

// ------------------------------------------------------------------
// definition of a macro instructions that helps calculate the thread
// index inside a kernel
// ------------------------------------------------------------------

// grid of 1 block of x threads
#define cume_gtid_1_x() threadIdx.x

// grid of x blocks of 1 thread
#define cume_gtid_x_1() blockIdx.x

// grid of x by y blocks of 1 thread
#define cume_gtid_xy_1() (gridDim.x * blockIdx.y) + blockIdx.x

// grid of 1 block of x by y threads 
#define cume_gtid_1_xy() blockDim.x * threadIdx.y + threadIdx.x

// grid of x block of x' threads
#define cume_gtid_x_x() blockDim.x * blockIdx.x + threadIdx.x
  
// grid of x by y blocks of x' threads
#define cume_gtid_xy_x() (gridDim.x * blockDim.x) * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x

// grid of y blocks of x by y' threads
#define cume_gtid_x_xy() blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x

// grid of x by y blocks of x' by y' threads
#define cume_gtid_xy_xy() blockIdx.y * (blockDim.x * blockDim.y) * gridDim.x \
						+ blockIdx.x * (blockDim.x * blockDim.y) \
						+ threadIdx.y * blockDim.x + threadIdx.x

// grid of x by y by z blocks of 1 thread
#define cume_gtid_xyz_1() blockIdx.z * (gridDim.x * gridDim.y) \
						+ blockIdx.y * gridDim.x \
						+ blockIdx.x

// Grid mode
enum {
	GRID_1 = 1,      // (1,1,1)
	GRID_X = 2,      // (x,1,1)
	GRID_XY = 4,     // (x,y,1)
	GRID_XYZ = 8,    // (x,y,z) 
	GRID_GUESS = 16, // will determine how many grid.x blocks are needed (x,1,1)
	GRID_NO_CHECK = 32
};

// Block mode
enum {
	BLOCK_1 = 1,     // (1,1,1)
	BLOCK_X = 2,     // (x,1,1) 
	BLOCK_XY = 4,    // (x,y,1)
	BLOCK_XYZ = 8    // (x,y,z)
};

/**
 * definition of a class that will hande the parameters of a kernel
 * and setup the grid and blocks according to the number of threads
 * required
 */		
class Kernel {
public:
	
	// Kernel type (constant used with Resource structure)
	typedef enum {
		KERNEL_TYPE_NONE  = 0,
		KERNEL_TYPE_1_X   = 100 * GRID_1 + BLOCK_X ,
		KERNEL_TYPE_X_1   = 100 * GRID_X + BLOCK_1,
		KERNEL_TYPE_XY_1  = 100 * GRID_XY + BLOCK_1,
		KERNEL_TYPE_1_XY  = 100 * GRID_1 + BLOCK_XY,
		KERNEL_TYPE_X_X   = 100 * GRID_X + BLOCK_X,
		KERNEL_TYPE_XY_X  = 100 * GRID_XY + BLOCK_X,
		KERNEL_TYPE_X_XY  = 100 * GRID_X + BLOCK_XY,
		KERNEL_TYPE_XY_XY = 100 * GRID_XY + BLOCK_XY,
		KERNEL_TYPE_XYZ_1 = 100 * GRID_XYZ + BLOCK_1
	} KernelType_t;

	typedef struct Coordinate {
		int x, y, offset;
		__device__ Coordinate(int _x, int _y, int cols)  :x(_x), y(_y)  {
			offset = _y * cols + _x;
		}
	} Coordinate;
	
	/**
	 * structure passed as first argument of kernel when
	 * the macro instruction KERNEL_EXECUTE_WITH_RESOURCE
	 * is ued. It will give the exact global thread index 
	 * and local thread index (inside a block) in function 
	 * of the kernel type
	 */
	struct Resource {
		//
		KernelType_t kernel_type;
	
		/**
		 * return global thread index
		 */
		__device__ int get_global_tid() {
			switch(kernel_type) {
				case KERNEL_TYPE_NONE:
					return 0;
				
				case KERNEL_TYPE_1_X: 
					return  cume_gtid_1_x();
				
				case KERNEL_TYPE_X_1: 
					return  cume_gtid_x_1();
				
				case KERNEL_TYPE_XY_1: 
					return  cume_gtid_xy_1();
			
				case KERNEL_TYPE_1_XY: 
					return  cume_gtid_1_xy();	
				
				case KERNEL_TYPE_X_X: 
					return  cume_gtid_x_x();
				
				case KERNEL_TYPE_XY_X: 
					return  cume_gtid_xy_x();
				
				case KERNEL_TYPE_X_XY: 
					return  cume_gtid_x_xy();
				
				case KERNEL_TYPE_XY_XY: 
					return  cume_gtid_xy_xy();
					
				case KERNEL_TYPE_XYZ_1:
					return cume_gtid_xyz_1();									
			}
			return 0;
		}
	
		/**
		 * return local thread index (within a block)
		 */
		__device__ int get_local_tid() {
			switch(kernel_type) {
				case KERNEL_TYPE_NONE:
					return 0;
				
				case KERNEL_TYPE_1_X: 
					return  threadIdx.x;
				
				case KERNEL_TYPE_X_1: 
					return  0;
				
				case KERNEL_TYPE_XY_1: 
					return  0;
			
				case KERNEL_TYPE_1_XY: 
					return  threadIdx.y * blockDim.x + threadIdx.x;	
				
				case KERNEL_TYPE_X_X: 
					return  threadIdx.x;
				
				case KERNEL_TYPE_XY_X: 
					return  threadIdx.x;
				
				case KERNEL_TYPE_X_XY: 
					return  threadIdx.y * blockDim.x + threadIdx.x;
				
				case KERNEL_TYPE_XY_XY: 
					return  threadIdx.y * blockDim.x + threadIdx.x;	
					
				case KERNEL_TYPE_XYZ_1:
					return 0;								
			}
			return 0;
		}
		
		__device__ Coordinate get_coordinate() {
			switch(kernel_type) {
				case KERNEL_TYPE_1_XY:
					return Coordinate(threadIdx.x, threadIdx.y,
						blockDim.x);
				case KERNEL_TYPE_XY_1: 
					return 	Coordinate(blockIdx.x, blockIdx.y,
						gridDim.x);
				case KERNEL_TYPE_XY_XY: 
					return Coordinate(blockIdx.x * blockDim.x + threadIdx.x,
						blockIdx.y * blockDim.y + threadIdx.y,
						blockDim.x * gridDim.x);
				default:
					return Coordinate(0,0,0);					
			};
		}
		
	};	
		
protected:
	// grid and block definition
	dim3 grid, block;
	// shared memory used by kernel
	int shared;
	// stream used for execution
	cudaStream_t stream;	
	// number of threads to handle
	size_t required_threads;
	// device id on which the kernel will be run
	int device_id;
	// is timer needed ?
	bool timer_needed;
	// resource in host memory
	Resource *cpu_resource;
	// copy of host resource in device memory
	Resource *gpu_resource;

public:	
	/**
	 * constructor with number of threads to handle
	 * @param nbt number of threads required
	 * @param dev_id identifier of device to perform calculation
	 */
	Kernel(size_t nbt, int dev_id = Devices::DEVICE_0);
	
	/**
	 * copy constructor
	 */
	Kernel(const Kernel& obj);
	
	/**
	 * assignment operator overloading
	 */
	Kernel& operator=(const Kernel& obj);
	
	/**
	 * destructor
	 */
	~Kernel();
		
	/**
	 * function to call after construction of an instance
	 * of KernelConfig in order to setup the grid and block
	 * dimensions. First the grid and block mode/type are
	 * set followed by the dimensions
	 * @param gty grid type (X, XY, XYZ or GUESS)
	 * @param gty grid type (X, XY or XYZ)
	 * @param dimensions
	 * For example to have a grid of 4 blocks, each block
	 * having 32 by 8 threads, use :
	 * set_config(KernelConfig::GRID_X, KernelConfig::GRID_XY, 4, 32, 8)
	 * then grid.x = 4, block.x = 32, block.y = 8
	 *
	 * Use GRID_GUESS to let the method determine how many blocks are
	 * required and provide only information for the block:
	 * KernelConfig kcfg(1024);
	 * kcfg.set_config(KernelConfig::GRID_GUESS, KernelConfig::GRID_X, 128)
	 * then grid.x = 1024/128 = 8
	 * @param gmod grid mode
	 * @param bmod block mode
	 */
	void configure(int gmod, int bmod, ...) ;
	
	/**
	 * return identifier of device on which computations
	 * are performed
	 */
	int get_device_id() {
		return device_id;
	}
	
	void set_timer_needed(bool value=true) {
		timer_needed = value;
	}
	
	bool is_timer_needed() {
		return timer_needed;
	}
	
	/**
	 * return pointer to resource in device memory
	 */
	Resource *get_resource() {
		return gpu_resource;
	}
	
	/**
	 * return grid dimensions
	 */
	dim3 get_grid() {
		return grid;
	}	
	
	/**
	 * return block dimension
	 */
	dim3 get_block() {
		return block;
	}
	
	/**
	 * return amount of shared memory required
	 */
	int get_shared() {
		return shared;
	}
	
	/**
	 * set amount of shared memory required
	 */
	void set_shared(int shr) {
		shared = shr;
	}
	
	/**
	 * return stream used
	 */
	cudaStream_t get_stream() {
		return stream;
	}
	
	/**
	 * set stream used
	 */
	void set_stream(cudaStream_t s) {
		stream = s;
	}
	
	/**
	 * print information about the configuration of the kernel call
	 */
	ostream& print(ostream& out) ;
	
	friend ostream& operator<<(ostream& out, Kernel& obj) {
		return obj.print(out);
	}
};

} // end of namespace

#endif


