// ==================================================================
// CUME V2017
// Copyright 2015-2017 Jean-Michel RICHER
// License: BSD License
// Please see the README file for documentation
// ==================================================================

#ifndef CUME_ZERO_COPY_ARRAY_H
#define CUME_ZERO_COPY_ARRAY_H

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Definition of a generic array that handles data in host and 
// device memory. The array simplifies the allocation of data
// and the copy between host and device memory
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#include <cassert>
#include <numeric>
#include "cume.h"

namespace cume {

/**
 * generic array of fixed size
 * data are duplicated on host and device memory
 * <ul>
 * <li>use copy_to_host() method to transfer data from
 * device to host memory</li>
 * <li>use copy_to_device() method to transfer data from
 * host to device memory</li>
 * </ul>
 */
template<typename T>
class ZeroCopyArray {
protected:
	// data in central memory
	T *cpu_data;
	// data in global memory of GPU
	T *gpu_data;
	// number of elements of array
	size_t size;
	// number of elements per row when we print the array 
	size_t elements_per_row;
	
public:
	/**
	 * define iterator for this data structure to
	 * use with STL algorithms. Note that STL algorithms
	 * will only apply to data in central memory of CPU
	 */
	typedef T *iterator;

	/**
	 * constructor with number of elements
	 * @param sz number of elements of the array
	 */
	ZeroCopyArray(size_t sz) : size(sz) {
		assert(sz > 0);
		cpu_data = new T [ size ];
		cume_new_array_zero_copy(cpu_data, T, size);
		cudaHostGetDevicePointer( &gpu_data, cpu_data, 0 );
		elements_per_row = 10;
	}

	/**
	 * constructor with number of elements and initial value
	 * @param sz number of elements of the array
	 * @param value value to initialize all elements
	 */
	ZeroCopyArray(size_t sz, T value) : size(sz) {
		assert(sz > 0);
		cpu_data = new T [ size ];
		std::fill(&cpu_data[0], &cpu_data[size], value);
		cume_new_array_zero_copy(gpu_data, T, size);
		cudaHostGetDevicePointer( &gpu_data, cpu_data, 0 );
		std::fill(&cpu_data[0], &cpu_data[size], value);
		elements_per_row = 10;
	}
		
	/**
	 * destructor
	 */
	~ZeroCopyArray() {
		cume_free_host(cpu_data);
	}

	/**
	 * update data on host memory (central memory): copy
	 * data on device to host
	 */
	void pop() {
		//cume_push(cpu_data, gpu_data, T, size);
	}

	/**
	 * update data on device memory (global memory of GPU) : copy
	 * data from host to device
	 */
	void push() {
		//cume_push(gpu_data, cpu_data, T, size);
	}
	
	T *operator &() {
		return gpu_data;
	}
	
	/**
	 * return address of data on device memory
	 */
	T *get_daddr() {
		return gpu_data;
	}

	T *get_haddr() {
		return cpu_data;
	}

	size_t get_size() {
		return size;
	}

	T& operator[](int n) {
		assert((n >= 0) && (n < size));
		return cpu_data[n];
	}
	
	void set_elements_per_row(size_t s) {
		elements_per_row = s;
	}

	ostream& print(ostream& out) {
		for (size_t i=0; i<size; ++i) {
			out << cpu_data[i] << " ";
			if (((i+1) % elements_per_row) == 0) out << endl;
		}
		out << endl;
		return out;
	}
	
	ostream& print_range(ostream& out, size_t l, size_t h) {
		size_t k=0;
		assert((l < h) && (h <= size));
		for (size_t i=l; i<h; ++i, ++k) {
			out << cpu_data[i] << " ";
			if (((k+1) % elements_per_row) == 0) out << endl;
		}
		out << endl;
		return out;
	}
	
	friend ostream& operator<<(ostream& out, Array<T>& obj) {
		return obj.print(out);
	}

	iterator begin() {
		return &cpu_data[0];
	}

	iterator end() {
		return &cpu_data[size];
	}

	void iota(T value=0) {
		for (size_t i=0; i<size; ++i) {
			cpu_data[i] = value + i;
		} 
	}
	
};


} // end of namespace

#endif 
