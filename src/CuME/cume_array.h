// ==================================================================
// CUME V2017
// Copyright 2015-2017 Jean-Michel RICHER
// License: BSD License
// Please see the README file for documentation
// ==================================================================

#ifndef CUME_ARRAY_H
#define CUME_ARRAY_H

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
class Array {
public:
	typedef T element_type;
	typedef Array<T> self;

	// data in central memory
	T *m_cpu_data;
	// data in global memory of GPU
	T *m_gpu_data;
	// number of elements of array
	size_t m_size;
	// number of elements per row when we print the array 
	size_t m_elements_per_row;

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
	Array(size_t size) : m_size(size) {
		assert(size > 0);
		m_cpu_data = new T [ size ];
		cume_new_array_zero(m_gpu_data, T, m_size);
		m_elements_per_row = 10;
	}

	/**
	 * constructor with number of elements and initial value
	 * @param sz number of elements of the array
	 * @param value value to initialize all elements
	 */
	Array(size_t size, T value) : m_size(size) {
		assert(size > 0);
		m_cpu_data = new T [ m_size ];
		std::fill(&m_cpu_data[0], &m_cpu_data[m_size], value);
		cume_new_array(m_gpu_data, T, m_size);
		cume_new_array_zero(m_gpu_data, T, m_size);
		m_elements_per_row = 10;
	}

	Array(const self& object) {
		m_cpu_data = new T [ m_size = object.m_size ];
		std::copy(&object.m_cpu_data[0], &object.m_cpu_data[object.m_size], &m_cpu_data[0]);
		cume_new_array(m_gpu_data, T, m_size);
		cume_copy(m_gpu_data, object.m_gpu_data, T, m_size);
		m_elements_per_row = object.m_elements_per_row;
	}

	/**
	 * destructor
	 */
	~Array() {
		delete [] m_cpu_data;
		cume_free(m_gpu_data);
	}

	self& operator=(const self& object) {
		if (&object != this) {
			delete [] m_cpu_data;
			cume_free(m_gpu_data);
			m_cpu_data = new T [ m_size = object.m_size ];
			std::copy(&object.m_cpu_data[0], &object.m_cpu_data[object.m_size], &m_cpu_data[0]);
			cume_new_array(m_gpu_data, T, m_size);
			cume_copy(m_gpu_data, object.m_gpu_data, T, m_size);
			m_elements_per_row = object.m_elements_per_row;
		}
		return *this;
	}

	/**
	 * update data on device memory (global memory of GPU) : copy
	 * data from host to device
	 */
	void push() {
		cume_push(m_gpu_data, m_cpu_data, T, m_size);
	}

	/**
	 * update data on host memory (central memory): copy
	 * data on device to host
	 */
	void pull() {
		cume_pull(m_cpu_data, m_gpu_data, T, m_size);
	}


	T *operator &() {
		return m_gpu_data;
	}

	/**
	 * return address of data on device memory
	 */
	T *gpu_addr() {
		return m_gpu_data;
	}

	T *cpu_addr() {
		return m_cpu_data;
	}

	size_t size() {
		return m_size;
	}

	T& operator[](int n) {
		assert((n >= 0) && (n < m_size));
		return m_cpu_data[n];
	}

	void set_elements_per_row(size_t s) {
		m_elements_per_row = s;
	}

	ostream& print(ostream& out) {
		for (size_t i=0; i<m_size; ++i) {
			out << m_cpu_data[i] << " ";
			if (((i+1) % m_elements_per_row) == 0) out << endl;
		}
		out << endl;
		return out;
	}

	ostream& print_range(ostream& out, size_t l, size_t h) {
		size_t k=0;
		assert((l < h) && (h <= m_size));
		for (size_t i=l; i<h; ++i, ++k) {
			out << m_cpu_data[i] << " ";
			if (((k+1) % m_elements_per_row) == 0) out << endl;
		}
		out << endl;
		return out;
	}

	friend ostream& operator<<(ostream& out, Array<T>& obj) {
		return obj.print(out);
	}

	iterator begin() {
		return &m_cpu_data[0];
	}

	iterator end() {
		return &m_cpu_data[m_size];
	}

};


} // end of namespace

#endif 
