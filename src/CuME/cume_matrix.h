// ==================================================================
// CUME V2017
// Copyright 2015-2017 Jean-Michel RICHER
// License: BSD License
// Please see the README file for documentation
// ==================================================================

#ifndef CUME_MATRIX_H
#define CUME_MATRIX_H

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Definition of a generic matrix (or 2D array) that handles data in 
// host and device memory. The array simplifies the allocation of data
// and the copy between host and device memory
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#include <cassert>
#include <numeric>
#include "cume.h"

namespace cume {

/**
 * generic matrix of fixed size
 * data are duplicated on host and device memory
 * <ul>
 * <li>use copy_to_host() method to transfer data from
 * device to host memory</li>
 * <li>use copy_to_device() method to transfer data from
 * host to device memory</li>
 * </ul>
 */
template<typename T>
class Matrix {
public:
	typedef T element_type;
	typedef Matrix<T> self;

	// data in central memory
	T *m_cpu_data;
	// data in global memory of GPU
	T *m_gpu_data;
	// number of elements of array
	size_t m_size;
	// number of rows
	size_t m_rows;
	// number of columns
	size_t m_cols;

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
	Matrix(size_t r, size_t c) : m_rows(r), m_cols(c) {
		assert((r > 0) && (c > 0));
		m_size = m_rows * m_cols;
		m_cpu_data = new T [ m_size ];
		cume_new_array_zero(m_gpu_data, T, m_size);
	}

	/**
	 * copy constructor: duplicate data in host and
	 * device memory
	 * @param obj existing array
	 */
	Matrix(const Matrix<T>& object) {
		m_size = object.m_size;
		m_rows = object.m_rows;
		m_cols = object.m_cols;
		m_cpu_data = new T [ m_size ];
		cume_new_array_zero(m_gpu_data, T, m_size);
		memcpy(m_cpu_data, object.m_cpu_data, m_size * sizeof(T));
		cume_check(	cudaMemcpy(m_gpu_data, object.m_gpu_data,
				m_size * sizeof(T), cudaMemcpyDeviceToDevice) );
	}

	/**
	 * overloading of assignment operator
	 * @param obj existing array
	 */
	Matrix<T>& operator=(const Matrix<T>& object) {
		if (&object != this) {
			delete [] m_cpu_data;
			cume_free(m_gpu_data);
			m_size = object.m_size;
			m_rows = object.m_rows;
			m_cols = object.m_cols;
			m_cpu_data = new T [ m_size ];
			cume_new_array_zero(m_gpu_data, T, m_size);
			memcpy(m_cpu_data, object.m_cpu_data, m_size * sizeof(T));
			cume_check(	cudaMemcpy(m_gpu_data, object.m_gpu_data,
					m_size * sizeof(T), cudaMemcpyDeviceToDevice) );
		}
		return *this;
	}

	/**
	 * destructor
	 */
	~Matrix() {
		delete [] m_cpu_data;
		cume_free(m_gpu_data);
	}

	/**
	 * update data on host memory (central memory): copy
	 * data on device to host
	 */
	void pull() {
		cume_pull(m_cpu_data, m_gpu_data, T, m_size);
	}

	/**
	 * update data on device memory (global memory of GPU) : copy
	 * data from host to device
	 */
	void push() {
		cume_push(m_gpu_data, m_cpu_data, T, m_size);
	}

	/**
	 * return address of data on device memory
	 */
	T *operator &() {
		return m_gpu_data;
	}
	 
	T *gpu_addr() {
		return m_gpu_data;
	}

	T *cpu_addr() {
		return m_cpu_data;
	}

	size_t size() {
		return m_size;
	}
	
	size_t rows() {
		return m_rows;
	}
	
	size_t cols() {
		return m_cols;
	}

	T& operator[](int n) {
		assert((n >= 0) && (n < m_size));
		return m_cpu_data[n];
	}
	
	void set(size_t y, size_t x, T value) {
		m_cpu_data[ y*m_cols + x ] = value;
	}
	
	T get(size_t y, size_t x) {
		return m_cpu_data[ y*m_cols + x ];
	}

	ostream& print(ostream& out) {
		for (size_t y=0; y<m_rows; ++y) {
			for (size_t x=0; x<m_cols; ++x) {
				out << m_cpu_data[ y*m_cols + x ] << " ";
			}
			out << endl;
		}
		out << endl;
		return out;
	}
	
	friend ostream& operator<<(ostream& out, Matrix<T>& object) {
		return object.print(out);
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
