/*
 * Matrix.h
 *
 *  Created on: Jan 2, 2018
 *      Author: liuben10
 */

#include <vector>
#include <boost/multiprecision/cpp_dec_float.hpp>

using namespace std;
using namespace boost::multiprecision;


#ifndef MATRIX_H_
#define MATRIX_H_

namespace sigmoid {

class Matrix {
public:
	static vector<cpp_dec_float_100 > hadamard(vector<cpp_dec_float_100> a, vector<cpp_dec_float_100> b) {
		vector<cpp_dec_float_100> result = vector<cpp_dec_float_100>(a.size(), 0);
		for(int i = 0; i < a.size(); i++) {
			result[i] = a[i] * b[i];
		}
		return result;
	}

	static vector<cpp_dec_float_100> sum(vector<cpp_dec_float_100> a, vector<cpp_dec_float_100> b) {
		vector<cpp_dec_float_100> result = vector<cpp_dec_float_100>(a.size(), 0);
		for(int i = 0; i < a.size(); i++) {
			result[i] = a[i] * b[i];
		}
		return result;
	}

	static vector<vector<cpp_dec_float_100> > transpose(vector<vector<cpp_dec_float_100> > toTranspose) {
		int rows = toTranspose.size();
		int cols = toTranspose[0].size();

		vector<vector<cpp_dec_float_100> > resultTranspose = vector<vector<cpp_dec_float_100> >();
		for(int i = 0 ; i < cols; i++) {
			vector<cpp_dec_float_100> curRow = vector<cpp_dec_float_100>();
			for(int j = 0; j < rows; j++) {
				curRow.push_back(toTranspose[rows][cols]);
			}
			resultTranspose.push_back(curRow);
		}
		return resultTranspose;
	}

	static vector<cpp_dec_float_100> transposeAndMultiplyOneDim(vector<vector<cpp_dec_float_100> > a, vector<cpp_dec_float_100> b) {
		int m1 = a.size();
		int n1 = a[0].size();
		printf("\na_size=%d a_row_size=%d, b_size=%d\n", a.size(), n1, b.size());
		vector<cpp_dec_float_100> result = vector<cpp_dec_float_100>(m1, 0);
		for(int i = 0; i < m1; i++) {
			for(int j = 0; j < n1; j++) {
				cpp_dec_float_100 prod = a[i][j] * b[j];
//				printf("\nproduct=%f of a=%f, b=%f\n", prod, a[i][j], b[j]);
				result[i] += prod;
			}
		}
		return result;
	}

	static vector<vector<cpp_dec_float_100> > transposeAndMultiply(vector<cpp_dec_float_100> a, vector<cpp_dec_float_100> b) {
		int m1 = a.size();
		int n2 = b.size();
		vector<vector<cpp_dec_float_100> > result = vector<vector<cpp_dec_float_100> >(m1, vector<cpp_dec_float_100>(n2, 0));
		for(int i = 0; i < m1; i++) {
			for(int k = 0; k < n2; k++) {
				result[i][k] += a[i] * b[k];
			}
		}
		return result;
	}

	static vector<vector<cpp_dec_float_100> > matrixMultiply(vector<vector<cpp_dec_float_100> > a, vector<vector<cpp_dec_float_100> > b) {
		int m1 = a.size();
		int n1 = a[0].size();
		int n2 = b[0].size();
		vector<vector<cpp_dec_float_100> > result = vector<vector<cpp_dec_float_100> >(m1, vector<cpp_dec_float_100>(n2, 0));
		for(int i = 0; i < m1; i++) {
			for(int j = 0; j < n1; j++) {
				for(int k = 0; k < n2; k++) {
					result[i][k] += a[i][j] * b[j][k];
				}
			}
		}
		return result;
		}
	}; /* namespace sigmoid */
}

#endif /* MATRIX_H_ */
