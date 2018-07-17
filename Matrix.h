/*
 * Matrix.h
 *
 *  Created on: Jan 2, 2018
 *      Author: liuben10
 */

#include <vector>

using namespace std;

#ifndef MATRIX_H_
#define MATRIX_H_

namespace sigmoid {

class Matrix {
public:
	static vector<float > hadamard(vector<float> a, vector<float> b) {
		vector<float> result = vector<float>(a.size(), 0);
		for(int i = 0; i < a.size(); i++) {
			result[i] = a[i] * b[i];
		}
		return result;
	}

	static vector<float> sum(vector<float> a, vector<float> b) {
		vector<float> result = vector<float>(a.size(), 0);
		for(int i = 0; i < a.size(); i++) {
			result[i] = a[i] * b[i];
		}
		return result;
	}

	static vector<vector<float> > transpose(vector<vector<float> > toTranspose) {
		int rows = toTranspose.size();
		int cols = toTranspose[0].size();

		vector<vector<float> > resultTranspose = vector<vector<float> >();
		for(int i = 0 ; i < cols; i++) {
			vector<float> curRow = vector<float>();
			for(int j = 0; j < rows; j++) {
				curRow.push_back(toTranspose[rows][cols]);
			}
			resultTranspose.push_back(curRow);
		}
		return resultTranspose;
	}

	static vector<float> transposeAndMultiplyOneDim(vector<vector<float> > a, vector<float> b) {
		int m1 = a.size();
		int n1 = a[0].size();
		printf("\na_size=%d a_row_size=%d, b_size=%d\n", a.size(), n1, b.size());
		vector<float> result = vector<float>(m1, 0);
		for(int i = 0; i < m1; i++) {
			for(int j = 0; j < n1; j++) {
				float prod = a[i][j] * b[j];
//				printf("\nproduct=%f of a=%f, b=%f\n", prod, a[i][j], b[j]);
				result[i] += prod;
			}
		}
		return result;
	}

	static vector<vector<float> > transposeAndMultiply(vector<float> a, vector<float> b) {
		int m1 = a.size();
		int n2 = b.size();
		vector<vector<float> > result = vector<vector<float> >(m1, vector<float>(n2, 0));
		for(int i = 0; i < m1; i++) {
			for(int k = 0; k < n2; k++) {
				result[i][k] += a[i] * b[k];
			}
		}
		return result;
	}

	static vector<vector<float> > matrixMultiply(vector<vector<float> > a, vector<vector<float> > b) {
		int m1 = a.size();
		int n1 = a[0].size();
		int n2 = b[0].size();
		vector<vector<float> > result = vector<vector<float> >(m1, vector<float>(n2, 0));
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
