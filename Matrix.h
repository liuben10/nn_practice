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
	vector<float > hadamard(vector<float> a, vector<float> b) {
		vector<float> result = vector<float>(a.size(), 0);
		for(int i = 0; i < a.size(); i++) {
			result[i] = a[i] * b[i];
		}
		return result;
	}
	vector<float> sum(vector<float> a, vector<float> b) {
		vector<float> result = vector<float>(a.size(), 0);
		for(int i = 0; i < a.size(); i++) {
			result[i] = a[i] * b[i];
		}
		return result;
	}

	vector<vector<float>> transpose(vector<vector<float>> toTranspose) {

	}

	vector<vector<float> > matrixMultiply(vector<vector<float> > a, vector<vector<float> > b) {
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
} /* namespace sigmoid */

#endif /* MATRIX_H_ */
