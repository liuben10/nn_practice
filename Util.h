/*
 * Util.h
 *
 *  Created on: Dec 31, 2017
 *      Author: liuben10
 */
#include <iostream>
#include <fstream>
#include <vector>

#ifndef UTIL_H_
#define UTIL_H_

#include <stdlib.h>

using namespace std;

namespace sigmoid {

class Util {

public:
	static vector<float> initOneDimVector(int len, float in[]) {
		if (len == 0) {
			return vector<float>();
		}
		vector<float> res = vector<float>();
		for(int i = 0; i < len; i++) {
			res.push_back(in[i]);
		}
		return res;
	}

	static vector<vector<float> > initTwoDimVector(int row, int col, float in[][col]) {
		vector<vector<float> > twoDimVector = vector<vector<float> >();
		for(int i = 0; i < row; i++) {
			vector<float> row = vector<float>();
			for(int j = 0; j < col; j++) {
				row.push_back(in[i][j]);
			}
			twoDimVector.push_back(row);
		}
		return twoDimVector;
	}
};


}


#endif /* UTIL_H_ */
