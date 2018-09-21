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
	static vector<double> initOneDimVector(int len, double in[]) {
		if (len == 0) {
			return vector<double>();
		}
		vector<double> res = vector<double>();
		for(int i = 0; i < len; i++) {
			res.push_back(in[i]);
		}
		return res;
	}

	static vector<vector<double> > initTwoDimVector(int row, int col, double in[][col]) {
		vector<vector<double> > twoDimVector = vector<vector<double> >();
		for(int i = 0; i < row; i++) {
			vector<double> row = vector<double>();
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
