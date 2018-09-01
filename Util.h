/*
 * Util.h
 *
 *  Created on: Dec 31, 2017
 *      Author: liuben10
 */
#include <boost/multiprecision/cpp_dec_float.hpp>

#include <iostream>
#include <fstream>
#include <vector>

#ifndef UTIL_H_
#define UTIL_H_

#include <stdlib.h>

using namespace std;
using namespace boost::multiprecision;


namespace sigmoid {

class Util {

public:
	static vector<cpp_dec_float_100> initOneDimVector(int len, cpp_dec_float_100 in[]) {
		if (len == 0) {
			return vector<cpp_dec_float_100>();
		}
		vector<cpp_dec_float_100> res = vector<cpp_dec_float_100>();
		for(int i = 0; i < len; i++) {
			res.push_back(in[i]);
		}
		return res;
	}

	static vector<vector<cpp_dec_float_100> > initTwoDimVector(int row, int col, cpp_dec_float_100 in[][col]) {
		vector<vector<cpp_dec_float_100> > twoDimVector = vector<vector<cpp_dec_float_100> >();
		for(int i = 0; i < row; i++) {
			vector<cpp_dec_float_100> row = vector<cpp_dec_float_100>();
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
