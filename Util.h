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
	static vector<number<cpp_dec_float<300> >   > initOneDimVector(int len, number<cpp_dec_float<300> >    in[]) {
		if (len == 0) {
			return vector<number<cpp_dec_float<300> >   >();
		}
		vector<number<cpp_dec_float<300> >   > res = vector<number<cpp_dec_float<300> >   >();
		for(int i = 0; i < len; i++) {
			res.push_back(in[i]);
		}
		return res;
	}

	static vector<vector<number<cpp_dec_float<300> >   > > initTwoDimVector(int row, int col, number<cpp_dec_float<300> >    in[][col]) {
		vector<vector<number<cpp_dec_float<300> >   > > twoDimVector = vector<vector<number<cpp_dec_float<300> >   > >();
		for(int i = 0; i < row; i++) {
			vector<number<cpp_dec_float<300> >   > row = vector<number<cpp_dec_float<300> >   >();
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
