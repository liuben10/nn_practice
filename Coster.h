/*
 * Coster.h
 *
 *  Created on: Dec 30, 2017
 *      Author: liuben10
 */
#include <boost/multiprecision/cpp_dec_float.hpp>

#include <vector>
#include <math.h>

using namespace std;
using namespace boost::multiprecision;

#ifndef COSTER_H_
#define COSTER_H_

namespace sigmoid {

class Coster {

public:
	static vector<cpp_dec_float_100> toBinary(int n) {
		int cop = n;
		vector<cpp_dec_float_100> bin = vector<cpp_dec_float_100>(10, 0);
		int pos = 0;
		while (cop > 0) {
			int andded = cop & 1;
			bin[pos] = andded;
			cop = cop >> 1;
			pos += 1;
		}
		return bin;
	}


	static cpp_dec_float_100 evaluate(vector<cpp_dec_float_100> actual, vector<cpp_dec_float_100> expected) {
		cpp_dec_float_100 totalSum = 0;
		int n = actual.size();
		for(int i = 0; i < actual.size(); i++) {
			totalSum += pow(expected[i] - actual[i], 2);
		}
		return totalSum / n;
	}
};

} /* namespace sigmoid */

#endif /* COSTER_H_ */
