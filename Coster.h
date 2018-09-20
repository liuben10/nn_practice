/*
 * Coster.h
 *
 *  Created on: Dec 30, 2017
 *      Author: liuben10
 */
#include <boost/multiprecision/cpp_dec_float.hpp>

#include "Matrix.h"
#include <vector>
#include <math.h>

using namespace std;
using namespace boost::multiprecision;

#ifndef COSTER_H_
#define COSTER_H_

namespace sigmoid {

class Coster {

public:
	static MATRIX toBinary(int n) {
		int cop = n;
		MATRIX bin = MATRIX(10, ROW(1, 0));
		int pos = 0;
		while (cop > 0) {
			int andded = cop & 1;
			bin[pos][0] = andded;
			cop = cop >> 1;
			pos += 1;
		}
		return bin;
	}


	static double evaluate(MATRIX actual, MATRIX expected) {
		double totalSum = 0;
		int n = actual.size();
		for(int i = 0; i < actual.size(); i++) {
			totalSum += pow(expected[i][0] - actual[i][0], 2);
		}
		return totalSum / n;
	}
};

} /* namespace sigmoid */

#endif /* COSTER_H_ */
