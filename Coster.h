/*
 * Coster.h
 *
 *  Created on: Dec 30, 2017
 *      Author: liuben10
 */

#include <vector>
#include <math.h>

using namespace std;

#ifndef COSTER_H_
#define COSTER_H_

namespace sigmoid {

class Coster {

public:
	static vector<float> toBinary(int n) {
		int cop = n;
		vector<float> bin = vector<float>(10, 0);
		int pos = 0;
		while (cop > 0) {
			int andded = cop & 1;
			bin[pos] = andded;
			cop = cop >> 1;
			pos += 1;
		}
		return bin;
	}


	static float evaluate(vector<float> actual, vector<float> expected) {
		float totalSum = 0;
		int n = actual.size();
		for(int i = 0; i < actual.size(); i++) {
			totalSum += pow(expected[i] - actual[i], 2);
		}
		return totalSum / n;
	}
};

} /* namespace sigmoid */

#endif /* COSTER_H_ */
