/*
 * WeightsAndBiasUpdates.h
 *
 *  Created on: Jan 4, 2018
 *      Author: liuben10
 */
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <vector>

using namespace std;
using namespace boost::multiprecision;


#ifndef WEIGHTSANDBIASUPDATES_H_
#define WEIGHTSANDBIASUPDATES_H_

namespace sigmoid {

class WeightsAndBiasUpdates {
private:
	vector<vector<vector<cpp_dec_float_100> > > weightUpdates;
	vector<vector<cpp_dec_float_100> > biasUpdates;
public:
	void addWeightUpdate(vector<vector<cpp_dec_float_100> > weightUpdate);
	void addBiasUpdate(vector<cpp_dec_float_100> biasUpdate);
	WeightsAndBiasUpdates();
	string toString();
	virtual ~WeightsAndBiasUpdates();
};

} /* namespace sigmoid */

#endif /* WEIGHTSANDBIASUPDATES_H_ */
