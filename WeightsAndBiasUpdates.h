/*
 * WeightsAndBiasUpdates.h
 *
 *  Created on: Jan 4, 2018
 *      Author: liuben10
 */
#include <vector>

using namespace std;

#ifndef WEIGHTSANDBIASUPDATES_H_
#define WEIGHTSANDBIASUPDATES_H_

namespace sigmoid {

class WeightsAndBiasUpdates {
private:
	vector<vector<vector<float> > > weightUpdates;
	vector<vector<float> > biasUpdates;
public:
	void addWeightUpdate(vector<vector<float> > weightUpdate);
	void addBiasUpdate(vector<float> biasUpdate);
	WeightsAndBiasUpdates();
	string toString();
	virtual ~WeightsAndBiasUpdates();
};

} /* namespace sigmoid */

#endif /* WEIGHTSANDBIASUPDATES_H_ */
