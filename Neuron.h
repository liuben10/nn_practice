/*
 * Neuron.h
 *
 *  Created on: Dec 29, 2017
 *      Author: liuben10
 */

#ifndef NEURON_H_
#define NEURON_H_

#include <vector>
#include <map>

#include "Input.h"

using namespace std;

namespace sigmoid {

class Neuron {
private:
	map<string, Input*> inputs;
	float bias;
public:
	float dotProduct(map<string, float> inputs);
	float activation(map<string, float> inputs);
	float sigmoid(float val);
	Neuron(map<string, Input*> inputs, float bias);
	virtual ~Neuron();
};

} /* namespace sigmoid */

#endif /* NEURON_H_ */
