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
	double bias;
public:
	double dotProduct(map<string, double> inputs);
	double activation(map<string, double> inputs);
	double sigmoid(double val);
	Neuron(map<string, Input*> inputs, double bias);
	virtual ~Neuron();
};

} /* namespace sigmoid */

#endif /* NEURON_H_ */
