/*
 * Neuron.h
 *
 *  Created on: Dec 29, 2017
 *      Author: liuben10
 */
#include <boost/multiprecision/cpp_dec_float.hpp>

#ifndef NEURON_H_
#define NEURON_H_

#include <vector>
#include <map>

#include "Input.h"

using namespace std;
using namespace boost::multiprecision;


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
