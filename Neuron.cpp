/*
 * Neuron.cpp
 *
 *  Created on: Dec 29, 2017
 *      Author: liuben10
 */

#include <boost/multiprecision/cpp_dec_float.hpp>

#include "Neuron.h"
#include <utility>
#include <map>
#include "Input.h"
#include <stdio.h>
#include <math.h>


using namespace std;
using std::unique_ptr;
using namespace boost::multiprecision;


namespace sigmoid {

Neuron::Neuron(map<string, Input*> inputs, double    bias) {
	this->inputs = inputs;
	this->bias = bias;
}

double    Neuron::dotProduct(map<string, double   > values) {
	double    totalActivation = 0;
	for(map<string, double   >::iterator iter = values.begin(); iter != values.end(); ++iter) {
		Input * i = inputs[iter->first];
		totalActivation += i->product(iter->second);
	}
	return totalActivation;
}

double    Neuron::activation(map<string, double   > values) {
	double    dots = this->dotProduct(values);
	double    biased = dots + this->bias;
	return this->sigmoid(biased);
}

double    Neuron::sigmoid(double    w) {
	return 1 / (1 + exp(-1 * w));
}

Neuron::~Neuron() {
}

} /* namespace sigmoid */
