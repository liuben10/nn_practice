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

Neuron::Neuron(map<string, Input*> inputs, cpp_dec_float_100 bias) {
	this->inputs = inputs;
	this->bias = bias;
}

cpp_dec_float_100 Neuron::dotProduct(map<string, cpp_dec_float_100> values) {
	cpp_dec_float_100 totalActivation = 0;
	for(map<string, cpp_dec_float_100>::iterator iter = values.begin(); iter != values.end(); ++iter) {
		Input * i = inputs[iter->first];
		totalActivation += i->product(iter->second);
	}
	return totalActivation;
}

cpp_dec_float_100 Neuron::activation(map<string, cpp_dec_float_100> values) {
	cpp_dec_float_100 dots = this->dotProduct(values);
	cpp_dec_float_100 biased = dots + this->bias;
	return this->sigmoid(biased);
}

cpp_dec_float_100 Neuron::sigmoid(cpp_dec_float_100 w) {
	return 1 / (1 + exp(-1 * w));
}

Neuron::~Neuron() {
}

} /* namespace sigmoid */
