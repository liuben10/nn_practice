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

Neuron::Neuron(map<string, Input*> inputs, number<cpp_dec_float<300> >    bias) {
	this->inputs = inputs;
	this->bias = bias;
}

number<cpp_dec_float<300> >    Neuron::dotProduct(map<string, number<cpp_dec_float<300> >   > values) {
	number<cpp_dec_float<300> >    totalActivation = 0;
	for(map<string, number<cpp_dec_float<300> >   >::iterator iter = values.begin(); iter != values.end(); ++iter) {
		Input * i = inputs[iter->first];
		totalActivation += i->product(iter->second);
	}
	return totalActivation;
}

number<cpp_dec_float<300> >    Neuron::activation(map<string, number<cpp_dec_float<300> >   > values) {
	number<cpp_dec_float<300> >    dots = this->dotProduct(values);
	number<cpp_dec_float<300> >    biased = dots + this->bias;
	return this->sigmoid(biased);
}

number<cpp_dec_float<300> >    Neuron::sigmoid(number<cpp_dec_float<300> >    w) {
	return 1 / (1 + exp(-1 * w));
}

Neuron::~Neuron() {
}

} /* namespace sigmoid */
