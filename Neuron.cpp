/*
 * Neuron.cpp
 *
 *  Created on: Dec 29, 2017
 *      Author: liuben10
 */

#include "Neuron.h"
#include <utility>
#include <map>
#include "Input.h"
#include <stdio.h>
#include <math.h>


using namespace std;
using std::unique_ptr;

namespace sigmoid {

Neuron::Neuron(map<string, Input*> inputs, float bias) {
	this->inputs = inputs;
	this->bias = bias;
}

float Neuron::dotProduct(map<string, float> values) {
	float totalActivation = 0;
	for(map<string, float>::iterator iter = values.begin(); iter != values.end(); ++iter) {
		Input * i = inputs[iter->first];
		totalActivation += i->product(iter->second);
	}
	return totalActivation;
}

float Neuron::activation(map<string, float> values) {
	float dots = this->dotProduct(values);
	float biased = dots + this->bias;
	return this->sigmoid(biased);
}

float Neuron::sigmoid(float w) {
	return 1 / (1 + exp(-1 * w));
}

Neuron::~Neuron() {
	free(this);
}

} /* namespace sigmoid */
