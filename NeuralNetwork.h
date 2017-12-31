/*
 * NeuralNetwork.h
 *
 *  Created on: Dec 30, 2017
 *      Author: liuben10
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "SigmoidLayer.h";

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

using namespace std;

namespace sigmoid {

class NeuralNetwork {
	int inputDim;
	int outputDim;
	vector<SigmoidLayer> layers;
public:
	vector<float> output(vector<float> input);
	NeuralNetwork(int inputDim, int outputDim);
	virtual ~NeuralNetwork();
};

} /* namespace sigmoid */

#endif /* NEURALNETWORK_H_ */
