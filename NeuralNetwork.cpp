/*
 * NeuralNetwork.cpp
 *
 *  Created on: Dec 30, 2017
 *      Author: liuben10
 */

#include "NeuralNetwork.h"
#include "SigmoidLayer.h"

namespace sigmoid {

NeuralNetwork::NeuralNetwork(int inputDim, int outputDim) {
	this->inputDim = inputDim;
	this->outputDim = outputDim;
	this->layers = vector<SigmoidLayer>();
	SigmoidLayer * s1 = new SigmoidLayer(784, 16);
	this->layers.push_back(*s1);

	SigmoidLayer * s2 = new SigmoidLayer(16, 16);
	this->layers.push_back(*s2);

	SigmoidLayer * s3 = new SigmoidLayer(16, 16);
	this->layers.push_back(*s3);

	SigmoidLayer * s4 = new SigmoidLayer(16, 110);
	this->layers.push_back(*s4);
}

vector<float> NeuralNetwork::output(vector<float> input) {
	vector<float> lastResult = input;
	printf("layers size: %d\n", this->layers.size());
	for(int i = 0; i < this->layers.size(); i++) {
		printf("inputSize! %d\n", lastResult.size());
		SigmoidLayer currentLayer = this->layers[i];

		vector<float> outputLayer = currentLayer.outputs(lastResult);

		printf("single layer size: %d\n", outputLayer.size());
	    for(int i = 0; i < outputLayer.size(); i++) {
	    		printf("out = %f,", outputLayer[i]);
	    }
	    printf("\n");
		lastResult = outputLayer;
	}
	return lastResult;
}

NeuralNetwork::~NeuralNetwork() {
}

} /* namespace sigmoid */
