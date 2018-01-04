/*
 * NeuralNetwork.cpp
 *
 *  Created on: Dec 30, 2017
 *      Author: liuben10
 */

#include "NeuralNetwork.h"
#include "Util.h"


namespace sigmoid {

NeuralNetwork::NeuralNetwork(int inputDim, int outputDim) {
	this->inputDim = inputDim;
	this->outputDim = outputDim;
	this->layers = vector<SigmoidLayer>();
	SigmoidLayer * s1 = new SigmoidLayer(inputDim, 2);
	this->layers.push_back(*s1);

	SigmoidLayer * s2 = new SigmoidLayer(1, 1);
	this->layers.push_back(*s2);

	SigmoidLayer * s3 = new SigmoidLayer(1, outputDim);
	this->layers.push_back(*s3);
}

vector<float> hadamardProduct(vector<float> a, vector<float> b) {
	vector<float> result = vector<float>(a.size(), 0);
	for(int i = 0; i < a.size(); i++) {
		result[i] = a[i] * b[i];
	}
	return result;
}

void NeuralNetwork::backPropagate(vector<float> input, vector<float> expected) {
	vector<vector<float> > activations = vector<vector<float> >();
	vector<vector<float> > zvectors = vector<vector<float> >();
	this->feedForwardWithSave(input, zvectors, activations);
	vector<float> activation = activations[activations.size() - 1];
	vector<float> costDerivative = this->costDerivative(activation, expected);
	vector<float> sigmoidPrime = this->sigmoidDeriv(activation);
	vector<float> delta = this->hadamardProduct(costDerivative, sigmoidPrime);
	vector<float> biasUpdate = delta;
//	vector<float> weightUpdate = this->matrixMultiply(biasUpdate, zvectors[zvectors.size() - 1]);





}

vector<float> NeuralNetwork::sigmoidDeriv(vector<float> activation) {
	vector<float> sigPrime = vector<float>();
	for(int i = 0; i < activation.size(); i++) {
		sigPrime[i] = SigmoidLayer::derivSigmoid(activation[i]);
	}
	return sigPrime;
}

vector<float> NeuralNetwork::oneDimVectorMultiply(vector<float> activation, vector<float> expected) {
	vector<float> result = vector<float>();
	for(int i = 0; i < activation.size(); i++) {
		float curAct = activation[i];
		float curExp = expected[i];
		float product = curAct*curExp;
		result.push_back(product);
	}
	return result;
}

vector<float> NeuralNetwork::costDerivative(vector<float> activation, vector<float> expected) {
	vector<float> delta = vector<float>();
	for(int i = 0; i < activation.size(); i++) {
		delta[i] = activation[i] - expected[i];
	}
	return delta;
}

vector<float> NeuralNetwork::feedForwardWithSave(vector<float> input, vector<vector<float> > zvecsCont, vector<vector<float> > activationCont) {
	vector<float> lastResult = input;
	for(int i = 0; i < this->layers.size(); i++) {
		SigmoidLayer currentLayer = this->layers[i];

		vector<float> zvector = currentLayer.dotAndBiased(lastResult);

		vector<float> activation = currentLayer.activations(zvector);

		printf("single layer size: %d\n", activation.size());
	    for(int i = 0; i < activation.size(); i++) {
	    		printf("out = %f,", activation[i]);
	    }
	    printf("\n");
		lastResult = activation;
	}
	return lastResult;
}

vector<float> NeuralNetwork::feedForward(vector<float> input) {
	vector<float> lastResult = input;
	for(int i = 0; i < this->layers.size(); i++) {
		SigmoidLayer currentLayer = this->layers[i];

		vector<float> zvector = currentLayer.dotAndBiased(lastResult);

		vector<float> activation = currentLayer.activations(zvector);

		printf("single layer size: %d\n", activation.size());
	    for(int i = 0; i < activation.size(); i++) {
	    		printf("out = %f,", activation[i]);
	    }
	    printf("\n");
		lastResult = activation;
	}
	return lastResult;
}

NeuralNetwork::~NeuralNetwork() {
}

} /* namespace sigmoid */
