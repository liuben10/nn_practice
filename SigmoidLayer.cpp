/*
 * SigmoidLayer.cpp
 *
 *  Created on: Dec 30, 2017
 *      Author: liuben10
 */

#include "SigmoidLayer.h"

#include <math.h>
#include <vector>
#include <iostream>

using namespace std;

namespace sigmoid {

SigmoidLayer::SigmoidLayer(int inputLength, int layerLength) {
	this->inputLength = inputLength;
	this->layerLength = layerLength;
	this->weights = vector<vector<float> >(layerLength, vector<float>(inputLength, 0));
	this->biases = vector<float>(layerLength, 0);
}

vector<float> SigmoidLayer::getBiases() {
	return this->biases;
}

vector<vector<float> > SigmoidLayer::getWeights() {
	return this->weights;
}

void SigmoidLayer::setWeights(vector<vector<float> > newWeights) {
	this->weights = newWeights;
}

void SigmoidLayer::setWeight(float newWeight, int row, int col) {
	this->weights[row][col] = newWeight;
}

void SigmoidLayer::setBiases(vector<float> biases) {
	this->biases = biases;
}

void SigmoidLayer::setBias(float newBias, int neuron) {
	this->biases[neuron] = newBias;
}

vector<float> SigmoidLayer::dotAndBiased(vector<float> inputs) {
	vector<float> result(layerLength, 0);


	for(int j = 0; j < layerLength; j++) {
		for(int i = 0; i < inputLength; i++) {
			result[j] += this->weights[j][i] * inputs[i];
		}
	}

	for(int j = 0; j < layerLength; j++) {
		result[j] += this->biases[j];
	}


	return result;
}

vector<float> SigmoidLayer::outputs(vector<float> inputs) {
	vector<float> z = dotAndBiased(inputs);

	for (int i = 0; i < layerLength; i++) {
		float sigmoidified = this->sigmoid(z[i]);
		z[i] = sigmoidified;
	}

	return z;
}

float SigmoidLayer::sigmoid(float w) {
	return 1 / (1 + exp(-1 * w));
}

SigmoidLayer::~SigmoidLayer() {
	free(this);
}

} /* namespace sigmoid */

using namespace sigmoid;
using namespace std;

//int main() {
//	SigmoidLayer * sl = new SigmoidLayer(10, 4);
//	vector<vector<float> > weights;
//
//	float firstRow[10] = {0, 10, 0, 10, 0, 10, 0, 10, 0, 10};
//	vector<float> firstRowVec(firstRow, firstRow + (sizeof(firstRow) / sizeof(firstRow[0])) );
//	weights.push_back(firstRowVec);
//
//	float secondRow[10] = {0, 0, 10, 10, 0, 0, 10, 10, 0, 0};
//	vector<float> secondRowVec(secondRow, secondRow + (sizeof(secondRow) / sizeof(secondRow[0])) );
//	weights.push_back(secondRowVec);
//
//	float thirdRow[10] = {0, 0, 0, 0, 10, 10, 10, 10, 0, 0};
//	vector<float> thirdRowVec(thirdRow, thirdRow + (sizeof(thirdRow) / sizeof(thirdRow[0])) );
//	weights.push_back(thirdRowVec);
//
//	float fourthRow[10] = {0, 0, 0, 0, 0, 0, 0, 0, 10, 10};
//	vector<float> fourthRowVec(fourthRow, fourthRow + (sizeof(fourthRow) / sizeof(fourthRow[0])) );
//	weights.push_back(fourthRowVec);
//
//	sl->setWeights(weights);
//
//	float biases[4] = {-5, -5, -5, -5};
//	vector<float> biasVec(biases, biases + (sizeof(biases) / sizeof(biases[0])));
//
//	sl->setBiases(biasVec);
//
//	float inputs[10] = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
//	vector<float> inputsVec(inputs, inputs + (sizeof(inputs) / sizeof(inputs[0])));
//
//	vector<float> activations = sl->outputs(inputsVec);
//}
