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
#include <random>

using namespace std;

namespace sigmoid {

SigmoidLayer::SigmoidLayer(int inputLength, int layerLength) {
	this->inputLength = inputLength;
	this->layerLength = layerLength;
	this->weights = vector<vector<float> >(layerLength, vector<float>(inputLength, 0));
	this->biases = vector<float>(layerLength, 0);
	default_random_engine generator;
	normal_distribution<double> distribution(-1.0, 1.0);

	for(int i = 0; i < this->weights.size(); i++) {
		for(int j = 0; j < this->weights[i].size(); j++) {
			this->weights[i][j] = distribution(generator);
		}
	}

	for(int i = 0; i < this->biases.size(); i++) {
		this->biases[i] = distribution(generator);
	}
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

	printf("SIZE: %d, \n", inputLength);

	for(int i = 0; i < inputLength; i++) {
		printf("in=%f, ", inputs[i]);
	}

	printf("\n");


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

vector<float> SigmoidLayer::activations(vector<float> z) {
	vector<float> activations = z;
	for (int i = 0; i < layerLength; i++) {
		float sigmoidified = this->sigmoid(activations[i]);
		activations[i] = sigmoidified;
	}

	return activations;
}

float SigmoidLayer::sigmoid(float w) {
	return 1 / (1 + exp(-1 * w));
}

float SigmoidLayer::derivSigmoid(float z) {
	float sigZ = SigmoidLayer::sigmoid(z);
	return  sigZ * SigmoidLayer::sigmoid(1 - sigZ);
}

SigmoidLayer::~SigmoidLayer() {
}

} /* namespace sigmoid */

