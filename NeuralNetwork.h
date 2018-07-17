/*
 * NeuralNetwork.h
 *
 *  Created on: Dec 30, 2017
 *      Author: liuben10
 */

#include <vector>

#include "SigmoidLayer.h"
#include "WeightsAndBiasUpdates.h"

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

using namespace std;

namespace sigmoid {

class NeuralNetwork {
	int inputDim;
	int outputDim;
	vector<SigmoidLayer> layers;
public:
	WeightsAndBiasUpdates backPropagate(vector<float> input, vector<float> y);
	vector<float> feedForward(vector<float> input);
	vector<float> feedForwardWithSave(vector<float> input, vector<vector<float> > * zvecsCont, vector<vector<float> > * activationCont);
	NeuralNetwork(int neurons[], int numLayers);
	vector<float> oneDimVectorMultiply(vector<float> src, vector<float> dest);
	vector<float> hadamardProduct(vector<float> a, vector<float> b);
	vector<float> sigmoidDeriv(vector<float> activation);
	vector<float> costDerivative(vector<float> activation, vector<float> expected);
	virtual ~NeuralNetwork();
};

} /* namespace sigmoid */

#endif /* NEURALNETWORK_H_ */
