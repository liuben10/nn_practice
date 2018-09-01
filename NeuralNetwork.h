/*
 * NeuralNetwork.h
 *
 *  Created on: Dec 30, 2017
 *      Author: liuben10
 */
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <vector>

#include "SigmoidLayer.h"
#include "WeightsAndBiasUpdates.h"

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

using namespace std;
using namespace boost::multiprecision;


namespace sigmoid {

class NeuralNetwork {
	int inputDim;
	int outputDim;
	vector<SigmoidLayer> layers;
public:
	WeightsAndBiasUpdates backPropagate(vector<cpp_dec_float_100> input, vector<cpp_dec_float_100> y);
	vector<cpp_dec_float_100> feedForward(vector<cpp_dec_float_100> input);
	vector<cpp_dec_float_100> feedForwardWithSave(vector<cpp_dec_float_100> input, vector<vector<cpp_dec_float_100> > * zvecsCont, vector<vector<cpp_dec_float_100> > * activationCont);
	NeuralNetwork(int neurons[], int numLayers);
	vector<cpp_dec_float_100> oneDimVectorMultiply(vector<cpp_dec_float_100> src, vector<cpp_dec_float_100> dest);
	vector<cpp_dec_float_100> hadamardProduct(vector<cpp_dec_float_100> a, vector<cpp_dec_float_100> b);
	vector<cpp_dec_float_100> sigmoidDeriv(vector<cpp_dec_float_100> activation);
	vector<cpp_dec_float_100> costDerivative(vector<cpp_dec_float_100> activation, vector<cpp_dec_float_100> expected);
	virtual ~NeuralNetwork();
};

} /* namespace sigmoid */

#endif /* NEURALNETWORK_H_ */
