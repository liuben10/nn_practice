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
using namespace boost::multiprecision;


namespace sigmoid {

  class NeuralNetwork {
    int inputDim;
    int outputDim;
    vector<SigmoidLayer *> layers;
  public:
    WeightsAndBiasUpdates backPropagate(vector<double   > input, vector<double   > y);
    vector<double   > feedForward(vector<double   > input);
    vector<double   > feedForwardWithSave(vector<double   > input, vector<vector<double   > > * zvecsCont, vector<vector<double   > > * activationCont);
    NeuralNetwork(int neurons[], int numLayers);
    vector<double   > oneDimVectorMultiply(vector<double   > src, vector<double   > dest);
    vector<double   > hadamardProduct(vector<double   > a, vector<double   > b);
    vector<double   > sigmoidDeriv(vector<double   > activation);
    vector<double   > costDerivative(vector<double   > activation, vector<double   > expected);
    void applyUpdates(WeightsAndBiasUpdates *updates);
    void printNetwork();
    virtual ~NeuralNetwork();
  };

} /* namespace sigmoid */

#endif /* NEURALNETWORK_H_ */
