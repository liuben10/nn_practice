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
    WeightsAndBiasUpdates backPropagate(MATRIX input, MATRIX y);
    MATRIX feedForward(MATRIX input);
    MATRIX feedForwardWithSave(MATRIX input, vector<MATRIX> * zvecsCont, vector<MATRIX> * activationCont);
    NeuralNetwork(int neurons[], int numLayers);
    MATRIX sigmoidDeriv(MATRIX activation);
    MATRIX costDerivative(MATRIX activation, MATRIX expected);
    void applyUpdates(WeightsAndBiasUpdates updates);
    void printNetwork();
    virtual ~NeuralNetwork();
  };

} /* namespace sigmoid */

#endif /* NEURALNETWORK_H_ */
