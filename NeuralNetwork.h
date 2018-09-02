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
    WeightsAndBiasUpdates backPropagate(vector<number<cpp_dec_float<200> > > input, vector<number<cpp_dec_float<200> > > y);
    vector<number<cpp_dec_float<200> > > feedForward(vector<number<cpp_dec_float<200> > > input);
    vector<number<cpp_dec_float<200> > > feedForwardWithSave(vector<number<cpp_dec_float<200> > > input, vector<vector<number<cpp_dec_float<200> > > > * zvecsCont, vector<vector<number<cpp_dec_float<200> > > > * activationCont);
    NeuralNetwork(int neurons[], int numLayers);
    vector<number<cpp_dec_float<200> > > oneDimVectorMultiply(vector<number<cpp_dec_float<200> > > src, vector<number<cpp_dec_float<200> > > dest);
    vector<number<cpp_dec_float<200> > > hadamardProduct(vector<number<cpp_dec_float<200> > > a, vector<number<cpp_dec_float<200> > > b);
    vector<number<cpp_dec_float<200> > > sigmoidDeriv(vector<number<cpp_dec_float<200> > > activation);
    vector<number<cpp_dec_float<200> > > costDerivative(vector<number<cpp_dec_float<200> > > activation, vector<number<cpp_dec_float<200> > > expected);
    void applyUpdates(WeightsAndBiasUpdates *updates);
    void printNetwork();
    virtual ~NeuralNetwork();
  };

} /* namespace sigmoid */

#endif /* NEURALNETWORK_H_ */
