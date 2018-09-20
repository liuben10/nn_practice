/*
 * SigmoidLayer.h
 *
 *  Created on: Dec 30, 2017
 *      Author: liuben10
 */

#include "Matrix.h"
#include <vector>
#include <string>

using namespace std;

#ifndef SIGMOIDLAYER_H_
#define SIGMOIDLAYER_H_

namespace sigmoid {

  class SigmoidLayer {
  private:
    int inputLength;
    int layerLength;
    MATRIX weights;
    MATRIX biases;
  public:
    void setWeights(MATRIX weights);
    void setWeight(double newWeight, int row, int col);
    void setBiases(MATRIX biases);
    void setBias(double newBias, int neuron);
    void applyWeight(MATRIX deltaW);
    void applyBiases(MATRIX deltaB);
    MATRIX getWeights();
    MATRIX getBiases();
    static double sigmoid(double input);
    static double derivSigmoid(double input);
    MATRIX dotAndBiased(MATRIX inputs);
    MATRIX activations(MATRIX z);
    SigmoidLayer(int inputLength, int outputLength);
    string weightString();
    string biasString();
    string toString();
    virtual ~SigmoidLayer();
  };

} /* namespace sigmoid */

#endif /* SIGMOIDLAYER_H_ */
