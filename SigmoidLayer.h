/*
 * SigmoidLayer.h
 *
 *  Created on: Dec 30, 2017
 *      Author: liuben10
 */

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
    vector<vector<float> > weights;
    vector<float> biases;
  public:
    void setWeights(vector<vector<float> > weights);
    void setWeight(float newWeight, int row, int col);
    void setBiases(vector<float> biases);
    void setBias(float newBias, int neuron);
    vector<vector<float> > getWeights();
    vector<float> getBiases();
    static float sigmoid(float input);
    static float derivSigmoid(float input);
    vector<float> dotAndBiased(vector<float> inputs);
    vector<float> activations(vector<float> z);
    SigmoidLayer(int inputLength, int outputLength);
    string weightString();
    string biasString();
    string toString();
    virtual ~SigmoidLayer();
  };

} /* namespace sigmoid */

#endif /* SIGMOIDLAYER_H_ */
