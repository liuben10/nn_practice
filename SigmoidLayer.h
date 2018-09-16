/*
 * SigmoidLayer.h
 *
 *  Created on: Dec 30, 2017
 *      Author: liuben10
 */
#include <boost/multiprecision/cpp_dec_float.hpp>

#include <vector>
#include <string>

using namespace std;
using namespace boost::multiprecision;

#ifndef SIGMOIDLAYER_H_
#define SIGMOIDLAYER_H_

namespace sigmoid {

  class SigmoidLayer {
  private:
    int inputLength;
    int layerLength;
    vector<vector<double   > > weights;
    vector<double   > biases;
  public:
    void setWeights(vector<vector<double   > > weights);
    void setWeight(double    newWeight, int row, int col);
    void setBiases(vector<double   > biases);
    void setBias(double    newBias, int neuron);
    void applyWeight(vector<vector<double   > > deltaW);
    void applyBiases(vector<double   > deltaB);
    vector<vector<double   > > getWeights();
    vector<double   > getBiases();
    static double    sigmoid(double    input);
    static double    derivSigmoid(double    input);
    vector<double   > dotAndBiased(vector<double   > inputs);
    vector<double   > activations(vector<double   > z);
    SigmoidLayer(int inputLength, int outputLength);
    string weightString();
    string biasString();
    string toString();
    virtual ~SigmoidLayer();
  };

} /* namespace sigmoid */

#endif /* SIGMOIDLAYER_H_ */
