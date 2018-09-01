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
    vector<vector<cpp_dec_float_100> > weights;
    vector<cpp_dec_float_100> biases;
  public:
    void setWeights(vector<vector<cpp_dec_float_100> > weights);
    void setWeight(cpp_dec_float_100 newWeight, int row, int col);
    void setBiases(vector<cpp_dec_float_100> biases);
    void setBias(cpp_dec_float_100 newBias, int neuron);
    vector<vector<cpp_dec_float_100> > getWeights();
    vector<cpp_dec_float_100> getBiases();
    static cpp_dec_float_100 sigmoid(cpp_dec_float_100 input);
    static cpp_dec_float_100 derivSigmoid(cpp_dec_float_100 input);
    vector<cpp_dec_float_100> dotAndBiased(vector<cpp_dec_float_100> inputs);
    vector<cpp_dec_float_100> activations(vector<cpp_dec_float_100> z);
    SigmoidLayer(int inputLength, int outputLength);
    string weightString();
    string biasString();
    string toString();
    virtual ~SigmoidLayer();
  };

} /* namespace sigmoid */

#endif /* SIGMOIDLAYER_H_ */
