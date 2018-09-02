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
    vector<vector<number<cpp_dec_float<300> >   > > weights;
    vector<number<cpp_dec_float<300> >   > biases;
  public:
    void setWeights(vector<vector<number<cpp_dec_float<300> >   > > weights);
    void setWeight(number<cpp_dec_float<300> >    newWeight, int row, int col);
    void setBiases(vector<number<cpp_dec_float<300> >   > biases);
    void setBias(number<cpp_dec_float<300> >    newBias, int neuron);
    void applyWeight(vector<vector<number<cpp_dec_float<300> >   > > deltaW);
    void applyBiases(vector<number<cpp_dec_float<300> >   > deltaB);
    vector<vector<number<cpp_dec_float<300> >   > > getWeights();
    vector<number<cpp_dec_float<300> >   > getBiases();
    static number<cpp_dec_float<300> >    sigmoid(number<cpp_dec_float<300> >    input);
    static number<cpp_dec_float<300> >    derivSigmoid(number<cpp_dec_float<300> >    input);
    vector<number<cpp_dec_float<300> >   > dotAndBiased(vector<number<cpp_dec_float<300> >   > inputs);
    vector<number<cpp_dec_float<300> >   > activations(vector<number<cpp_dec_float<300> >   > z);
    SigmoidLayer(int inputLength, int outputLength);
    string weightString();
    string biasString();
    string toString();
    virtual ~SigmoidLayer();
  };

} /* namespace sigmoid */

#endif /* SIGMOIDLAYER_H_ */
