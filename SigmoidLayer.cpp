/*
 * SigmoidLayer.cpp
 *
 *  Created on: Dec 30, 2017
 *      Author: liuben10
 */

#include "SigmoidLayer.h"
#include <boost/multiprecision/cpp_dec_float.hpp>

#include <math.h>
#include <vector>
#include <iostream>
#include <random>
#include <string>

using namespace std;
using namespace boost::multiprecision;

namespace sigmoid {

  SigmoidLayer::SigmoidLayer(int inputLength, int layerLength) {
    this->inputLength = inputLength;
    this->layerLength = layerLength;
    this->weights = vector<vector<number<cpp_dec_float<200> > > >(layerLength, vector<number<cpp_dec_float<200> > >(inputLength, 0));
    this->biases = vector<number<cpp_dec_float<200> > >(layerLength, 0);
    default_random_engine generator;
    normal_distribution<double> distribution(0, 0.2);

    for(int i = 0; i < this->weights.size(); i++) {
      for(int j = 0; j < this->weights[i].size(); j++) {
	this->weights[i][j] = distribution(generator);
      }
    }

    for(int i = 0; i < this->biases.size(); i++) {
      this->biases[i] = distribution(generator);
    }
  }

  void SigmoidLayer::applyWeight(vector<vector<number<cpp_dec_float<200> > > > deltaW) {
    for(int i = 0; i < this->weights.size(); i++) {
      for(int j = 0; j < this->weights[i].size(); j++) {
	this->weights[i][j] = this->weights[i][j] + deltaW[i][j];
      }
    }
  }
  
  void SigmoidLayer::applyBiases(vector<number<cpp_dec_float<200> > > deltaB) {
    for(int i = 0; i < this->biases.size(); i++) {
      this->biases[i] = this->biases[i] + deltaB[i];
    }
  }

  vector<number<cpp_dec_float<200> > > SigmoidLayer::getBiases() {
    return this->biases;
  }

  vector<vector<number<cpp_dec_float<200> > > > SigmoidLayer::getWeights() {
    return this->weights;
  }

  void SigmoidLayer::setWeights(vector<vector<number<cpp_dec_float<200> > > > newWeights) {
    this->weights = newWeights;
  }

  void SigmoidLayer::setWeight(number<cpp_dec_float<200> >  newWeight, int row, int col) {
    this->weights[row][col] = newWeight;
  }

  void SigmoidLayer::setBiases(vector<number<cpp_dec_float<200> > > biases) {
    this->biases = biases;
  }

  void SigmoidLayer::setBias(number<cpp_dec_float<200> >  newBias, int neuron) {
    this->biases[neuron] = newBias;
  }

  vector<number<cpp_dec_float<200> > > SigmoidLayer::dotAndBiased(vector<number<cpp_dec_float<200> > > inputs) {
    vector<number<cpp_dec_float<200> > > result(layerLength, 0);
    int inputLength = inputs.size();

    for(int j = 0; j < layerLength; j++) {
      for(int i = 0; i < inputLength; i++) {
	result[j] += this->weights[j][i] * inputs[i];
      }
    }

    for(int j = 0; j < layerLength; j++) {
      result[j] += this->biases[j];
    }


    return result;
  }

  vector<number<cpp_dec_float<200> > > SigmoidLayer::activations(vector<number<cpp_dec_float<200> > > z) {
    vector<number<cpp_dec_float<200> > > activations = z;
    for (int i = 0; i < layerLength; i++) {
      number<cpp_dec_float<200> >  sigmoidified = this->sigmoid(activations[i]);
      activations[i] = sigmoidified;
    }

    return activations;
  }

  string SigmoidLayer::toString() {
    string layerString;
    layerString.append("===============SigmoidLayer================");
    layerString.append(this->weightString());
    layerString.append(this->biasString());
   
    layerString.append("==========================================");
    return layerString;
  }
  string SigmoidLayer::weightString() {
    ostringstream o;
    o << "\n===Weights===\n";
    for(int i = 0; i < this->weights.size(); i++) {
      ostringstream row;
      for(int j = 0; j < this->weights[i].size(); j++) {
	row << this->weights[i][j] << ", ";
      }
      o << row.str() << "\n";
    }
    return o.str();
  }

  string SigmoidLayer::biasString() {
    ostringstream o;
    o << "\n===Biases===\n";
    for(int i = 0; i < this->biases.size(); i++) {
      o << this->biases[i] << ", ";
    }
    o << "\n";
    return o.str();
  }

  number<cpp_dec_float<200> >  SigmoidLayer::sigmoid(number<cpp_dec_float<200> >  w) {
    return 1 / (1 + exp(-1 * w));
  }

  number<cpp_dec_float<200> >  SigmoidLayer::derivSigmoid(number<cpp_dec_float<200> >  z) {
    number<cpp_dec_float<200> >  sigZ = SigmoidLayer::sigmoid(z);
    return  sigZ * SigmoidLayer::sigmoid(1 - sigZ);
  }

  SigmoidLayer::~SigmoidLayer() {
  }

} /* namespace sigmoid */

