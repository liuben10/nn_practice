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
    this->weights = vector<vector<cpp_dec_float_100> >(layerLength, vector<cpp_dec_float_100>(inputLength, 0));
    this->biases = vector<cpp_dec_float_100>(layerLength, 0);
    default_random_engine generator;
    normal_distribution<double> distribution(-1.0, 1.0);

    for(int i = 0; i < this->weights.size(); i++) {
      for(int j = 0; j < this->weights[i].size(); j++) {
	this->weights[i][j] = distribution(generator);
      }
    }

    for(int i = 0; i < this->biases.size(); i++) {
      this->biases[i] = distribution(generator);
    }
  }

  vector<cpp_dec_float_100> SigmoidLayer::getBiases() {
    return this->biases;
  }

  vector<vector<cpp_dec_float_100> > SigmoidLayer::getWeights() {
    return this->weights;
  }

  void SigmoidLayer::setWeights(vector<vector<cpp_dec_float_100> > newWeights) {
    this->weights = newWeights;
  }

  void SigmoidLayer::setWeight(cpp_dec_float_100 newWeight, int row, int col) {
    this->weights[row][col] = newWeight;
  }

  void SigmoidLayer::setBiases(vector<cpp_dec_float_100> biases) {
    this->biases = biases;
  }

  void SigmoidLayer::setBias(cpp_dec_float_100 newBias, int neuron) {
    this->biases[neuron] = newBias;
  }

  vector<cpp_dec_float_100> SigmoidLayer::dotAndBiased(vector<cpp_dec_float_100> inputs) {
    vector<cpp_dec_float_100> result(layerLength, 0);
    int inputLength = inputs.size();

    printf("SIZE: %d, \n", inputLength);

    for(int i = 0; i < inputLength; i++) {
      cout << "in=" << inputs[i] << "\n";
    }

    printf("\n");


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

  vector<cpp_dec_float_100> SigmoidLayer::activations(vector<cpp_dec_float_100> z) {
    vector<cpp_dec_float_100> activations = z;
    for (int i = 0; i < layerLength; i++) {
      cpp_dec_float_100 sigmoidified = this->sigmoid(activations[i]);
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

  cpp_dec_float_100 SigmoidLayer::sigmoid(cpp_dec_float_100 w) {
    return 1 / (1 + exp(-1 * w));
  }

  cpp_dec_float_100 SigmoidLayer::derivSigmoid(cpp_dec_float_100 z) {
    cpp_dec_float_100 sigZ = SigmoidLayer::sigmoid(z);
    return  sigZ * SigmoidLayer::sigmoid(1 - sigZ);
  }

  SigmoidLayer::~SigmoidLayer() {
  }

} /* namespace sigmoid */

