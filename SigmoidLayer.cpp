/*
 * SigmoidLayer.cpp
 *
 *  Created on: Dec 30, 2017
 *      Author: liuben10
 */
#include "SigmoidLayer.h"

#include "Matrix.h"
#include <math.h>
#include <vector>
#include <iostream>
#include <random>
#include <string>

using namespace std;

namespace sigmoid {

  SigmoidLayer::SigmoidLayer(int inputLength, int layerLength) {
    this->inputLength = inputLength;
    this->layerLength = layerLength;
    this->weights = MATRIX(layerLength, ROW(inputLength, 0));
    this->biases = MATRIX(layerLength, ROW(1, 0));
    default_random_engine generator;
    normal_distribution<double> distribution(0, 0.2);

    for(int i = 0; i < this->weights.size(); i++) {
      for(int j = 0; j < this->weights[i].size(); j++) {
	this->weights[i][j] = distribution(generator);
      }
    }

    for(int i = 0; i < this->biases.size(); i++) {
      this->biases[i][0] = distribution(generator);
    }
  }

  void SigmoidLayer::applyWeight(MATRIX deltaW) {
    Matrix::printMatrixSmallLabel(deltaW, "deltaW");
    Matrix::printMatrixSmallLabel(this->weights, "weights");
    if (deltaW.size() != this->weights.size() || deltaW[0].size() != this->weights[0].size()) {
      cerr << "Error Weights do not match" << "\n";
      Matrix::printMatrixSmallLabel(deltaW, "deltaW");
      Matrix::printMatrixSmallLabel(this->weights, "weights");
      throw "Error Weights do not match";
    }
    for(int i = 0; i < this->weights.size(); i++) {
      for(int j = 0; j < this->weights[0].size(); j++) {
	this->weights[i][j] = this->weights[i][j] + deltaW[i][j];
      }
    }
  }
  
  void SigmoidLayer::applyBiases(MATRIX deltaB) {
    if (deltaB.size() != this->biases.size() || deltaB[0].size() != this->biases[0].size()) {
      cerr << "Error Biases do not match" << "\n";
      Matrix::printMatrixSmallLabel(deltaB, "deltaB");
      Matrix::printMatrixSmallLabel(this->biases, "biases");
      throw "Error Biases do not match";
    }
    for(int i = 0; i < this->biases.size(); i++) {
      this->biases[i][0] = this->biases[i][0] + deltaB[i][0];
    }
  }

  MATRIX SigmoidLayer::getBiases() {
    return this->biases;
  }

  MATRIX SigmoidLayer::getWeights() {
    return this->weights;
  }

  void SigmoidLayer::setWeights(MATRIX newWeights) {
    this->weights = newWeights;
  }

  void SigmoidLayer::setWeight(double newWeight, int row, int col) {
    this->weights[row][col] = newWeight;
  }

  void SigmoidLayer::setBiases(MATRIX biases) {
    this->biases = biases;
  }

  void SigmoidLayer::setBias(double newBias, int neuron) {
    this->biases[neuron][0] = newBias;
  }

  MATRIX SigmoidLayer::dotAndBiased(MATRIX inputs) {
    Matrix::printMatrixSmallLabel(inputs, "input");
    MATRIX wproduct = Matrix::matrixMultiply(this->weights, inputs);
    Matrix::printMatrixSmallLabel(wproduct, "product");
    MATRIX sum = Matrix::sum(wproduct, this->biases);
    Matrix::printMatrixSmallLabel(sum, "sum");
    return sum;
  }

  MATRIX SigmoidLayer::activations(MATRIX z) {
    MATRIX activations = z;
    for (int i = 0; i < layerLength; i++) {
      activations[i][0] = this->sigmoid(activations[i][0]);
    }

    return activations;
  }

  string SigmoidLayer::toString() {
    string layerString;
    layerString.append("\n===============SigmoidLayer================\n");
    layerString.append(this->weightString());
    layerString.append(this->biasString());
   
    layerString.append("\n==========================================\n");
    return layerString;
  }
  
  string SigmoidLayer::weightString() {
    ostringstream o;
    o << "\n===Weights===\n";
    return Matrix::stringMatrixLabel(this->weights, "weights");
  }

  string SigmoidLayer::biasString() {
    return Matrix::stringMatrixLabel(this->biases, "bias");
  }

  double SigmoidLayer::sigmoid(double w) {
    return ((double)1) / (1 + exp(-1 * w));
  }

  double SigmoidLayer::derivSigmoid(double z) {
    double  sigZ = SigmoidLayer::sigmoid(z);
    return  sigZ * SigmoidLayer::sigmoid(1 - sigZ);
  }

} /* namespace sigmoid */

