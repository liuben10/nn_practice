/*
 * NeuralNetwork.cpp
 *
 *  Created on: Dec 30, 2017
 *      Author: liuben10
 */

#include "NeuralNetwork.h"
#include "Util.h"
#include "Matrix.h"
#include "WeightsAndBiasUpdates.h"
#include <iostream>

using namespace std;
using namespace boost::multiprecision;

namespace sigmoid {

  void checkIsColumnVector(MATRIX a, string label) {
    if (a[0].size() > 1) {
      cerr << "Error " << label << "_size==" << a[0].size() << " when it should be 1\n";
      throw "Error, matrix should be column vector";
    }
  }

  NeuralNetwork::NeuralNetwork(int neurons[], int layers) {
    this->inputDim = inputDim;
    this->outputDim = outputDim;
    this->layers = vector<SigmoidLayer *>();

    for(int i = 0; i < layers; i++) {
      SigmoidLayer *sl = new SigmoidLayer(neurons[i], neurons[i+1]);
      this->layers.push_back(sl);
    }
    this->printNetwork();
  }

  void NeuralNetwork::printNetwork() {
    cout << "==============NeuralNetwork===========" << "\n";
    for(int i = 0; i < layers.size(); i++) {
      SigmoidLayer *sl = this->layers[i];
      cout << sl->toString() << "\n";
    }
    cout << "====================================" << "\n";
  }

  void printCol(MATRIX *col, string label) {
    checkIsColumnVector(*col, label);
    cout << "\n==" << label << "_col" << "===\n";
    for(int i = 0; i < col->size(); i++) {
      cout << label << "=" << col->at(i)[0] << ", ";
    }
    cout << "\n=====\n";
  }

  void printMatrix(MATRIX * matrix, string label) {
    cout << "\n==" << label << "_matrix" << "===\n";
    for(int i = 0; i < matrix->size(); i++) {
      for(int j = 0; j < matrix->at(i).size(); j++) {
	cout << label << "=" << matrix->at(i).at(j) << ", ";
      }
      cout << "\n";
    }
  }

  void NeuralNetwork::applyUpdates(WeightsAndBiasUpdates weightAndBiasUpdates) {
    if (weightAndBiasUpdates.getWeightUpdates().size() != this->layers.size()) {
      throw "ERROR weight update size doesn't match";
    }
    if (weightAndBiasUpdates.getBiasUpdates().size() != this->layers.size()) {
      throw "ERROR bias update size doesn't match";
    }
    for(int i = this->layers.size() - 1; i >= 0; i--) {
      SigmoidLayer *layer = this->layers[i];
      cout << "\ni=" << i << " here \n";
      layer->applyWeight(weightAndBiasUpdates.weightAt(i));
      layer->applyBiases(weightAndBiasUpdates.biasAt(i));
    }
  }

  WeightsAndBiasUpdates NeuralNetwork::backPropagate(MATRIX input, MATRIX expected) {
    vector<MATRIX> * activations = new vector<MATRIX>();
    vector<MATRIX> * zvectors = new vector<MATRIX>();
    this->feedForwardWithSave(input, zvectors, activations);

    cout << "\n\n\n======REMOVE=======\n\n\n" << "\n";
    for(int i = 0; i< activations->size(); i++) {
      cout << "\ni=" << i << "\n";
      Matrix::printMatrixLabel(activations->at(i), "activation");
      Matrix::printMatrixLabel(zvectors->at(i), "zvectors");
    }
    
    MATRIX activation = activations->at(activations->size() - 1);
    MATRIX zvector = zvectors->at(zvectors->size()-1);
    MATRIX costDerivative = this->costDerivative(activation, expected);
    printMatrix(&costDerivative, "firstCostDerivative");
    MATRIX sigmoidPrime = this->sigmoidDeriv(zvector);
    printMatrix(&sigmoidPrime, "firstSigmoidPrime");
    
    MATRIX delta = Matrix::hadamard(costDerivative, sigmoidPrime);

    printCol(&delta, "delta");

    WeightsAndBiasUpdates updates = WeightsAndBiasUpdates();
    MATRIX biasUpdate = delta;
    MATRIX weightUpdate = Matrix::transposeAndMultiply(delta, activations->at(activations->size() - 2));
    printMatrix(&weightUpdate, "newWeightUpdate");

    updates.addBiasUpdate(biasUpdate);
    updates.addWeightUpdate(weightUpdate);

    int layers = this->layers.size() - 2;
    printf("weightDims={row=%d, col=%d}", weightUpdate.size(), weightUpdate[0].size());

    for(int i = layers; i >= 0; i--) {
      printf("\n=====iter: %d=======\n", i);
      MATRIX zvector = zvectors->at(i);

      SigmoidLayer *outputLayer =  this->layers[i+1];
      MATRIX prevWeights = outputLayer->getWeights();
      MATRIX sp = this->sigmoidDeriv(zvector);

      printCol(&sp, "sigmoid_prime");

      printCol(&delta, "prev_delta");

      printMatrix(&prevWeights, "prev_weights");

      delta = Matrix::matrixMultiply(delta, prevWeights);

      printMatrix(&delta, "delta_after_transpose_and_multiply");

      delta = Matrix::hadamard(delta, sp);

      printMatrix(&delta, "delta_after_hadamard_product");

      biasUpdate = delta;
      weightUpdate = Matrix::transposeAndMultiply(activations->at(i), delta);
      Matrix::printMatrixLabel(weightUpdate, "weight_update");
      Matrix::printMatrixLabel(biasUpdate, "bias_update");
      
      updates.addBiasUpdate(biasUpdate);
      updates.addWeightUpdate(weightUpdate);


      printf("\n====\n");
    }

    return updates;
  }

  MATRIX NeuralNetwork::sigmoidDeriv(MATRIX zvector) {
    checkIsColumnVector(zvector, "zvector");
    MATRIX sigPrime = MATRIX(zvector.begin(), zvector.end());
    for(int i = 0; i < zvector.size(); i++) {
      sigPrime[i][0] = SigmoidLayer::derivSigmoid(zvector[i][0]);
    }
    return sigPrime;
  }

  

  MATRIX NeuralNetwork::costDerivative(MATRIX activation, MATRIX expected) {
    checkIsColumnVector(activation, "activation");
    checkIsColumnVector(expected, "expected");
    MATRIX delta = MATRIX(activation.begin(), activation.end());
    for(int i = 0; i < activation.size(); i++) {
      delta[i][0] = 2 * (expected[i][0] - activation[i][0]);
    }
    return delta;
  }

  MATRIX NeuralNetwork::feedForwardWithSave(MATRIX input, vector<MATRIX> *zvecsCont, vector<MATRIX> *activationCont) {
    MATRIX lastResult = MATRIX(input);
    for(int i = 0; i < this->layers.size(); i++) {
      SigmoidLayer *currentLayer = this->layers[i];
      
      MATRIX zvector = currentLayer->dotAndBiased(lastResult);
      zvecsCont->push_back(zvector);

      MATRIX activation = currentLayer->activations(zvector);
      activationCont->push_back(activation);

      printCol(&activation, "activation");
      
      lastResult =  activation;
      for(int j = 0; j < lastResult.size(); j++) {
	cout << "lastRes=" << lastResult[j][0] << "\n";
      }

      printf("\n=======finished(lastResult: %d)=======\n", lastResult.size());
    }
    printf("\nFIN!\n");
    return lastResult;
  }

  MATRIX NeuralNetwork::feedForward(MATRIX input) {
    
    MATRIX lastResult = MATRIX(input);
    for(int i = 0; i < this->layers.size(); i++) {
      SigmoidLayer *currentLayer = this->layers[i];

      MATRIX zvector = currentLayer->dotAndBiased(lastResult);

      MATRIX activation = currentLayer->activations(zvector);
      printCol(&activation, "activation");
      cout << "\n";
      lastResult = activation;
    }
    return lastResult;
  }

  NeuralNetwork::~NeuralNetwork() {
  }

} /* namespace sigmoid */
