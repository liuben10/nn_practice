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

  NeuralNetwork::NeuralNetwork(int neurons[], int layers) {
    this->inputDim = inputDim;
    this->outputDim = outputDim;
    this->layers = vector<SigmoidLayer *>();

    for(int i = 0; i < layers-1; i++) {
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

  
  vector<double> NeuralNetwork::hadamardProduct(vector<double> a, vector<double> b) {
    vector<double> result(a.begin(), a.end());
    for(int i = 0; i < a.size(); i++) {
      result[i] *=  b[i];
    }
    return result;
  }

  void printCol(vector<double> * col, string label) {
    cout << "\n==" << label << "_col" << "===\n";
    for(int i = 0; i < col->size(); i++) {
      cout << label << "=" << col->at(i) << ", ";
    }
    cout << "\n=====\n";
  }

  void printMatrix(vector<vector<double> > * matrix, string label) {
    cout << "\n==" << label << "_matrix" << "===\n";
    for(int i = 0; i < matrix->size(); i++) {
      for(int j = 0; j < matrix->at(i).size(); j++) {
	cout << label << "=" << matrix->at(i).at(j) << ", ";
      }
      cout << "\n";
    }
  }

  void NeuralNetwork::applyUpdates(WeightsAndBiasUpdates *weightAndBiasUpdates) {
    for(int i = this->layers.size() - 1; i >= 0; i--) {
      SigmoidLayer *layer = this->layers[i];
      layer->applyWeight(weightAndBiasUpdates->weightAt(i));
      layer->applyBiases(weightAndBiasUpdates->biasAt(i));
    }
  }

  WeightsAndBiasUpdates NeuralNetwork::backPropagate(vector<double> input, vector<double> expected) {
    vector<vector<double> > * activations = new vector<vector<double> >();
    vector<vector<double> > * zvectors = new vector<vector<double> >();
    this->feedForwardWithSave(input, zvectors, activations);
    printMatrix(activations, "activation");
    printMatrix(zvectors, "zscore");
    
    vector<double> activation = activations->at(activations->size() - 1);
    vector<double> zvector = zvectors->at(zvectors->size()-1);
    vector<double> costDerivative = this->costDerivative(activation, expected);
    printCol(&costDerivative, "firstCostDerivative");
    vector<double> sigmoidPrime = this->sigmoidDeriv(zvector);
    printCol(&sigmoidPrime, "firstSigmoidPrime");
    
    vector<double> delta = this->hadamardProduct(costDerivative, sigmoidPrime);

    printCol(&delta, "delta");

    WeightsAndBiasUpdates updates = WeightsAndBiasUpdates();
    vector<double> biasUpdate = delta;
    vector<vector<double> > weightUpdate = Matrix::transposeAndMultiply(activations->at(activations->size() - 2), delta);

    printMatrix(&weightUpdate, "newWeightUpdate");

    updates.addBiasUpdate(biasUpdate);
    updates.addWeightUpdate(weightUpdate);

    int layers = this->layers.size() - 2;
    printf("weightDims={row=%d, col=%d}", weightUpdate.size(), weightUpdate[0].size());

    for(int i = layers; i >= 0; i--) {
      printf("\n=====iter: %d=======\n", i);
      vector<double> zvector = zvectors->at(i);

      SigmoidLayer *outputLayer =  this->layers[i+1];
      vector<vector<double> > prevWeights = outputLayer->getWeights();
      vector<double> sp = this->sigmoidDeriv(zvector);

      printCol(&sp, "sigmoid_prime");

      printCol(&delta, "prev_delta");

      printMatrix(&prevWeights, "prev_weights");

      delta = Matrix::transposeAndMultiplyOneDim(prevWeights, delta);

      printCol(&delta, "delta_after_transpose_and_multiply");

      delta = this->hadamardProduct(delta, sp);

      printCol(&delta, "delta_after_hadamard_product");

      for(int k = 0; k < delta.size(); k++) {
	cout << "new_delta=" << delta[k] << ", ";
      }
      printf("\n");

      biasUpdate = delta;
      weightUpdate = Matrix::transposeAndMultiply(activations->at(i), biasUpdate);

      printf("weightDims={row=%d, col=%d}\n", weightUpdate.size(), weightUpdate[0].size());

      for(int k = 0; k < weightUpdate.size(); k++) {
	for(int j = 0; j < weightUpdate[k].size(); j++) {
	  cout << "weight=" << weightUpdate[k][j] << ", ";
	}
	printf("\n");
      }

      updates.addBiasUpdate(biasUpdate);
      updates.addWeightUpdate(weightUpdate);

      printf("\n====\n");
    }

    return updates;
  }

  vector<double> NeuralNetwork::sigmoidDeriv(vector<double> zvector) {
    vector<double> sigPrime = vector<double>(zvector.begin(), zvector.end());
    for(int i = 0; i < zvector.size(); i++) {
      sigPrime[i] = SigmoidLayer::derivSigmoid(zvector[i]);
    }
    return sigPrime;
  }

  vector<double> NeuralNetwork::oneDimVectorMultiply(vector<double> activation, vector<double> expected) {
    vector<double> result = vector<double>();
    for(int i = 0; i < activation.size(); i++) {
      double curAct = activation[i];
      double curExp = expected[i];
      double product = curAct*curExp;
      result.push_back(product);
    }
    return result;
  }

  vector<double> NeuralNetwork::costDerivative(vector<double> activation, vector<double> expected) {
    vector<double> delta = vector<double>(activation.begin(), activation.end());
    for(int i = 0; i < activation.size(); i++) {
      delta[i] = 2 * (expected[i] - activation[i]);
    }
    return delta;
  }

  vector<double> NeuralNetwork::feedForwardWithSave(vector<double> input, vector<vector<double> > * zvecsCont, vector<vector<double> > * activationCont) {
    vector<double> lastResult = input;
    for(int i = 0; i < this->layers.size(); i++) {
      SigmoidLayer *currentLayer = this->layers[i];

      vector<double> zvector = currentLayer->dotAndBiased(lastResult);
      zvecsCont->push_back(zvector);

      vector<double> activation = currentLayer->activations(zvector);
      activationCont->push_back(activation);

      printCol(&activation, "activation");
      
      lastResult =  activation;
      for(int j = 0; j < lastResult.size(); j++) {
	cout << "lastRes=" << lastResult[j] << "\n";
      }

      printf("\n=======finished(lastResult: %d)=======\n", lastResult.size());
    }
    printf("\nFIN!\n");
    return lastResult;
  }

  vector<double> NeuralNetwork::feedForward(vector<double> input) {
    vector<double> lastResult = vector<double>();
    for(int i = 0; i < this->layers.size(); i++) {
      SigmoidLayer *currentLayer = this->layers[i];

      vector<double> zvector = currentLayer->dotAndBiased(lastResult);

      vector<double> activation = currentLayer->activations(zvector);
      printCol(&activation, "activation");
      cout << "\n";
      lastResult = activation;
    }
    return lastResult;
  }

  NeuralNetwork::~NeuralNetwork() {
  }

} /* namespace sigmoid */
