/*
 * NeuralNetwork.cpp
 *
 *  Created on: Dec 30, 2017
 *      Author: liuben10
 */

#include <boost/multiprecision/cpp_dec_float.hpp>

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
    this->layers = vector<SigmoidLayer>();

    for(int i = 0; i < layers-1; i++) {
      SigmoidLayer sl = SigmoidLayer(neurons[i], neurons[i+1]);
      cout << sl.toString() << "\n";
      this->layers.push_back(sl);
    }
  }

  vector<cpp_dec_float_100> NeuralNetwork::hadamardProduct(vector<cpp_dec_float_100> a, vector<cpp_dec_float_100> b) {
    vector<cpp_dec_float_100> result = vector<cpp_dec_float_100>(a.size(), 0);
    for(int i = 0; i < a.size(); i++) {
      result[i] = a[i] * b[i];
    }
    return result;
  }

  WeightsAndBiasUpdates NeuralNetwork::backPropagate(vector<cpp_dec_float_100> input, vector<cpp_dec_float_100> expected) {
    vector<vector<cpp_dec_float_100> > * activations = new vector<vector<cpp_dec_float_100> >();
    vector<vector<cpp_dec_float_100> > * zvectors = new vector<vector<cpp_dec_float_100> >();
    this->feedForwardWithSave(input, zvectors, activations);
    vector<cpp_dec_float_100> activation = activations->at(activations->size() - 1);
    vector<cpp_dec_float_100> zvector = zvectors->at(zvectors->size()-1);
    vector<cpp_dec_float_100> costDerivative = this->costDerivative(activation, expected);
    vector<cpp_dec_float_100> sigmoidPrime = this->sigmoidDeriv(zvector);

    vector<cpp_dec_float_100> delta = this->hadamardProduct(costDerivative, sigmoidPrime);

    for(int i = 0; i < delta.size(); i++) {
      cout << "new_delta" << delta[i] << ", ";
    }
    cout << "\n";

    WeightsAndBiasUpdates updates = WeightsAndBiasUpdates();
    vector<cpp_dec_float_100> biasUpdate = delta;
    vector<vector<cpp_dec_float_100> > weightUpdate = Matrix::transposeAndMultiply(activations->at(activations->size() - 2), biasUpdate);

    updates.addBiasUpdate(biasUpdate);
    updates.addWeightUpdate(weightUpdate);

    int layers = this->layers.size() - 2;
    printf("weightDims={row=%d, col=%d}", weightUpdate.size(), weightUpdate[0].size());

    for(int i = layers; i >= 0; i--) {
      printf("\n=====iter: %d=======\n", i);
      vector<cpp_dec_float_100> zvector = zvectors->at(i);

      SigmoidLayer outputLayer =  this->layers[i];
      vector<vector<cpp_dec_float_100> > prevWeights = outputLayer.getWeights();
      vector<cpp_dec_float_100> sp = this->sigmoidDeriv(zvector);

      for(int k = 0; k < sp.size(); k++) {
	cout << "sp=" << sp[k] << ", ";
      }
      printf("\n");

      for(int k = 0; k < delta.size(); k++) {
	cout << "prev_delta=" <<  delta[k] << ", ";
      }
      printf("\n");

      delta = Matrix::transposeAndMultiplyOneDim(prevWeights, delta);
      printf("\ndeltaSize=%d\n", delta.size());
      printf("\nsigmoidSize=%d\n", sp.size());

      delta = this->hadamardProduct(delta, sp);

      for(int k = 0; k < delta.size(); k++) {
	cout << "new_delta=" << delta[k] << ", ";
      }
      printf("\n");

      biasUpdate = delta;
      weightUpdate = Matrix::transposeAndMultiply(activations->at(i), biasUpdate);

      printf("weightDims={row=%d, col=%d}", weightUpdate.size(), weightUpdate[0].size());

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

  vector<cpp_dec_float_100> NeuralNetwork::sigmoidDeriv(vector<cpp_dec_float_100> activation) {
    vector<cpp_dec_float_100> sigPrime = vector<cpp_dec_float_100>();
    for(int i = 0; i < activation.size(); i++) {
      sigPrime.push_back(SigmoidLayer::derivSigmoid(activation[i]));
    }
    return sigPrime;
  }

  vector<cpp_dec_float_100> NeuralNetwork::oneDimVectorMultiply(vector<cpp_dec_float_100> activation, vector<cpp_dec_float_100> expected) {
    vector<cpp_dec_float_100> result = vector<cpp_dec_float_100>();
    for(int i = 0; i < activation.size(); i++) {
      cpp_dec_float_100 curAct = activation[i];
      cpp_dec_float_100 curExp = expected[i];
      cpp_dec_float_100 product = curAct*curExp;
      result.push_back(product);
    }
    return result;
  }

  vector<cpp_dec_float_100> NeuralNetwork::costDerivative(vector<cpp_dec_float_100> activation, vector<cpp_dec_float_100> expected) {
    vector<cpp_dec_float_100> delta = vector<cpp_dec_float_100>();
    for(int i = 0; i < activation.size(); i++) {
      delta.push_back(activation[i] - expected[i]);
    }
    return delta;
  }

  vector<cpp_dec_float_100> NeuralNetwork::feedForwardWithSave(vector<cpp_dec_float_100> input, vector<vector<cpp_dec_float_100> > * zvecsCont, vector<vector<cpp_dec_float_100> > * activationCont) {
    vector<cpp_dec_float_100> lastResult = input;
    activationCont->push_back(input);
    for(int i = 0; i < this->layers.size(); i++) {
      SigmoidLayer currentLayer = this->layers[i];

      vector<cpp_dec_float_100> zvector = currentLayer.dotAndBiased(lastResult);
      zvecsCont->push_back(zvector);

      vector<cpp_dec_float_100> activation = currentLayer.activations(zvector);
      activationCont->push_back(activation);

      printf("single layer size: %d\n", activation.size());
      for(int j = 0; j < activation.size(); j++) {
	cout << "out = " << activation[j];
      }
      printf("\n");
      lastResult =  activation;
      for(int j = 0; j < lastResult.size(); j++) {
	cout << "lastRes = %f," << lastResult[j];
      }

      printf("\n=======finished(lastResult: %d)=======\n", lastResult.size());
    }
    printf("\nFIN!\n");
    return lastResult;
  }

  vector<cpp_dec_float_100> NeuralNetwork::feedForward(vector<cpp_dec_float_100> input) {
    vector<cpp_dec_float_100> lastResult = vector<cpp_dec_float_100>();
    for(int i = 0; i < this->layers.size(); i++) {
      SigmoidLayer currentLayer = this->layers[i];

      vector<cpp_dec_float_100> zvector = currentLayer.dotAndBiased(lastResult);

      vector<cpp_dec_float_100> activation = currentLayer.activations(zvector);

      printf("single layer size: %d\n", activation.size());
      for(int j = 0; j < activation.size(); j++) {
	cout << "out=" << activation[i];
      }
      cout << "\n";
      lastResult = activation;
    }
    return lastResult;
  }

  NeuralNetwork::~NeuralNetwork() {
  }

} /* namespace sigmoid */
