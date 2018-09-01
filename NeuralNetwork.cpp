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

  vector<float> NeuralNetwork::hadamardProduct(vector<float> a, vector<float> b) {
    vector<float> result = vector<float>(a.size(), 0);
    for(int i = 0; i < a.size(); i++) {
      result[i] = a[i] * b[i];
    }
    return result;
  }

  WeightsAndBiasUpdates NeuralNetwork::backPropagate(vector<float> input, vector<float> expected) {
    vector<vector<float> > * activations = new vector<vector<float> >();
    vector<vector<float> > * zvectors = new vector<vector<float> >();
    this->feedForwardWithSave(input, zvectors, activations);
    vector<float> activation = activations->at(activations->size() - 1);
    vector<float> zvector = zvectors->at(zvectors->size()-1);
    vector<float> costDerivative = this->costDerivative(activation, expected);
    vector<float> sigmoidPrime = this->sigmoidDeriv(zvector);

    vector<float> delta = this->hadamardProduct(costDerivative, sigmoidPrime);

    for(int i = 0; i < delta.size(); i++) {
      printf("new_delta=%f, ", delta[i]);
    }
    printf("\n");

    WeightsAndBiasUpdates updates = WeightsAndBiasUpdates();
    vector<float> biasUpdate = delta;
    vector<vector<float> > weightUpdate = Matrix::transposeAndMultiply(activations->at(activations->size() - 2), biasUpdate);

    updates.addBiasUpdate(biasUpdate);
    updates.addWeightUpdate(weightUpdate);

    int layers = this->layers.size() - 2;
    printf("weightDims={row=%d, col=%d}", weightUpdate.size(), weightUpdate[0].size());

    for(int i = layers; i >= 0; i--) {
      printf("\n=====iter: %d=======\n", i);
      vector<float> zvector = zvectors->at(i);

      SigmoidLayer outputLayer =  this->layers[i];
      vector<vector<float> > prevWeights = outputLayer.getWeights();
      vector<float> sp = this->sigmoidDeriv(zvector);

      for(int k = 0; k < sp.size(); k++) {
	printf("sp=%f, ", sp[k]);
      }
      printf("\n");

      for(int k = 0; k < delta.size(); k++) {
	printf("prev_delta=%f, ", delta[k]);
      }
      printf("\n");

      delta = Matrix::transposeAndMultiplyOneDim(prevWeights, delta);
      printf("\ndeltaSize=%d\n", delta.size());
      printf("\nsigmoidSize=%d\n", sp.size());

      delta = this->hadamardProduct(delta, sp);

      for(int k = 0; k < delta.size(); k++) {
	printf("new_delta=%f, ", delta[k]);
      }
      printf("\n");

      biasUpdate = delta;
      weightUpdate = Matrix::transposeAndMultiply(activations->at(i), biasUpdate);

      printf("weightDims={row=%d, col=%d}", weightUpdate.size(), weightUpdate[0].size());

      for(int k = 0; k < weightUpdate.size(); k++) {
	for(int j = 0; j < weightUpdate[k].size(); j++) {
	  printf("weight=%f, ", weightUpdate[k][j]);
	}
	printf("\n");
      }

      updates.addBiasUpdate(biasUpdate);
      updates.addWeightUpdate(weightUpdate);

      printf("\n====\n");
    }

    return updates;
  }

  vector<float> NeuralNetwork::sigmoidDeriv(vector<float> activation) {
    vector<float> sigPrime = vector<float>();
    for(int i = 0; i < activation.size(); i++) {
      sigPrime.push_back(SigmoidLayer::derivSigmoid(activation[i]));
    }
    return sigPrime;
  }

  vector<float> NeuralNetwork::oneDimVectorMultiply(vector<float> activation, vector<float> expected) {
    vector<float> result = vector<float>();
    for(int i = 0; i < activation.size(); i++) {
      float curAct = activation[i];
      float curExp = expected[i];
      float product = curAct*curExp;
      result.push_back(product);
    }
    return result;
  }

  vector<float> NeuralNetwork::costDerivative(vector<float> activation, vector<float> expected) {
    vector<float> delta = vector<float>();
    for(int i = 0; i < activation.size(); i++) {
      delta.push_back(activation[i] - expected[i]);
    }
    return delta;
  }

  vector<float> NeuralNetwork::feedForwardWithSave(vector<float> input, vector<vector<float> > * zvecsCont, vector<vector<float> > * activationCont) {
    vector<float> lastResult = input;
    activationCont->push_back(input);
    for(int i = 0; i < this->layers.size(); i++) {
      SigmoidLayer currentLayer = this->layers[i];

      vector<float> zvector = currentLayer.dotAndBiased(lastResult);
      zvecsCont->push_back(zvector);

      vector<float> activation = currentLayer.activations(zvector);
      activationCont->push_back(activation);

      printf("single layer size: %d\n", activation.size());
      for(int j = 0; j < activation.size(); j++) {
	printf("out = %f,", activation[j]);
      }
      printf("\n");
      lastResult =  activation;
      for(int j = 0; j < lastResult.size(); j++) {
	printf("lastRes = %f,", lastResult[j]);
      }

      printf("\n=======finished(lastResult: %d)=======\n", lastResult.size());
    }
    printf("\nFIN!\n");
    return lastResult;
  }

  vector<float> NeuralNetwork::feedForward(vector<float> input) {
    vector<float> lastResult = vector<float>();
    for(int i = 0; i < this->layers.size(); i++) {
      SigmoidLayer currentLayer = this->layers[i];

      vector<float> zvector = currentLayer.dotAndBiased(lastResult);

      vector<float> activation = currentLayer.activations(zvector);

      printf("single layer size: %d\n", activation.size());
      for(int j = 0; j < activation.size(); j++) {
	printf("out = %f,", activation[i]);
      }
      printf("\n");
      lastResult = activation;
    }
    return lastResult;
  }

  NeuralNetwork::~NeuralNetwork() {
  }

} /* namespace sigmoid */
