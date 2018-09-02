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
    this->layers = vector<SigmoidLayer *>();

    for(int i = 0; i < layers-1; i++) {
      SigmoidLayer *sl = new SigmoidLayer(neurons[i], neurons[i+1]);
      cout << sl->toString() << "\n";
      this->layers.push_back(sl);
    }
  }

  void NeuralNetwork::printNetwork() {
    for(int i = 0; i < layers.size(); i++) {
      SigmoidLayer *sl = this->layers[i];
      cout << sl->toString() << "\n";
    }
  }

  
  vector<number<cpp_dec_float<300> >   > NeuralNetwork::hadamardProduct(vector<number<cpp_dec_float<300> >   > a, vector<number<cpp_dec_float<300> >   > b) {
    vector<number<cpp_dec_float<300> >   > result(a.begin(), a.end());
    for(int i = 0; i < a.size(); i++) {
      result[i] *=  b[i];
    }
    return result;
  }

  void printCol(vector<number<cpp_dec_float<300> >   > * col, string label) {
    cout << "\n==" << label << "_col" << "===\n";
    for(int i = 0; i < col->size(); i++) {
      cout << label << "=" << col->at(i) << ", ";
    }
    cout << "\n=====\n";
  }

  void printMatrix(vector<vector<number<cpp_dec_float<300> >   > > * matrix, string label) {
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

  WeightsAndBiasUpdates NeuralNetwork::backPropagate(vector<number<cpp_dec_float<300> >   > input, vector<number<cpp_dec_float<300> >   > expected) {
    for(int i = 0; i < layers.size(); i++) {
      SigmoidLayer *sl = this->layers[i];
      cout << sl->toString() << "\n";
    }
    vector<vector<number<cpp_dec_float<300> >   > > * activations = new vector<vector<number<cpp_dec_float<300> >   > >();
    vector<vector<number<cpp_dec_float<300> >   > > * zvectors = new vector<vector<number<cpp_dec_float<300> >   > >();
    this->feedForwardWithSave(input, zvectors, activations);
    printMatrix(activations, "activation");
    printMatrix(zvectors, "zscore");
    
    vector<number<cpp_dec_float<300> >   > activation = activations->at(activations->size() - 1);
    vector<number<cpp_dec_float<300> >   > zvector = zvectors->at(zvectors->size()-1);
    vector<number<cpp_dec_float<300> >   > costDerivative = this->costDerivative(activation, expected);
    printCol(&costDerivative, "firstCostDerivative");
    vector<number<cpp_dec_float<300> >   > sigmoidPrime = this->sigmoidDeriv(zvector);
    printCol(&sigmoidPrime, "firstSigmoidPrime");
    
    vector<number<cpp_dec_float<300> >   > delta = this->hadamardProduct(costDerivative, sigmoidPrime);

    printCol(&delta, "delta");

    WeightsAndBiasUpdates updates = WeightsAndBiasUpdates();
    vector<number<cpp_dec_float<300> >   > biasUpdate = delta;
    vector<vector<number<cpp_dec_float<300> >   > > weightUpdate = Matrix::transposeAndMultiply(activations->at(activations->size() - 2), delta);

    printMatrix(&weightUpdate, "newWeightUpdate");

    updates.addBiasUpdate(biasUpdate);
    updates.addWeightUpdate(weightUpdate);

    int layers = this->layers.size() - 2;
    printf("weightDims={row=%d, col=%d}", weightUpdate.size(), weightUpdate[0].size());

    for(int i = layers; i >= 0; i--) {
      printf("\n=====iter: %d=======\n", i);
      vector<number<cpp_dec_float<300> >   > zvector = zvectors->at(i);

      SigmoidLayer *outputLayer =  this->layers[i+1];
      vector<vector<number<cpp_dec_float<300> >   > > prevWeights = outputLayer->getWeights();
      vector<number<cpp_dec_float<300> >   > sp = this->sigmoidDeriv(zvector);

      printCol(&sp, "sigmoid_prime");

      printCol(&delta, "prev_delta");

      printMatrix(&prevWeights, "prev_weights");

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

  vector<number<cpp_dec_float<300> >   > NeuralNetwork::sigmoidDeriv(vector<number<cpp_dec_float<300> >   > zvector) {
    vector<number<cpp_dec_float<300> >   > sigPrime = vector<number<cpp_dec_float<300> >   >(zvector.begin(), zvector.end());
    for(int i = 0; i < zvector.size(); i++) {
      sigPrime[i] = SigmoidLayer::derivSigmoid(zvector[i]);
    }
    return sigPrime;
  }

  vector<number<cpp_dec_float<300> >   > NeuralNetwork::oneDimVectorMultiply(vector<number<cpp_dec_float<300> >   > activation, vector<number<cpp_dec_float<300> >   > expected) {
    vector<number<cpp_dec_float<300> >   > result = vector<number<cpp_dec_float<300> >   >();
    for(int i = 0; i < activation.size(); i++) {
      number<cpp_dec_float<300> >    curAct = activation[i];
      number<cpp_dec_float<300> >    curExp = expected[i];
      number<cpp_dec_float<300> >    product = curAct*curExp;
      result.push_back(product);
    }
    return result;
  }

  vector<number<cpp_dec_float<300> >   > NeuralNetwork::costDerivative(vector<number<cpp_dec_float<300> >   > activation, vector<number<cpp_dec_float<300> >   > expected) {
    vector<number<cpp_dec_float<300> >   > delta = vector<number<cpp_dec_float<300> >   >(activation.begin(), activation.end());
    for(int i = 0; i < activation.size(); i++) {
      delta[i] = 2 * (activation[i] - expected[i]);
    }
    return delta;
  }

  vector<number<cpp_dec_float<300> >   > NeuralNetwork::feedForwardWithSave(vector<number<cpp_dec_float<300> >   > input, vector<vector<number<cpp_dec_float<300> >   > > * zvecsCont, vector<vector<number<cpp_dec_float<300> >   > > * activationCont) {
    vector<number<cpp_dec_float<300> >   > lastResult = input;
    for(int i = 0; i < this->layers.size(); i++) {
      SigmoidLayer *currentLayer = this->layers[i];

      vector<number<cpp_dec_float<300> >   > zvector = currentLayer->dotAndBiased(lastResult);
      zvecsCont->push_back(zvector);

      vector<number<cpp_dec_float<300> >   > activation = currentLayer->activations(zvector);
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

  vector<number<cpp_dec_float<300> >   > NeuralNetwork::feedForward(vector<number<cpp_dec_float<300> >   > input) {
    vector<number<cpp_dec_float<300> >   > lastResult = vector<number<cpp_dec_float<300> >   >();
    for(int i = 0; i < this->layers.size(); i++) {
      SigmoidLayer *currentLayer = this->layers[i];

      vector<number<cpp_dec_float<300> >   > zvector = currentLayer->dotAndBiased(lastResult);

      vector<number<cpp_dec_float<300> >   > activation = currentLayer->activations(zvector);

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
