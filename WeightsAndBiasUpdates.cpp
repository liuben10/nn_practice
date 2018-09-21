/*
 * WeightsAndBiasUpdates.cpp
 *
 *  Created on: Jan 4, 2018
 *      Author: liuben10
 */
#include "WeightsAndBiasUpdates.h"
#include <string>

using namespace std;

namespace sigmoid {

  vector<MATRIX> push_front(vector<MATRIX> container, MATRIX update) {
    vector<MATRIX> copy = vector<MATRIX>(container.size() + 1);
    copy[0] = update;
    for(int i = 0; i < container.size(); i++) {
      copy[i+1] = container[i];
    }
    return copy;
  }

  void WeightsAndBiasUpdates::addWeightUpdate(MATRIX weightUpdate) {
    Matrix::printMatrixLabel(weightUpdate, "weight_update");
    this->weightUpdates = push_front(this->weightUpdates, weightUpdate);
  }
  
  void WeightsAndBiasUpdates::addBiasUpdate(MATRIX biasUpdate) {
    Matrix::printMatrixLabel(biasUpdate, "bias_update");
    this->biasUpdates = push_front(this->biasUpdates, biasUpdate);
  }

  MATRIX WeightsAndBiasUpdates::weightAt(int idx) {
    return this->weightUpdates[idx];
  }

  MATRIX WeightsAndBiasUpdates::biasAt(int idx) {
    return this->biasUpdates[idx];
  }

  vector<MATRIX> WeightsAndBiasUpdates::getWeightUpdates() {
    return this->weightUpdates;
  }

  vector<MATRIX> WeightsAndBiasUpdates::getBiasUpdates() {
    return this->biasUpdates;
  }

  WeightsAndBiasUpdates::WeightsAndBiasUpdates() {
    this->weightUpdates = vector<MATRIX>();
    this->biasUpdates = vector<MATRIX>();
  }

  string WeightsAndBiasUpdates::toString() {
    ostringstream o;
    cout << "WEIGHTANDBIASUPDATES" << "\n---\n";
    for(int i = 0; i < this->weightUpdates.size(); i++) {
      o << Matrix::stringMatrixLabel(this->weightUpdates[i], "WeightUpdate") << "\n==\n";
    }

    for(int i = 0; i < this->biasUpdates.size(); i++) {
      o << Matrix::stringMatrixLabel(this->biasUpdates[i], "BiasUpdate") << "\n==\n";
    }
    o << "\n";

    return o.str();
  }

}
