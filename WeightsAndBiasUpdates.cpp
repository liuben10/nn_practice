/*
 * WeightsAndBiasUpdates.cpp
 *
 *  Created on: Jan 4, 2018
 *      Author: liuben10
 */
#include <boost/multiprecision/cpp_dec_float.hpp>

#include "WeightsAndBiasUpdates.h"
#include <string>

using namespace std;
using namespace boost::multiprecision;


namespace sigmoid {

  void WeightsAndBiasUpdates::addWeightUpdate(MATRIX weightUpdate) {
    Matrix::printMatrixLabel(weightUpdate, "weight_update");
    this->weightUpdates.push_back(weightUpdate);
  }
  
  void WeightsAndBiasUpdates::addBiasUpdate(MATRIX biasUpdate) {
    Matrix::printMatrixLabel(biasUpdate, "bias_update");
    this->biasUpdates.push_back(biasUpdate);
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

  WeightsAndBiasUpdates::~WeightsAndBiasUpdates() {
    this->weightUpdates.clear();
    this->biasUpdates.clear();
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
