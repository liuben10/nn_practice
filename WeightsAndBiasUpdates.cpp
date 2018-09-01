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

  void WeightsAndBiasUpdates::addWeightUpdate(vector<vector<cpp_dec_float_100> > weightUpdate) {
    this->weightUpdates.push_back(weightUpdate);
  }
  void WeightsAndBiasUpdates::addBiasUpdate(vector<cpp_dec_float_100> biasUpdate) {
    this->biasUpdates.push_back(biasUpdate);
  }

  WeightsAndBiasUpdates::WeightsAndBiasUpdates() {
    this->weightUpdates = vector<vector<vector<cpp_dec_float_100> > >();
    this->biasUpdates = vector<vector<cpp_dec_float_100> >();
  }

  WeightsAndBiasUpdates::~WeightsAndBiasUpdates() {
    this->weightUpdates.clear();
    this->biasUpdates.clear();
  }

  string WeightsAndBiasUpdates::toString() {
    ostringstream o;
    o << "WeightUpdate: \n";
    for(int i = 0; i < this->weightUpdates.size(); i++) {
      for(int j = 0; j < this->weightUpdates[i].size(); j++) {
	for(int k = 0; k < this->weightUpdates[i][j].size(); k++) {
	  cpp_dec_float_100 elem = this->weightUpdates[i][j][k];
	  o << elem << ", ";
	}
	o << "\n";
      }
      o << "\n======\n";
    }

    o << "\n BiasUpdates:\n";
    for(int i = 0; i < this->biasUpdates.size(); i++) {
      for(int j = 0; j < this->biasUpdates[i].size(); j++) {
	o << this->biasUpdates[i][j] << ", ";
      }
      o << "\n";
    }

    return o.str();
  }

}
