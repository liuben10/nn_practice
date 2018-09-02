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

  void WeightsAndBiasUpdates::addWeightUpdate(vector<vector<number<cpp_dec_float<300> >   > > weightUpdate) {
    this->weightUpdates.push_back(weightUpdate);
  }
  void WeightsAndBiasUpdates::addBiasUpdate(vector<number<cpp_dec_float<300> >   > biasUpdate) {
    this->biasUpdates.push_back(biasUpdate);
  }

  vector<vector<number<cpp_dec_float<300> >   > > WeightsAndBiasUpdates::weightAt(int idx) {
    return this->weightUpdates[idx];
  }

  vector<number<cpp_dec_float<300> >   > WeightsAndBiasUpdates::biasAt(int idx) {
    return this->biasUpdates[idx];
  }

  WeightsAndBiasUpdates::WeightsAndBiasUpdates() {
    this->weightUpdates = vector<vector<vector<number<cpp_dec_float<300> >   > > >();
    this->biasUpdates = vector<vector<number<cpp_dec_float<300> >   > >();
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
	  number<cpp_dec_float<300> >    elem = this->weightUpdates[i][j][k];
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
