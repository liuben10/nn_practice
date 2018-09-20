/*
 * WeightsAndBiasUpdates.h
 *
 *  Created on: Jan 4, 2018
 *      Author: liuben10
 */
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <vector>

#include "Matrix.h"

using namespace std;
using namespace boost::multiprecision;


#ifndef WEIGHTSANDBIASUPDATES_H_
#define WEIGHTSANDBIASUPDATES_H_

namespace sigmoid {

  class WeightsAndBiasUpdates {
  private:
    vector<MATRIX> weightUpdates;
    vector<MATRIX> biasUpdates;
  public:
    void addWeightUpdate(MATRIX weightUpdate);
    void addBiasUpdate(MATRIX biasUpdate);
    MATRIX weightAt(int idx);
    MATRIX biasAt(int idx);
    WeightsAndBiasUpdates();
    vector<MATRIX> getWeightUpdates();
    vector<MATRIX> getBiasUpdates();
  
    string toString();
    virtual ~WeightsAndBiasUpdates();
  };

} /* namespace sigmoid */

#endif /* WEIGHTSANDBIASUPDATES_H_ */
