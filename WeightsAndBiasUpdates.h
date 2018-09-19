/*
 * WeightsAndBiasUpdates.h
 *
 *  Created on: Jan 4, 2018
 *      Author: liuben10
 */
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <vector>

using namespace std;
using namespace boost::multiprecision;


#ifndef WEIGHTSANDBIASUPDATES_H_
#define WEIGHTSANDBIASUPDATES_H_

namespace sigmoid {

  class WeightsAndBiasUpdates {
  private:
    vector<vector<vector<double> > > weightUpdates;
    vector<vector<double> > biasUpdates;
  public:
    void addWeightUpdate(vector<vector<double> > weightUpdate);
    void addBiasUpdate(vector<double> biasUpdate);
    vector<vector<double> > weightAt(int idx);
    vector<double> biasAt(int idx);
    WeightsAndBiasUpdates();
  
    string toString();
    virtual ~WeightsAndBiasUpdates();
  };

} /* namespace sigmoid */

#endif /* WEIGHTSANDBIASUPDATES_H_ */
