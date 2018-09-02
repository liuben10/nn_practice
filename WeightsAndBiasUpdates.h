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
    vector<vector<vector<number<cpp_dec_float<200> > > > > weightUpdates;
    vector<vector<number<cpp_dec_float<200> > > > biasUpdates;
  public:
    void addWeightUpdate(vector<vector<number<cpp_dec_float<200> > > > weightUpdate);
    void addBiasUpdate(vector<number<cpp_dec_float<200> > > biasUpdate);
    vector<vector<number<cpp_dec_float<200> > > > weightAt(int idx);
    vector<number<cpp_dec_float<200> > > biasAt(int idx);
    WeightsAndBiasUpdates();
  
    string toString();
    virtual ~WeightsAndBiasUpdates();
  };

} /* namespace sigmoid */

#endif /* WEIGHTSANDBIASUPDATES_H_ */
