/*
 * Input.h
 *
 *  Created on: Dec 29, 2017
 *      Author: liuben10
 */
#include <boost/multiprecision/cpp_dec_float.hpp>
#ifndef INPUT_H_
#define INPUT_H_

#include <string>
using namespace std;
using namespace boost::multiprecision;


namespace sigmoid {

class Input {
private:
	number<cpp_dec_float<300> >    weight;
	string id;
public:
	Input(string id, number<cpp_dec_float<300> >    weight) {
			this->weight = weight;
			this->id = id;
	}

	void setWeight(number<cpp_dec_float<300> >    weight) {
		this->weight = weight;
	}

	number<cpp_dec_float<300> >    getWeight() {
		return this->weight;
	}

	string getId() {
		return this->id;
	}

	void setId(string id) {
		this->id = id;
	}

	number<cpp_dec_float<300> >    product(number<cpp_dec_float<300> >    value) {
		return this->weight * value;
	}

	~Input() {
		free(this);
	}
};

} /* namespace sigmoid */

#endif /* INPUT_H_ */
