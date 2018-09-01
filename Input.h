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
	cpp_dec_float_100 weight;
	string id;
public:
	Input(string id, cpp_dec_float_100 weight) {
			this->weight = weight;
			this->id = id;
	}

	void setWeight(cpp_dec_float_100 weight) {
		this->weight = weight;
	}

	cpp_dec_float_100 getWeight() {
		return this->weight;
	}

	string getId() {
		return this->id;
	}

	void setId(string id) {
		this->id = id;
	}

	cpp_dec_float_100 product(cpp_dec_float_100 value) {
		return this->weight * value;
	}

	~Input() {
		free(this);
	}
};

} /* namespace sigmoid */

#endif /* INPUT_H_ */
