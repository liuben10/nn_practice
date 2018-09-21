/*
 * Input.h
 *
 *  Created on: Dec 29, 2017
 *      Author: liuben10
 */
#ifndef INPUT_H_
#define INPUT_H_

#include <string>
using namespace std;



namespace sigmoid {

class Input {
private:
	double weight;
	string id;
public:
	Input(string id, double weight) {
			this->weight = weight;
			this->id = id;
	}

	void setWeight(double weight) {
		this->weight = weight;
	}

	double getWeight() {
		return this->weight;
	}

	string getId() {
		return this->id;
	}

	void setId(string id) {
		this->id = id;
	}

	double product(double value) {
		return this->weight * value;
	}

	~Input() {
		free(this);
	}
};

} /* namespace sigmoid */

#endif /* INPUT_H_ */
