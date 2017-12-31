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
	float weight;
	string id;
public:
	Input(string id, float weight) {
			this->weight = weight;
			this->id = id;
	}

	void setWeight(float weight) {
		this->weight = weight;
	}

	float getWeight() {
		return this->weight;
	}

	string getId() {
		return this->id;
	}

	void setId(string id) {
		this->id = id;
	}

	float product(float value) {
		return this->weight * value;
	}

	~Input() {
		free(this);
	}
};

} /* namespace sigmoid */

#endif /* INPUT_H_ */
