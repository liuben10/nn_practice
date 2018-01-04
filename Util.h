/*
 * Util.h
 *
 *  Created on: Dec 31, 2017
 *      Author: liuben10
 */
#include <iostream>
#include <fstream>
#include <vector>

#ifndef UTIL_H_
#define UTIL_H_

#include <stdlib.h>
using namespace std;

namespace sigmoid {

class Util {
public:
	template<typename T, typename... Args>
	static std::unique_ptr<T> make_unique(Args&&... args)
	{
	    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
	}
	Util();
	virtual ~Util();
};

} /* namespace sigmoid */

#endif /* UTIL_H_ */
