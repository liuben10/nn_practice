/*
 * WeightsAndBiasUpdates.cpp
 *
 *  Created on: Jan 4, 2018
 *      Author: liuben10
 */

#include "WeightsAndBiasUpdates.h"
#include <string>

using namespace std;

namespace sigmoid {

void WeightsAndBiasUpdates::addWeightUpdate(vector<vector<float> > weightUpdate) {
	this->weightUpdates.push_back(weightUpdate);
}
void WeightsAndBiasUpdates::addBiasUpdate(vector<float> biasUpdate) {
	this->biasUpdates.push_back(biasUpdate);
}

WeightsAndBiasUpdates::WeightsAndBiasUpdates() {
	this->weightUpdates = vector<vector<vector<float> > >();
	this->biasUpdates = vector<vector<float> >();
}

WeightsAndBiasUpdates::~WeightsAndBiasUpdates() {
	this->weightUpdates.clear();
	this->biasUpdates.clear();
}

string WeightsAndBiasUpdates::toString() {
	string updatesString;
	updatesString.append("WeightUpdate: \n");
	for(int i = 0; i < this->weightUpdates.size(); i++) {
		string updateBlock;
		for(int j = 0; j < this->weightUpdates[i].size(); j++) {
			string singleUpdateRow;
			for(int k = 0; k < this->weightUpdates[i][j].size(); k++) {
				float elem = this->weightUpdates[i][j][k];
				string stringified = to_string(elem);
				singleUpdateRow.append(stringified);
				singleUpdateRow.append(",");
			}
			singleUpdateRow.append("\n");
			updateBlock.append(singleUpdateRow);
		}
		updateBlock.append("\n======\n");
		updatesString.append(updateBlock);
	}

	updatesString.append("\n BiasUpdates:\n");
	for(int i = 0; i < this->biasUpdates.size(); i++) {
		string singleBiasUpdates;
		for(int j = 0; j < this->biasUpdates[i].size(); j++) {
			string elem = to_string(this->biasUpdates[i][j]);
			singleBiasUpdates.append(elem);
			singleBiasUpdates.append(", ");
		}
		singleBiasUpdates.append("\n");
		updatesString.append(singleBiasUpdates);
	}

	return updatesString;

}

}
