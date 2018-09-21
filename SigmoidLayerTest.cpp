#include "SigmoidLayer.h"
#include "SigmoidLayer.cpp"
#include "Matrix.h"
#include <iostream>

using namespace std;
using namespace sigmoid;


MATRIX test_matrix(double seed) {
  MATRIX matrix = MATRIX();

  double k = 0.0;
  for(int i = 1; i < 10; i++) {
    vector<double> row = vector<double>();
    for(int j = 1; j < 10; j++) {
      row.push_back(((double) 1.0 / ((double) j * i * (k+1) * (seed))));
      k += 1;
    }
    matrix.push_back(row);
  }

  return matrix;
}

MATRIX test_matrix_with_dims(double seed, int rows, int cols) {
  vector<vector<double>> res(rows, vector<double>(cols, 0));
  for(int i = 0; i < rows; i++) {
    vector<double> row(cols, 0);
    for(int j = 0; j < cols; j++){
      row[j] = ((double) 1.0 / ((double) ((j+1) * (i + 1)  * (seed))));
    }
    res[i] = row;
  }
  return res;
}

ROW test_row(double seed) {
  ROW testRow = ROW();
  for(int i = 1; i < 11; i++) {
    testRow.push_back(((double) 1.0 / ((double) i * (seed))));
  }
  return testRow;
}

void isItWithNN() {
  int layers = 3;
  int neurons = {2, 2, 2};
  NeuralNetwork nn(neurons, numlayers);
  MATRIX input = MATRIX(3, row(1, 0));
  input[1][0] = 1;
  Matrix::printMatrix(input);
}

void testZVector() {
  SigmoidLayer test(3, 2);
  cout << test.toString() << "\n";
  MATRIX a = test_matrix_with_dims(0.08, 3, 1);
  MATRIX zvec = test.dotAndBiased(a);
  Matrix::printMatrix(zvec);
  MATRIX activations = test.activations(zvec);
  Matrix::printMatrix(activations);  
}

int main() {
  isItWithNN();
}
