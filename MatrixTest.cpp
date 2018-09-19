#include "Matrix.h"
#include <iostream>
#include <vector>

using namespace std;
using namespace sigmoid;

#define MATRIX vector<vector<double>>

#define ROW vector<double>

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

ROW test_row(double seed) {
  ROW testRow = ROW();
  for(int i = 1; i < 11; i++) {
    testRow.push_back(((double) 1.0 / ((double) i * (seed))));
  }
  return testRow;
}

void testMatrixTranspose() {
  MATRIX matrix = test_matrix(0.9);

  Matrix::printMatrix(matrix);

  MATRIX transposed = Matrix::transpose(matrix);

  Matrix::printMatrix(transposed);
}

void testMatrixTransposeAndMultiply() {
  MATRIX a = test_matrix(0.3);
  Matrix::printMatrix(a);
  ROW b = test_row(2.3);
  Matrix::printRow(b);

  ROW individual = Matrix::transposeAndMultiplyOneDim(a, b);
  Matrix::printRow(individual);  
}

void testMatrixMultiply() {
  MATRIX a = test_matrix(0.8);
  MATRIX b = test_matrix(0.9);

  MATRIX res = Matrix::matrixMultiply(a, b);
  MATRIX k = test_matrix(0.88);
  MATRIX producted = Matrix::matrixMultiply(res, k);
  MATRIX j = test_matrix(0.98);
  producted = Matrix::matrixMultiply(producted, j);
  Matrix::printMatrix(producted);
}

void testTransposeAndMultiplyOneDim() {
  MATRIX prevWeights = test_matrix(0.95);
  ROW delta = test_row(0.89);
  ROW multiplied = Matrix::transposeAndMultiplyOneDim(prevWeights, delta);
  Matrix::printRow(multiplied);
}

int main() {
  // testTransposeAndMultiplyOneDim();
  double negw1 = -0.247947;
  double negw2 = -0.0814944;
  double delta = 0.305414;

  double res = negw1 * delta + negw2 * delta;
  cout << res << "\n";
}
