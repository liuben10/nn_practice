#include "Matrix.h"
#include <iostream>
#include <vector>

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

void testMatrixTranspose() {
  MATRIX matrix = test_matrix(0.9);

  Matrix::printMatrix(matrix);

  MATRIX transposed = Matrix::transpose(matrix);

  Matrix::printMatrix(transposed);
}

void testMatrixTransposeUnmatchedDim() {
  MATRIX matrix = test_matrix_with_dims(0.95, 1, 3);
  Matrix::printMatrix(matrix);

  MATRIX transposed = Matrix::transpose(matrix);
  Matrix::printMatrix(transposed);
}

void testMatrixHadamard() {
  MATRIX a = test_matrix_with_dims(0.9, 3, 1);
  MATRIX b = test_matrix_with_dims(0.8, 3, 1);
  Matrix::printMatrix(a);
  Matrix::printMatrix(b);
  
  MATRIX prod = Matrix::hadamard(a, b);
  Matrix::printMatrix(prod);
}

void testMatrixSum() {
  MATRIX a = test_matrix_with_dims(0.9, 3, 1);
  MATRIX b = test_matrix_with_dims(0.8, 3, 1);
  Matrix::printMatrix(a);
  Matrix::printMatrix(b);
  
  MATRIX sum = Matrix::sum(a, b);
  Matrix::printMatrix(sum);
}

void testMatrixTransposeAndMultiply() {
  MATRIX a = test_matrix_with_dims(0.8, 1, 3);
  Matrix::printMatrix(a);
  MATRIX b = test_matrix_with_dims(2.3, 1, 1);
  Matrix::printMatrix(b);

  MATRIX individual = Matrix::transposeAndMultiply(a, b);
  Matrix::printMatrix(individual);  
}

void testMatrixMultiplyDotProduct() {
  MATRIX a = test_matrix_with_dims(0.8, 1, 3);
  Matrix::printMatrix(a);
  MATRIX b = test_matrix_with_dims(2.3, 3, 1);
  Matrix::printMatrix(b);

  MATRIX individual = Matrix::matrixMultiply(a, b);
  Matrix::printMatrix(individual);
}

void testMatrixMultiplyWeightProduct() {
  MATRIX a = test_matrix_with_dims(0.8, 2, 2);
  MATRIX b = test_matrix_with_dims(0.9, 2, 1);

  MATRIX prod = Matrix::matrixMultiply(a, b);
  Matrix::printMatrix(prod);
}

void testMatrixMultiplySquare() {
  MATRIX a = test_matrix(0.8);
  MATRIX b = test_matrix(0.9);

  MATRIX res = Matrix::matrixMultiply(a, b);
  MATRIX k = test_matrix(0.88);
  MATRIX producted = Matrix::matrixMultiply(res, k);
  MATRIX j = test_matrix(0.98);
  producted = Matrix::matrixMultiply(producted, j);
  Matrix::printMatrix(producted);
}

int main() {
  testMatrixMultiplyWeightProduct();
}
