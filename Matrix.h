/*
 * Matrix.h
 *
 *  Created on: Jan 2, 2018
 *      Author: liuben10
 */
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
using namespace std;

#define LOG 1


#ifndef MATRIX_H_
#define MATRIX_H_

#define MATRIX vector<vector<double>>

#define ROW vector<double>

namespace sigmoid {

  class Matrix {
  public:
    static MATRIX hadamard(MATRIX a, MATRIX b) {
      if (a.size() != b.size() || a[0].size() != b[0].size()) {
	cerr << "Hadamard dimensions do not match" << endl;
	throw "Hadamard dimensions do not match";
      }
      for(int i = 0; i < a.size(); i++) {
	for(int j = 0; j < a[i].size(); j++) {
	  a[i][j] *= b[i][j];
	}
      }
      return a;
    }

    static MATRIX sum(MATRIX a, MATRIX b) {
      if (a.size() != b.size() || a[0].size() != b[0].size()) {
	cerr << "sum dimensions do not match" << endl;
	throw "sum dimensions do not match";
      }
      MATRIX result = MATRIX(a);
      for(int i = 0; i < a.size(); i++) {
	for(int j = 0; j < a[0].size(); j++) {
	  result[i][j] += b[i][j];
	}
      }
      return result;
    }

    static MATRIX swap(MATRIX orig, int row1, int row2, int col1, int col2) {
      double tmp = orig[row1][col1];
      orig[row1][col1] = orig[row2][col2];
      orig[row2][col2] = tmp;
      return orig;
    }

    static void printMatrix(MATRIX matrix) {
      if (LOG) {
	Matrix::printMatrixLabel(matrix, "matrix");
      }
    }

    static void printMatrixSmall(MATRIX matrix) {
      if (LOG) {
	Matrix::printMatrixSmallLabel(matrix, "matrix");
      }
    }

    static void printMatrixSmallLabel(MATRIX matrix, string label) {
      if (LOG) {
	cout << Matrix::stringMatrixSmallLabel(matrix, label);
      }
    }

    static string stringMatrixSmallLabel(MATRIX matrix, string label) {
      ostringstream o;
      o << label << "_dims=" << matrix.size() << "x" << matrix[0].size() << "\n";
      return o.str();
    }

    static void printMatrixLabel(MATRIX matrix, string label) {
      cout << stringMatrixLabel(matrix, label);
    }

    static string stringMatrixLabel(MATRIX matrix, string label) {
      ostringstream output;
      for(int i = 0; i < matrix.size(); i++) {
	for(int j = 0; j < matrix[i].size(); j++) {
	  output << label << "=" << matrix[i][j] << ", ";
	}
	output << "\n";
      }
      output << "\n";
      return output.str();
    }

    static void printRow(vector<double> row) {
      if (LOG) {
	Matrix::printRowLabel(row, "row");
      }
    }

    static void printRowLabel(vector<double> row, string label) {
      if (LOG) {
	cout << Matrix::stringRowLabel(row, label);
      }
    }

    static void printMatrixSucc(MATRIX input) {
      for(int i = 0; i < input.size(); i++) {
	for(int j = 0; j < input[0].size(); j++) {
	  cout << input[i][j] << ", ";
	}
	cout << endl;
      }
    }

    static string stringRowLabel(vector<double> row, string label) {
      ostringstream output;
      for(int i = 0; i < row.size(); i++) {
	output << label << "=" << row[i] << ", ";
      }
      output << "\n";
      return output.str();
    }
				  
    static MATRIX transpose(MATRIX toTranspose) {
      int rows = toTranspose.size();
      int cols = toTranspose[0].size();
      MATRIX transposed(cols, vector<double>(rows, 0));
      for(int r = 0; r < rows; r++) {
	for(int c = 0; c < cols; c++) {
	  transposed[c][r] = toTranspose[r][c];
	}
      }

      return transposed;
    }

    static MATRIX multiplyByScalar(MATRIX a, double scalar) {
      for(int i = 0; i < a.size(); i++) {
	for(int j = 0; j < a[0].size(); j++) {
	  a[i][j] = a[i][j] * scalar;
	}
      }
      return a;
    }

    static MATRIX transposeAndMultiply(MATRIX a, MATRIX b) {
      int m1 = a.size();
      int n2 = b.size();
      MATRIX transposed = Matrix::transpose(b);
      return Matrix::matrixMultiply(a, transposed);
    }

    static MATRIX matrixMultiply(MATRIX a, MATRIX b) {
      Matrix::printMatrixSmall(a);
      Matrix::printMatrixSmall(b);
      int arows = a.size();
      int acols = a[0].size();
      int bcols = b[0].size();
      if (acols != b.size()) {
	cerr << "ERROR mismatched dims, acols=" << a[0].size() << " brows=" << b.size() << "\n";
	throw "Error, dims do not match for matrix multiply!";
      }
      MATRIX result = MATRIX(arows, vector<double>(bcols, 0));
      for(int i = 0; i < arows; i++) {
	for(int k = 0; k < bcols; k++) {
	  for(int j = 0; j < acols; j++) {
	    result[i][k] += a[i][j] * b[j][k];
	  }
	}
      }
      return result;
    }
  }; /* namespace sigmoid */ 
}

#endif /* MATRIX_H_ */
