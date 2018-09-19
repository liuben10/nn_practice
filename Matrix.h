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


#ifndef MATRIX_H_
#define MATRIX_H_

//TODO FIX ME transpose is completely broken. That needs to be fixed.
// I was way too baked looking at this code.

//TODO Also, get rid of vector<double> only ever use vector<vector<double>>
// And define a macro for it
namespace sigmoid {

  class Matrix {
  public:
    static vector<double> hadamard(vector<double> a, vector<double> b) {
      vector<double> result = vector<double>(a.size(), 0);
      for(int i = 0; i < a.size(); i++) {
	result[i] = a[i] * b[i];
      }
      return result;
    }

    static vector<double> sum(vector<double> a, vector<double> b) {
      vector<double> result = vector<double>(a.size(), 0);
      for(int i = 0; i < a.size(); i++) {
	result[i] = a[i] * b[i];
      }
      return result;
    }

    static vector<vector<double>> swap(vector<vector<double>> orig, int row1, int row2, int col1, int col2) {
      double tmp = orig[row1][col1];
      orig[row1][col1] = orig[row2][col2];
      orig[row2][col2] = tmp;
      return orig;
    }

    static void printMatrix(vector<vector<double>> matrix) {
      Matrix::printMatrixLabel(matrix, "matrix");
    }

    static void printMatrixLabel(vector<vector<double>> matrix, string label) {
      cout << stringMatrixLabel(matrix, label);
    }

    static string stringMatrixLabel(vector<vector<double>> matrix, string label) {
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
      Matrix::printRowLabel(row, "row");
    }

    static void printRowLabel(vector<double> row, string label) {
      cout << Matrix::stringRowLabel(row, label);
    }

    static string stringRowLabel(vector<double> row, string label) {
      ostringstream output;
      for(int i = 0; i < row.size(); i++) {
	output << label << "=" << row[i] << ", ";
      }
      output << "\n";
      return output.str();
    }
    
				  

    static vector<vector<double>> transposeMatrix(vector<vector<double>> toTranspose) {
      int rows = toTranspose.size();
      int cols = toTranspose[0].size();

      vector<vector<double>> result = vector<vector<double>>(toTranspose);
      int diags = 0;
      for(int i = 0; i < result.size(); i++) {
	for(int j = 0; j < result[0].size(); j++) {
	  if (j > diags) {
	    result = swap(result, i, j, j, i);
	  }
	}
      }

      return result;
    }

    static vector<double> transposeAndMultiplyOneDim(vector<vector<double>> a, vector<double> b) {
      // vector<vector<double>> a = Matrix::transpose(untransposed);
      int m1 = a.size();
      int n1 = a[0].size();
      cout << "a_row_size=" << a[0].size() << ", b_row_size=" << b.size() << "\n";
      vector<double> result = vector<double>(m1, 0);
      for(int i = 0; i < m1; i++) {
	for(int j = 0; j < n1; j++) {
	  double prod = a[i][j] * b[j];
	  result[i] += prod;
	}
      }
      return result;
    }

    static vector<vector<double>> transposeAndMultiply(vector<double> a, vector<double> b) {
      int m1 = a.size();
      int n2 = b.size();
      vector<vector<double>> result = vector<vector<double>>(m1, vector<double>(n2, 0));
      for(int i = 0; i < m1; i++) {
	for(int k = 0; k < n2; k++) {
	  result[i][k] += a[i] * b[k];
	}
      }
      return result;
    }

    static vector<vector<double>> matrixMultiply(vector<vector<double>> a, vector<vector<double>> b) {
      int m1 = a.size();
      int n1 = a[0].size();
      int n2 = b[0].size();
      vector<vector<double>> result = vector<vector<double>>(m1, vector<double>(n2, 0));
      for(int i = 0; i < m1; i++) {
	for(int j = 0; j < n1; j++) {
	  for(int k = 0; k < n2; k++) {
	    result[i][k] += a[i][j] * b[j][k];
	  }
	}
      }
      return result;
    }
  }; /* namespace sigmoid */
}

#endif /* MATRIX_H_ */
