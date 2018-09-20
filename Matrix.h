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

#define MATRIX vector<vector<double>>

#define ROW vector<double>

//TODO FIX ME transpose is completely broken. That needs to be fixed.
// I was way too baked looking at this code.

//TODO Also, get rid of vector<double> only ever use MATRIX
// And define a macro for it
namespace sigmoid {

  class Matrix {
  public:
    static MATRIX hadamard(MATRIX a, MATRIX b) {
      MATRIX res(a);
      for(int i = 0; i < a.size(); i++) {
	for(int j = 0; j < a[i].size(); j++) {
	  res[i][j] *= b[i][j];
	}
      }
      return res;
    }

    static MATRIX sum(MATRIX a, MATRIX b) {
      MATRIX result = MATRIX(a.size(), vector<double>(a[0].size(), 0));
      for(int i = 0; i < a.size(); i++) {
	vector<double> row(a[i].size(), 0);
	for(int j = 0; j < b.size(); j++) {
	  result[i][j] = a[i][j] + b[i][j];
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
      Matrix::printMatrixLabel(matrix, "matrix");
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

    static MATRIX transposeAndMultiply(MATRIX a, MATRIX b) {
      int m1 = a.size();
      int n2 = b.size();
      MATRIX transposed = Matrix::transpose(b);
      return Matrix::matrixMultiply(a, transposed);
    }

    static MATRIX matrixMultiply(MATRIX a, MATRIX b) {
      Matrix::printMatrix(a);
      Matrix::printMatrix(b);
      int arows = a.size();
      int acols = a[0].size();
      int bcols = b[0].size();
      if (acols != b.size()) {
	cerr << "ERROR mismatched dims, acols=" << a[0].size() << " brows=" << b.size() << "\n";
	throw "Error, dims do not match for matrix multiply!";
      }
      MATRIX result = MATRIX(arows, vector<double>(bcols, 0));
      for(int i = 0; i < arows; i++) {
	for(int j = 0; j < acols; j++) {
	  for(int k = 0; k < bcols; k++) {
	    result[i][k] += a[i][j] * b[j][k];
	  }
	}
      }
      return result;
    }
  }; /* namespace sigmoid */ 
}

#endif /* MATRIX_H_ */
