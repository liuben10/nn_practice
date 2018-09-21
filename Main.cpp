#include <iterator>
#include <algorithm>

#include <iostream>
#include <fstream>
#include <vector>
#include "NeuralNetwork.h"
#include "Coster.h"
#include "WeightsAndBiasUpdates.h"
//#include "Util.h"

using namespace std;
using namespace sigmoid;



typedef unsigned char uchar;

class Wrapper {
private:
  int expected;
  MATRIX *values;
public:
  int getExpected() {
    return this->expected;
  }

  MATRIX * getValues() {
    return this->values;
  }

  Wrapper(int expected,MATRIX *values) {
    this->expected = expected;
    this->values = values;
  }
  ~Wrapper() {
  }
};

int reverseInt(int i) {
  unsigned char c1, c2, c3, c4;
  c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
};

uchar** read_mnist_images(string full_path, int& number_of_images, int& image_size) {

  /*
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000 or 60000   number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel

    Pixels are organized row-wise. Pixel values are 0 to 255.
    0 means background (white), 255 means foreground (black).
  */


  ifstream file(full_path);

  if(file.is_open()) {
    int magic_number = 0, n_rows = 0, n_cols = 0;

    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

    file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
    file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
    file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

    image_size = n_rows * n_cols;

    uchar** _dataset = new uchar*[number_of_images];
    for(int i = 0; i < number_of_images; i++) {
      _dataset[i] = new uchar[image_size];
      file.read((char *)_dataset[i], image_size);
    }

    // _dataset[number_of_images][image_size]
    return _dataset;
  } else {
    throw runtime_error("Cannot open file `" + full_path + "`!");
  }
}

uchar* read_mnist_labels(string full_path, int number_of_images) {

  /*
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000 or 60000   number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label

    The labels values are 0 to 9.
  */

  // Users of Intel processors and other low-endian machines must flip the bytes of the header.

  ifstream file(full_path, fstream::in);

  if(file.is_open()) {
    int magic_number = 0, n_images = 0;

    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    if(magic_number != 2049) throw runtime_error("Invalid MNIST image file!");

    file.read((char *)&n_images, sizeof(n_images)), n_images = reverseInt(n_images);

    if(number_of_images != n_images) throw runtime_error("Labels don't correspond to the previous dataset!");

    uchar* _labels = new uchar[number_of_images]();

    uchar uch;
    int i = 0;

    while (file >> noskipws >> uch) {
      _labels[i++] = uch;
    }

    // _labels[number_of_images]
    return _labels;
  } else {
    throw runtime_error("Cannot open file `" + full_path + "`!");
  }
}

Wrapper showRandomCharacterInBinary(uchar **dataset, uchar *labels, int number_of_images) {

  // show a random character
  int ind;

  /* initialize random seed: */
  srand ( time(NULL) );

  /* generate secret number: */
  ind = rand() % number_of_images;
  //ind = 59879;

  cout << "" << endl;
  cout << "Opening a  example: " << endl;
  cout << +labels[ind] << endl;
  cout << "ind: " << ind << endl;
  cout << "" << endl;

  int expected = +labels[ind];
  MATRIX * values = new MATRIX(784, ROW(1, 0));

  Wrapper result(expected, values);

  // 28 rows
  for(int i = 0; i < 28; i++) {
    // 28 cols
    for(int j = 0; j < 28; j++) {
      double cur = dataset[ind][i*28 + j];
      if(dataset[ind][i*28+j] > 80) {
	cout << 1;
	result.getValues()->at(i*28 + j) = vector<double>(1, cur);
      } else {
	cout << 0;
	result.getValues()->at(i*28 + j) = vector<double>(1, cur);
      }
    }
    cout << "" << endl;
  }

  return result;
}


// void checkSigmoid() {
//   SigmoidLayer * sl = new SigmoidLayer(10, 4);
//   MATRIX weights;

//   double firstRow[10] = {0, 10, 0, 10, 0, 10, 0, 10, 0, 10};
//   ROW firstRowVec(firstRow, firstRow + (sizeof(firstRow) / sizeof(firstRow[0])) );
//   weights.push_back(firstRowVec);

//   double secondRow[10] = {0, 0, 10, 10, 0, 0, 10, 10, 0, 0};
//   ROW secondRowVec(secondRow, secondRow + (sizeof(secondRow) / sizeof(secondRow[0])) );
//   weights.push_back(secondRowVec);

//   double thirdRow[10] = {0, 0, 0, 0, 10, 10, 10, 10, 0, 0};
//   ROW thirdRowVec(thirdRow, thirdRow + (sizeof(thirdRow) / sizeof(thirdRow[0])) );
//   weights.push_back(thirdRowVec);

//   double fourthRow[10] = {0, 0, 0, 0, 0, 0, 0, 0, 10, 10};
//   ROW fourthRowVec(fourthRow, fourthRow + (sizeof(fourthRow) / sizeof(fourthRow[0])) );
//   weights.push_back(fourthRowVec);

//   sl->setWeights(weights);

//   double biases[4] = {-5, -5, -5, -5};
//   MATRIX biasVec(biases, biases + (sizeof(biases) / sizeof(biases[0])));

//   sl->setBiases(biasVec);

//   double inputs[10] = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
//   ROW inputsVec(inputs, inputs + (sizeof(inputs) / sizeof(inputs[0])));

//   ROW activations = sl->dotAndBiased(inputsVec);
// }

void checkSigmoidVeryEasy() {
  int numLayers = 4;
  int neurons[4] = {4, 3, 3, 2};
  NeuralNetwork nn = NeuralNetwork(neurons, numLayers);
  MATRIX input = MATRIX(4, ROW(1, 0));
  input[1][0] = 1;
  input[3][0] = 1;
  MATRIX activations = nn.feedForward(input);
  Matrix::printMatrix(activations);
}

void checkSigmoidSafe() {
  int numLayers = 3;
  int neurons[3] = {3, 3, 2};
  NeuralNetwork nn = NeuralNetwork(neurons, numLayers);
  ROW errors = ROW();
  for(int i = 0; i < 1000; i++) {
    cout << "###########################################" << "\n";
    cout << "###########################################" << "\n";
    cout << "###########################################" << "\n";
    cout << "ITERATION:  " << i << "\n";
    cout << "###########################################" << "\n";
    cout << "###########################################" << "\n";
    cout << "###########################################" << "\n";

    
    MATRIX input = MATRIX(3, ROW(1, 0.0));
    MATRIX expected = MATRIX(2, ROW(1, 0));
    input[i%3][0] = 1;
    expected[i%2][0] = 1;

    
    if (i % 25 == 0) {
      MATRIX output = nn.feedForward(input);
      double error = Coster::evaluate(output, expected);
      cout << "Total Error at iteration (" << i << ")="
	   << error;
      errors.push_back(error);
    }

    
    WeightsAndBiasUpdates wb = nn.backPropagate(input, expected);
    nn.applyUpdates(wb);
    cout << "\n" << wb.toString() << "\n\n Post Update \n\n";
    nn.printNetwork();
  }
  
  MATRIX input = MATRIX(3, ROW(1, 0));
  MATRIX expected = MATRIX(2, ROW(1, 0));
  
  input[1] = ROW(1, 1);
  expected[1] = ROW(1, 1);

  MATRIX prediction = nn.feedForward(input);
  Matrix::printMatrixLabel(prediction, "output");
  
  cout << "Total Error=" 
       << Coster::evaluate(prediction, expected)
       << "\n\n";
  Matrix::printRowLabel(errors, "errors");
}

template<typename T, typename... Args>
static std::unique_ptr<T> make_unique(Args&&... args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

void checkMain() {
  int number_of_images;
  int image_size;
  int BATCH_SIZE = 100;

  /*
    TRAINING SET - 60 000 images
  */

  uchar **dataset = read_mnist_images("/Users/liuben10/Downloads/train-images-idx3-ubyte", number_of_images, image_size);
  // Read Labels
  uchar *labels = read_mnist_labels("/Users/liuben10/Downloads/train-labels-idx1-ubyte", number_of_images);

  
  int numLayers = 4;
  int sigmoidLayers[4] = {784, 16, 16, 10};
  NeuralNetwork network = NeuralNetwork(sigmoidLayers, numLayers);

  // for(int k = 0; k < 5; k++) {
  //   cout << "Training :" << k << "\n";
  //   Wrapper input = showRandomCharacterInBinary(dataset, labels, number_of_images);

  //   //       ROW output = network.feedForward(input->getValues());

  //   MATRIX expectedBin = Coster::toBinary(input.getExpected());

  //   WeightsAndBiasUpdates updates = network.backPropagate(*(input.getValues()), expectedBin);

  //   std::cout << "\n\n==Updates==\n\n" <<  updates.toString() << "\n\n\n\n";

  //   network.applyUpdates(updates);

  //   //        printf("\n\n == k: %d == \n\n", k);
  // }

  Wrapper input = showRandomCharacterInBinary(dataset, labels, number_of_images);
  

  MATRIX output = network.feedForward(*(input.getValues()));

  for(int i = 0; i < output.size(); i++) {
    cout << output[i][0] << "\n";
  }
}

void checkSigmoidRand() {
  SigmoidLayer s = SigmoidLayer(16, 10);
  for(int i = 0; i < s.getWeights().size(); i++) {
    for(int j = 0; j < s.getWeights()[i].size(); j++) {
      cout << s.getWeights()[i][j]; 
    }
    printf("\n");
  }

  for(int i = 0; i < s.getBiases().size(); i++) {
    cout << "Sigmoid Rand=" << s.getBiases()[i][0] << "\n";
  }
}

int main()
{
  // checkSigmoidRand();
  // checkMain();
  checkSigmoidSafe();
  // checkSigmoidVeryEasy();
  return 0;
}

