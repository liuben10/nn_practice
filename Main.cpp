#include <boost/multiprecision/cpp_dec_float.hpp>

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
using namespace boost::multiprecision;


typedef unsigned char uchar;

class Wrapper {
private:
  int expected;
  vector<number<cpp_dec_float<300> >   > values;
public:
  int getExpected() {
    return this->expected;
  }

  vector<number<cpp_dec_float<300> >   > getValues() {
    return this->values;
  }

  Wrapper(int expected, vector<number<cpp_dec_float<300> >   > values) {
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
  vector<number<cpp_dec_float<300> >   > * values = new vector<number<cpp_dec_float<300> >   >(784, 0);

  Wrapper result(expected, *values);

  // 28 rows
  for(int i = 0; i < 28; i++) {
    // 28 cols
    for(int j = 0; j < 28; j++) {
      number<cpp_dec_float<300> >    cur = dataset[ind][i*28 + j];
      if(dataset[ind][i*28+j] > 80) {
	cout << 1;
	result.getValues().push_back(cur);
      } else {
	cout << 0;
	result.getValues().push_back(cur);
      }
    }
    cout << "" << endl;
  }

  return result;
}


void checkSigmoid() {
  SigmoidLayer * sl = new SigmoidLayer(10, 4);
  vector<vector<number<cpp_dec_float<300> >   > > weights;

  number<cpp_dec_float<300> >    firstRow[10] = {0, 10, 0, 10, 0, 10, 0, 10, 0, 10};
  vector<number<cpp_dec_float<300> >   > firstRowVec(firstRow, firstRow + (sizeof(firstRow) / sizeof(firstRow[0])) );
  weights.push_back(firstRowVec);

  number<cpp_dec_float<300> >    secondRow[10] = {0, 0, 10, 10, 0, 0, 10, 10, 0, 0};
  vector<number<cpp_dec_float<300> >   > secondRowVec(secondRow, secondRow + (sizeof(secondRow) / sizeof(secondRow[0])) );
  weights.push_back(secondRowVec);

  number<cpp_dec_float<300> >    thirdRow[10] = {0, 0, 0, 0, 10, 10, 10, 10, 0, 0};
  vector<number<cpp_dec_float<300> >   > thirdRowVec(thirdRow, thirdRow + (sizeof(thirdRow) / sizeof(thirdRow[0])) );
  weights.push_back(thirdRowVec);

  number<cpp_dec_float<300> >    fourthRow[10] = {0, 0, 0, 0, 0, 0, 0, 0, 10, 10};
  vector<number<cpp_dec_float<300> >   > fourthRowVec(fourthRow, fourthRow + (sizeof(fourthRow) / sizeof(fourthRow[0])) );
  weights.push_back(fourthRowVec);

  sl->setWeights(weights);

  number<cpp_dec_float<300> >    biases[4] = {-5, -5, -5, -5};
  vector<number<cpp_dec_float<300> >   > biasVec(biases, biases + (sizeof(biases) / sizeof(biases[0])));

  sl->setBiases(biasVec);

  number<cpp_dec_float<300> >    inputs[10] = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
  vector<number<cpp_dec_float<300> >   > inputsVec(inputs, inputs + (sizeof(inputs) / sizeof(inputs[0])));

  vector<number<cpp_dec_float<300> >   > activations = sl->dotAndBiased(inputsVec);
}

void checkSigmoidSafe() {
  int numLayers = 3;
  int neurons[3] = {2, 2, 1};
  NeuralNetwork nn = NeuralNetwork(neurons, numLayers);
  for(int i = 0; i < 100; i++) {
    cout << "###########################################" << "\n";
    cout << "###########################################" << "\n";
    cout << "###########################################" << "\n";
    cout << "###########################################" << "\n";
    cout << "###########################################" << "\n";
    cout << "ITERATION:  " << i << "\n";
    cout << "###########################################" << "\n";
    cout << "###########################################" << "\n";
    cout << "###########################################" << "\n";
    cout << "###########################################" << "\n";
    cout << "###########################################" << "\n";

    
    vector<number<cpp_dec_float<300> >   > input = vector<number<cpp_dec_float<300> >   >();
    input.push_back(0);
    input.push_back(1);

    vector<number<cpp_dec_float<300> >   > output = nn.feedForward(input);
    for(int i = 0; i < output.size(); i++) {
      cout << "Checking Sigmoid Output=" << output[i] << "\n";
    }

    vector<number<cpp_dec_float<300> >   > expected;
    expected.push_back(1);
    WeightsAndBiasUpdates wb = nn.backPropagate(input, expected);
    nn.applyUpdates(&wb);
    cout << "\n" << wb.toString() << "\n\n Post Update \n\n";
    nn.printNetwork();
  }
  
  vector<number<cpp_dec_float<300> >   > input = vector<number<cpp_dec_float<300> >   >();
  input.push_back(0);
  input.push_back(1);
  nn.feedForward(input);
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

  for(int k = 0; k < 100; k++) {
    Wrapper input = showRandomCharacterInBinary(dataset, labels, number_of_images);

    //        vector<number<cpp_dec_float<300> >   > output = network.feedForward(input->getValues());

    vector<number<cpp_dec_float<300> >   > expectedBin = Coster::toBinary(input.getExpected());

    WeightsAndBiasUpdates updates = network.backPropagate(input.getValues(), expectedBin);

    std::cout << "\n\n==Updates==\n\n" <<  updates.toString() << "\n\n\n\n";

    network.applyUpdates(&updates);

    //        printf("\n\n == k: %d == \n\n", k);
  }

  Wrapper input = showRandomCharacterInBinary(dataset, labels, number_of_images);

  

  vector<number<cpp_dec_float<300> >   > output = network.feedForward(input.getValues());

  for(int i = 0; i < output.size(); i++) {
    cout << output[i] << "\n";
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
    cout << "Sigmoid Rand=" << s.getBiases()[i] << "\n";
  }
}

int main()
{
  // checkSigmoidRand();
  //checkMain();
  checkSigmoidSafe();
}

