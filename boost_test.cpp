#include <iostream>
#include <iterator>
#include <algorithm>
#include <math.h>

using namespace std;

int main()
{
  double res = 1/(1 + exp(-.005));
  double res2 = res + .001;
  cout << res2 << "\n";
}
