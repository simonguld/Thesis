#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <typeinfo>
#include "serialization.hpp"

using namespace std;

struct A
{
  double value1 = 0.1;
  string value2 = "chose";

  template<class Archive>
  void serialize(Archive& ar)
  {
    ar & auto_name(value1) & auto_name(value2);
  }
};

REGISTER_TYPE_NAME(A);

int main(int argc, char ** argv)
{
  ofstream file("test.json");
  oarchive ar(file, "test", 1);

  // fundamental types
  double a = 0.2;
  unsigned b = 3;
  vector<string> c = { "hello", "goodbye", "cheers mate!" };
  vector<vector<int>> d = { {1,2,3}, {4,5,6}, {7,8,9}, {10,11,12} }; 
  vector<array<double, 3>> e = { {1,2,3}, {4,5,6}, {7,8,9}, {10,11,12} }; 

  ar & auto_name(a)
     & auto_name(b)
     & auto_name(c)
     & auto_name(d)
     & auto_name(e);

  // user defined types
  A x;
  array<A, 3> y;
  array<array<A, 3>, 3> z;

  ar & auto_name(x)
     & auto_name(y)
     & auto_name(z);

  return 0;
}
