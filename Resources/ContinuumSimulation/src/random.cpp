// random numbers

#include "header.hpp"
#include "random.hpp"

using namespace std;

// truly random device to generate seed
random_device rd;
// pseudo random generator
// shlow
mt19937 gen(rd());
// super fasht
//ranlux24 gen(rd());
// uniform distribution on (0, 1)
uniform_real_distribution<> dist01(0, 1);
// uniform distribution on (0, 1)
uniform_real_distribution<> dist11(-1, 1);

// return random real, uniform distribution
double random_real(double min, double max)
{
  return uniform_real_distribution<>(min, max)(gen);
}

// return random real, gaussian distributed
double random_normal(double sigma)
{
  return normal_distribution<>(0., sigma)(gen);
}

// return geometric dist numbers, prob is p
unsigned random_geometric(double p)
{
  return geometric_distribution<>(p)(gen);
}

unsigned randu()
{
  return gen();
}

void set_seed(unsigned s)
{
  gen.seed(s);
}
