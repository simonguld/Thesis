#ifndef RANDOM_HPP_
#define RANDOM_HPP_

/** Return random real between min and max, uniform distribution */
double random_real(double min=0., double max=1.);

/** Return random real, normal distribution with variance sigma and zero mean */
double random_normal(double sigma=1.);

/** Return geometric distributed integers */
unsigned random_geometric(double p);

/** Return random iunsigned int */
unsigned randu();

/** Set seed of the random number generator */
void set_seed(unsigned);

#endif//RANDOM_HPP_
