CC = mpicxx 
CC_INCLUDE = -I /usr/local/Cellar/eigen/3.3.9/include/eigen3/Eigen/ 
TARGETS = Main2


all: $(TARGETS)

Main2: maintest.cpp $(wildcard *.h)
	$(CC) $< -Wall -w -O3 -DEIGEN_NO_DEBUG -o $@ -O3 -std=c++11 $(CC_INCLUDE) -fopenmp -msse2 -pedantic

clean:
	rm -f $(TARGETS)
	
again: clean $(TARGETS)

