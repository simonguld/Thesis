
INCS = -I src/ -I build/

FLAGS = -Wall -Wextra -Wpedantic -std=c++1y -O3 -fopenmp -g

LIBS = # -lboost_program_options

LIBPATH = 

COMP = g++

HEADERS = src/tools.hpp

BASE = \
       build/geometry.o \
       build/options.o \
       build/serialization.o \
       build/random.o \
       build/write.o \
       build/models.o

MODELS = \
         build/models/minimal.o \
         build/models/lyotropic.o \
         build/models/nematic.o \
	 build/models/polarlyotropic.o \
	 build/models/drypolarlyotropic.o  
#

OUT_DIR = build/ build/models/

MKDIR_P = mkdir -p

.PHONY: all clean cleanall directories debug hydra

all: LIBS += -lboost_program_options

all: directories mass

debug: FLAGS += -DDEBUG -ggdb3

debug: all

hydra: HYDRALIBS = /usr/local/shared/boost/1.64.0-gcc5.4.0/lib/libboost_program_options.a

hydra: LIBPATH += -L/usr/local/boost/stage/lib/

hydra: directories mass

directories: ${OUT_DIR}

${OUT_DIR}:
	${MKDIR_P} ${OUT_DIR}

src/header.hpp.gch: src/header.hpp
	$(COMP) -c $< -o $@ $(INCS) $(FLAGS) $(LIBPATH) $(LIBS)

build/%.o: src/%.cpp src/%.hpp src/header.hpp.gch
	$(COMP) -c $< -o $@ $(INCS) $(FLAGS) $(LIBPATH) $(LIBS)

mass: src/main.cpp $(BASE) $(MODELS) $(HEADERS)
	$(COMP) $^ -o $@ $(INCS) $(FLAGS) $(LIBPATH) $(LIBS) $(HYDRALIBS)

test_serialization: src/test_serialization.cpp build/serialization.o
	$(COMP) $^ -o $@ $(INCS) $(FLAGS) $(LIBPATH) $(LIBS)

clean:
	rm -f build/*.o build/models/*.o

cleanall: clean
	rm -rf build/
	rm -rf src/header.hpp.gch
