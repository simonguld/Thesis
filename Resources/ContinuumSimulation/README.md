# README #

Example run command

./mass inputFile.dat -o outputFolder

### WARNING ###

A major part of this fork of the code has been changed to ensure Navier Stokes equation is actually solved properly. This means that most of the old source files will not work correctly with this code. 

Changelog:

* Changed the weights in the LB file

* Changed the structure of the code to compute the force first

* Changed the velocity updation algorithm to follow Guo et. al.

* Changed the source term to follow Guo et.al. in the collision step


Changes implemented in:

* TwoFluid.cpp

* nematic.cpp

* TwoCompMD.cpp

* lyotropic.cpp

Be careful before using any other codes in this package. Good way to check that the code is correct is to look for a forcing term in the velocity update (UpdateFluidQuantities), and a source term in the distribution function collision term (UpdateFluidFields). See book by Kruger on the Lattice Boltzmann method, Chapter 6, for further details.

### Many Active Systems Simulations ###

To run, use the folowing example command:
./mass lyotropic.dat -fco out -t2

This code has been used for the following paper:
Biphasic, Lyotropic, Active Nematics. Blow, Thampi, Yeomans (2014). [https://arxiv.org/abs/1407.7493]
and others.

This version also fixes a bug where the multi-threading was not being used, although being implemented in a parallelized way.

### Compiling ###

The code relies on the boost::program_options library. Once it is installed, a
simple `make` in the main directory should do it.

### Running ###

The code is run from the command line and a runcard must always be given as the
first argument:

`./mass runcard.dat`

A runcard is a simple file providing the parameters for the run. Example
runcards can be found in the `example/` directory. Every option can also be
given to the program using the command line as
`./mass runcard.dat --option=arg`. A complete list of available options can be
obtained by typing `./mass -h`.

By default the program writes output files in the current directory. This can be
changed using `--output=dir/` or `-o dir/`, where `dir/` is the target
directory. The program also supports compressed output with the option flag
`--compression` or `-c`. When compression is on, the output name does not
denote the target directory but rather the file name of the compressed archive.
Note that with the current implementation, compression is not recommended for
long runs as the time to add a file to the archive grows with the size of the
archive.

Type `./mass -h` or `./mass -m model-name -h` for a list of available options.

### Plotting ###

Plotting is supported with both matlab and python (using matplotlib). Take a
look at the example in plot/.

### Contributing ###

For simplicity the master branch is not protected such that everybody can directly
push to it. This means that you have the responsability not to break the code!
In particular:

* Run `make clean && make` before pushing to ensure that the code compiles.

* Review your code with other people if you have any doubt.

* If you are making big changes (like implementing a new model) think about branching.


Other remarks about contributions:

* We do not have a strict coding style but try to respect what you see in the
rest of the project.

* Good code is easy to read and understandable, not especially 'smart'... In
particular give understandable names to variables and functions.

* Try to respect the hierarchy of the files, etc.

* Commenting does not make the code slower!

* Get some inspiration from: https://www.doc.ic.ac.uk/~susan/475/unmain.html


#### Adding a new model ####

Adding a new model should be simple and can be done with the following steps:

* Add model declaration and implementation in `src/models/` as `your_model_name.hpp`
and `your_model_name.cpp`. You can either copy an existing model or write these
files from scratch. Take a look at model `minimal` for a simple example of a
minimal implementation.

* In the file `src/declare_models.cpp` add the corresponding
`#include<models/your_model_name.hpp>` and declare your model using the
`declare_model<ClassName>` template function.

* In the `Makefile` add the model
to model list. Finally type `make clean && make` in the main directory to check
that everything is fine.



#### Example Runcard ####

Copy and paste the following to an empty file named 'inputFile.dat', and run the given command.

    
\# datafile for nematic code
    

\# model and initial configuration

model    = nematic

nsteps   = 1000

ninfo    = 100

LX       = 200

LY       = 200

bc	 = 0


\# model parameters... 

Gamma    = 0.1

xi 	 = 0.0

tau      = 2

rho      = 40

friction = 0.0

CC	 = 0.10

LL       = 0.01

zeta     = 0.03

backflow_on = true


\# initial configuration options... 

angle    = 0

noise    = 0

n_preinit= 0

npc	 = 0

