Framework for generating particles with specified properties, and using them in LS-DEM simulation
Each simulation is an instance of the RVE (Representative Volume Element) class. 
Each RVE instance has attributes and functions. With this methods, multiple simulations can be created,
stored, analyzed and compared autonomously. 

RVE attributes are listed in rve.py. An rve class only needs the directory path to hold relevant files specified at creation.
See example.py for a simple example of rve class creation. 
The RVE class contains methods to create particles given certain propreties, set up the positions of the particles for pluviation,
run LS-DEM, then analyze packing fraction and coordination number of the simulation. 
Details and explanations of each function are given in rve.py.  
 
