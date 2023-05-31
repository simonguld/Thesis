#example showing usage of rve 
import os
os.chdir("Thesis\SiavashCode")
import helper
import plotConfiguration
import analyzeSim
import computePackingFraction
from rve import rve

morphDirName = "exampleDir/" #directory where morph files are storied 
mainDirName = "exampleDir/" #directory where all relevant information to specific simulation is stored 

Rve = rve(morphDirName,mainDirName) #create instance of rve class for this simulation 

#Rve.nGrains = 900  #number of grains in the experiment 
#Rve.simDir = "LSDEM2D/" #directory where simulation will take place 
#Rve.nTot = 1000000 #total timesteps 
#Rve.nOut = 10000  #how often data is output from LSDEM 
#Rve.shearHistFileIn = "None" #shear histories input file name. 'None' if there's none 
#Rve.shearHistFileOut = "shearHistories.dat" #shear histories output file name

Rve.nShapes = 5 #number of different unique morphs 
Rve.aspectRatio = 0.2
Rve.Mu_roundness = 0.2
Rve.Mu_circularity = 0.88

#self.aspectRatio = 1.0 #AR = max axis/min axis of starting ellipse
#self.Mu_roundness = 1.0  #Average of roundness targets
#self.Std_roundness = 0.01 #standard deviation of roundness targets
#self.Mu_circularity = 1.0 #same, but for circularity
#self.Std_circularity = 0.01 
               
Rve.showVals() #show current attributes 
Rve.createParticles() #create particles given default roundness/circularity distribution moments

#Rve.makeInitPositions() #set up initial positions/velocities 
#Rve.executeDEM(pluviate=0,run=1) #run simulation. If run == 1, runs locally. If run == 0, creates a scriptrun.sh file that, if executes, submits job on supercomputer  
#Rve.saveObj("rveTest") #save this rve in directory rveTest

#Rve = helper.loadObj("rveTest.pkl") #load this rve 
#Rve.getDataFromFiles() #extract data from simulation 
#Rve.plot(Rve.rightWallPos,Rve.rightWallPos,morphDirName + "figs/",morphDirName + "vid.mp4",makePics = 1, makeVideos = 1) #plot simulation 
