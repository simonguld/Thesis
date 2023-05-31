
#Robert Buarque de Macedo
#2020

import helper
import plotConfiguration
import analyzeSim
import computePackingFraction
from rve import rve
import itertools

import os #python libraries 
import numpy as np
import sys
from grain import Grain
from numpy import linalg as LA
import matplotlib.pyplot as plt
import numpy.ma as ma  
import pickle
import re
import natsort
from shutil import copyfile

dt = 0.000005
#test RCP limit for disks 

mainDirName = "rcpTest/data/" #Where simulation-specific files go 
morphDirName = "rcpTest/morphs/" #where shapes go 
Rve = rve(morphDirName,mainDirName) #initialize RVE instance with default variables 
if os.path.exists(Rve.simDir + "scriptrun.sh"): os.remove(Rve.simDir + "scriptrun.sh") #remove old scriptRun file

#Parameters for cloning 
Rve.nGrains = 900
Rve.nShapes = 1 #just circles! 
Rve.nTot = 10000 #total timesteps 
Rve.nOut = 1000  #how often data is output from LSDEM 
Rve.shearHistFileIn = "None" #shear histories input file name. 'None' if there's none 
Rve.shearHistFileOut = "shearHistories.dat" #shear histories output file name
Rve.nShapes = 1 
Rve.showVals() #show current attributes 
Rve.createParticles() #create particles given default roundness/circularity distribution moments
Rve.makeInitPositions() #set up initial positions/velocities 
#Rve.executeDEM(pluviate=0,run=1) #run simulation. If run == 1, runs locally. If run == 0, creates a scriptrun.sh file that, if executes, submits job on supercomputer  
Rve.getDataFromFiles()
#Rve.plot(Rve.rightWallPos,Rve.rightWallPos,mainDirName + "figs/",mainDirName + "vid.mp4",makePics = 1, makeVideos = 1) #plot simulation
plotDir = "rcpTest/"
#(D,PF) = Rve.getPackingFraction(plotDir,plotDir + "packingFraction","Equilibrium Density",kostas= True) #get packing fraction, return radius (D) and packing fraction (PF)

#shake it like a polaroid camera 

newPositions = Rve.positions[-1,:,:] #get final positions from previous run, make these start positions in this run
np.savetxt(fname= mainDirName + "newPositions.dat", X = newPositions, fmt = "%.16f") 
newVelocities = Rve.velocities[-1,:,:]  #save with velocities! 
np.savetxt(fname = mainDirName + "newVelocities.dat", X = newVelocities, fmt = "%.16f")
Rve.startPosFile = "newPositions.dat" #set new input files 
Rve.startVelFile = "newVelocities.dat"			 
Rve.posFile = "positionsOut2.dat" #and output files! 
Rve.velFile = "velocitiesOut2.dat"	
Rve.shearHistFileIn = "shearHistories.dat" #shear histories input file name
Rve.shearHistFileOut = "shearHistories2.dat" #shear histories output file name


Rve.A = 0.04 #amplitude of shaking 
Rve.T = (1./100)*(1./dt) #time period of shaking! 
Rve.nOut = 10000
Rve.nTot = 1000000

#Rve.showVals() #print the new parameters to check they're right
Rve.executeDEM(pluviate=0,run=0) #run it! 
plotDir = mainDirName
Rve.getDataFromFiles()
Rve.plot(Rve.rightWallPos,Rve.rightWallPos,mainDirName + "figs/",mainDirName + "vid.mp4",makePics =0, makeVideos = 0) #plot simulation
(D,PF,PFs) = Rve.getPackingFraction(plotDir,plotDir + "packingFraction","Equilibrium Density",kostas= True, allSteps = True) #get packing fraction, return radius (D) and packing fraction (PF)
np.savetxt(fname=mainDirName+"PFs",X = PFs, fmt="%f")