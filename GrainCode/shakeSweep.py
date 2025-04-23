
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


import numpy as np

#main function. Give list of aspect ratios, number of trials (attempts), and roundnesses 
#runs a simulation for each roundness/attemp pair 

roundnessVals = [0.95] #list of roundnesses 
circularityVals = [1.0] #list of circularities 
aspectVals = [1.0] #list of aspect ratios 
trials = 5 #number of attempts per roundness value 
dt = 0.000005
As =  [1,2,5,10,16] #list of amplitudes to check 
Ts = 1./(np.array([0.1,0.3,0.8,1,5,15,25,35])) * (1/dt)#[1000,5000,10000]#[500,1000,1500,2000] #[500,1000,1500,2000] #list of time periods to check 

shake = 1 #if section 1 is run 
postProcess = 0 #if section 2 is run 

PFs = np.loadtxt("shakeSweep/plots/R_%.2f/packingFractions"%roundnessVals[0]).tolist() #each row: A,T,trial & PF of run
CNs = np.ones((( len(As), len(Ts), trials )))*-1.0
simDir = "LSDEM2D/" #simulation directory
rveDir = "shakeSweep/savedRVEs/" #where RVE's are saved
if os.path.exists(simDir + "scriptrun.sh"): os.remove(simDir + "scriptrun.sh") #remove old scriptRun file

if (shake):		
	for i, j,trial in itertools.product(np.arange(len(As)),np.arange(len(Ts)),np.arange(trials)): #iterate through amplitude/time period combinations 
		if (trial != 3): continue 
		#try:
		A,T = As[i],Ts[j]
		aspectRatio =aspectVals[0] 
		roundness = roundnessVals[0]
		mainDirName = "shakeSweep/R_%.2f/A_%f_T_%f/trial_%d/"%(roundness,A,T,trial) #Where simulation-specific files go 
		morphDirName = "shakeSweep/morphsR_%.2f/shapes/"%roundness #where shapes are 
		Rve = rve(morphDirName,mainDirName) #initialize RVE instance with default variables 
		#os.system( "cp -r shakeSweep/morphsR_%.2f/trial_0/* "%roundness + mainDirName) #start from end of previous pluviation
		#copy_tree("/groups/geomechanics/rob/shakeSweep/morphsR_%.2f/trial_0/"%roundness,mainDirName)
		
		Rve.startPosFile = "newPositions.dat" #set new input files 
		Rve.startVelFile = "newVelocities.dat"			 
		Rve.posFile = "positionsOut2.dat" #and output files! 
		Rve.velFile = "velocitiesOut2.dat"	
		Rve.shearHistFileIn = "shearHistories.dat" #shear histories input file name
		Rve.shearHistFileOut = "shearHistories2.dat" #shear histories output file name

		Rve.A = A #amplitude of shaking 
		Rve.T = T #time period of shaking! 
		Rve.aramp = 1./50000
		Rve.nOut = 10000
		Rve.nTot = 100000
		Rve.stopStep = Rve.nTot*0.9#when to stop shaking                        
				
		#Rve.showVals() #print the new parameters to check they're right
		Rve.executeDEM(pluviate=0,run=0) #run it! 

		#simple post processing just for shaking 
		print ("mainDirName",mainDirName,i,j)			
		Rve.getDataFromFiles() #collect data for visualization 
		if (Rve.nSteps <= 545): continue #skip unfinished simulations
		visDir = "shakeSweep/plots/R_%.2f/A_%f_F_%f_trial_%d/"%(roundness,A,1./(dt*T),trial)  #directory for visualizing this RVE 
		plotDir = visDir #folder to hold plots
		figDir = plotDir + "figs/" #where the figures will be put, before being made into a video
		vidName  = plotDir + "shake_R%0.2f_A%f_F%f.mp4"%(roundness,Rve.A,1./(Rve.T*dt) )
		print ("plotDir",plotDir)
		Rve.plot(Rve.rightWallPos,Rve.rightWallPos,figDir,vidName,0,0) #plot the simulation
		(D,PF,PFarray) = Rve.getPackingFraction(plotDir,plotDir + "packingFraction","Equilibrium Density",kostas= True, allSteps=True) #get packing fraction, return radius (D) and packing fraction (PF)
		np.savetxt(fname = plotDir + "allSteps",X = PFarray) #save packing fraction array, ordered by time station
		PFs = helper.addValue(PFs,A,(1./T)*(1/dt),PF,trial) 
		print ("added PF",A,(1./T)*(1/dt),PF)
		np.savetxt(fname = "shakeSweep/plots/R_%.2f/packingFractions"%roundness,X = PFs, fmt="%f")		#Save global data - use in resultsPlot.py 
	
		#np.savetxt(fname = "shakeSweep/plots/" + "coordNums.txt",X = CNs, fmt = "%.4f")
		#except Exception as e:
			#print (e) 
			#continue 
