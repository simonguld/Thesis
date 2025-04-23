
#Robert Buarque de Macedo
#2020

import helper
import plotConfiguration
import analyzeSim
import computePackingFraction
from rve import rve

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


#main function. Give list of aspect ratios, and number of trials (attempts)
#runs a simulation for each aspect ratio/attemp pair. File handling done automatically
#A directory is made to hold data for each aspect ratio 
#plots in directory "plots"
#user parameters are lines 30-60

if __name__ == "__main__":
    for i, arg in enumerate(sys.argv):
        if (i == 1):
                attempts  = int(arg)
                #val = float(arg)
                #aspectVals += [val]
        if (i == 2):
                aspectVals = [float(arg)]

roundnessVals = [1.0]
circularityVals = [1.0]
attempts = 5
aspectVals = np.arange(2.0,6.0,1.0)
makeRVEs = 1
postProcess = 1

PFs = np.ones((len(aspectVals),attempts))*-1.0 #measured packing fractions, -1.0 if no value due to simulation failing 
CNs = np.ones((len(aspectVals),attempts))*-1.0
runList = [] #list of commands printed to sriptRun
simDir = "LSDEM2D/" #simulation directory
rveDir = "savedRVEs/" #where RVE's are saved
if os.path.exists(simDir + "scriptrun.sh"): os.remove(simDir + "scriptrun.sh") #remove old scriptRun file

#Main function for running full cloning pipeline 
count = 0 
if (makeRVEs):
	os.system("rm savedRVEs/*")
	for j in range(len(aspectVals)):
		for attempt in range(attempts):
			aspectRatio =aspectVals[j] 

			morphDirName = "mu05/morphs" + "A_" + "%.1f"%aspectRatio + "/" + "trial_" + str(attempt) + "/" #Where morph files/ point files go
			Rve = rve(morphDirName) #initialize RVE instance with default variables 
			
			#Parameters for cloning 
			Rve.aspectRatio = aspectRatio #AR = max axis/min axis of starting ellipse
			Rve.Mu_roundness = roundnessVals[0] #Average of roundness targets
			Rve.Std_roundness = 0.01 #standard deviation of roundness targets
			Rve.Mu_circularity = circularityVals[0] #same, but for circularity
			Rve.Std_circularity = 0.01    		
	 
			Rve.trial = attempt
			Rve.nGrains = 900
			Rve.nShapes = 1 

			#Parameters for simulation (input)
			Rve.startPosFile = "positions.dat"
			Rve.startVelFile =  "velocities.dat"
			Rve.morphFile = "morphIDs.dat" #list of morph IDs in expt. file name
			Rve.paramFile      = "param_file.txt" #parameter file name 

			# simulation files(output)
			Rve.posFile = "positionsOut.dat" 
			Rve.velFile = "velocitiesOut.dat"
			Rve.simDir = simDir #simulation directory

			#create simulation domain ICs
			Rve.numParticlesRow = 9. #number of particles in each row
			Rve.numParticlesCol = Rve.nGrains/Rve.numParticlesRow
			Rve.grainRadiusHorz = 50. #(1/2)*distance between particles horizontally
			Rve.grainRadiusVert = 40 # '' vertically		
			Rve.startSpot = [2*Rve.grainRadiusHorz,2*Rve.grainRadiusVert] #bottom left of particle grid (0,0)

			#simulation parameters
			Rve.nTot =  1 #total timesteps 
			Rve.nOut =  50000 #every nOut timestep data recorded 
			Rve.stopStep = 1900000 #when to stop shaking 
			Rve.A = 0.0 #Amplitude of shaking 
			Rve.T = 0.0 #Time period of shaking 
			Rve.topWallPos = 5000  #position of top wall
			Rve.rightWallPos = 600 #position of right wall 
			Rve.Force = 0.0 #force applied on top wall 
			Rve.dropDist = 50 #just ignore this 
			Rve.randomMult = 200 #random number which is maximum starting velocity in a direction 

			####################################################################################### end parameters
#			Rve.createParticles() #create particles
			Rve.makeInitPositions() 		#Now set up pluviation simulation
			runList = Rve.executeDEM(pluviate=0,run=0) #run equilibration

			Rve.saveObj(rveDir + 'RVE_' + str(count)) #save simulation object 	
			count += 1	

##############################################################################################
#Post Processing 

#for a in []:
if (postProcess):
	rveFileList = helper.getSortedFileList(rveDir)#get list of saved rve's
	for rveFile in rveFileList:
		#Visualization parameters 
		Rve = helper.loadObj(rveDir + rveFile)
		visDir = "mu05_plots/" #directory for visualizing this RVE 
		plotDir        = visDir + "Asp%0.2f/"%Rve.aspectRatio#folder to hold plots 
		vidName      = plotDir + "equil_Asp%0.2f_Num%d.mp4"%(Rve.aspectRatio,Rve.trial) #name of video of simulation

		figDir = plotDir + "figs/" #where the figures will be put, before being made into a video 
		PFfileName = plotDir + "equilibriumDensity_num_kostas%d"%Rve.trial #file name for packing fraction vs. radius plot 
		CNfileName = plotDir + "equilibriumCN"  #file name for packing fraction vs. radius plot 

		lX = Rve.rightWallPos #max X coordinate in plot 
		lY = Rve.rightWallPos #max Y coordinate in plot 

		try: Rve.getDataFromFiles() #get simulation output (including grain data)
		except: continue #if missing data for this simulation
		Rve.getDataFromFiles()
		Rve.plot(lX,lY,figDir,vidName,makePics = 1,makeVideos = 1)

		(D,PF) = Rve.getPackingFraction(plotDir,PFfileName,"Equilibrium Density",kostas= True) #get packing fraction, return radius (D) and packing fraction (PF)
		Index = int( (Rve.aspectRatio - aspectVals[0])/(aspectVals[1] - aspectVals[0]) )
		PFs[ Index ,Rve.trial] = PF #save it
		Z = Rve.getAvgCoordinationNumber("Coordination Number") #kostas method
		CNs[ Index ,Rve.trial ] = Z  #save it

		#Save global data - use in resultsPlot.py 
		np.savetxt(fname = visDir + "packingFractions.txt",X = PFs, fmt = "%.4f")
		np.savetxt(fname = visDir + "aspectRatios.txt",X = aspectVals, fmt = "%.4f")
		np.savetxt(fname = visDir + "coordNums.txt",X = CNs, fmt = "%.4f")
