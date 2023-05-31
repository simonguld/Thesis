
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

import numpy as np

def ellipseCost(AR):
	a,b = AR,1 
	Theta = np.linspace(0,2*math.pi,16, endpoint = False)		
	r = (Min_prin*Max_prin)/np.sqrt( (Min_prin*np.cos(Theta))**2 + (Max_prin*np.sin(Theta))**2 )
	ar = getPA(lset)	
	circularityCurr = (2*np.sqrt(np.pi*area)/(perim))**2
	return (circularityCurr - circularity)**2 

#main function. Give list of aspect ratios, number of trials (attempts), and roundnesses 
#runs a simulation for each roundness/attemp pair 

roundnessVals = [0.8,0.85,0.9,0.95] #list of roundnesses 
circularityVals = [1.0] #list of circularities 
aspectVals = [1.0] #list of aspect ratios 
attempts = 5 #number of attempts per roundness value  

makeRVEs = 1 #if section 1 is run
shake = 0 #if section 2 is run 
postProcess = 0 #if section 3 is run 

PFs = np.ones((len(roundnessVals),attempts))*-1.0 #measured packing fractions, -1.0 if no value due to simulation failing 
CNs = np.ones((len(roundnessVals),attempts))*-1.0
simDir = "LSDEM2D/" #simulation directory
rveDir = "savedRVEs/" #where RVE's are saved
if os.path.exists(simDir + "scriptrun.sh"): os.remove(simDir + "scriptrun.sh") #remove old scriptRun file

#SECTION 1 : Main function for running full cloning pipeline. Here, iterate through roundness and for each attempt generate particles. 
#Don't have to re-generate particles every time as particles are stored in a separate directory to file holding simulation data. 

if (makeRVEs):
	for j in range(len(roundnessVals)):
		for attempt in range(attempts):
			aspectRatio =aspectVals[0] 
			roundness = roundnessVals[j]
			mainDirName = "roundnessTest/morphs" + "R_" + "%.2f"%roundness + "/" + "trial_" + str(attempt) + "/" #Where simulation-specific files go 
			morphDirName = "roundnessTest/morphs" + "R_" + "%.2f"%roundness + "/shapes/" #where shapes go 
			Rve = rve(morphDirName,mainDirName) #initialize RVE instance with default variables 
		
			#Parameters for cloning 
			Rve.aspectRatio = aspectRatio # max axis/min axis of starting ellipse
			Rve.Mu_roundness = roundness #Average of roundness targets
			Rve.Std_roundness = 0.001 #standard deviation of roundness targets
			Rve.Mu_circularity = circularityVals[0] #same, but for circularity
			Rve.Std_circularity = 0.001
			Rve.trial = attempt
			Rve.nGrains = 900
			Rve.nShapes = 900

			#Parameters for simulation (input)
			Rve.startPosFile = "positions.dat"
			Rve.startVelFile =  "velocities.dat"
			Rve.morphFile = "morphIDs.dat" #list of morph IDs in expt. file name
			Rve.paramFile      = "param_file.txt" #parameter file name 
			Rve.shearHistFileIn = "None" #shear histories input file name. 'None' if there's none 
			Rve.shearHistFileOut = "shearHistories.dat" #shear histories output file name

			# simulation files(output)
			Rve.posFile = "positionsOut.dat" 
			Rve.velFile = "velocitiesOut.dat"
			Rve.simDir = simDir #simulation directory

			#create simulation domain ICs
			Rve.numParticlesRow = 15. #number of particles in each row
			Rve.grainRadiusHorz = 100./3 #(1/2)*distance between particles horizontally
			Rve.grainRadiusVert = 40 # '' vertically		
			Rve.startSpot = [2*Rve.grainRadiusHorz,2*Rve.grainRadiusVert] #bottom left of particle grid (0,0)

			#simulation parameters
			Rve.nTot =   1000000#total timesteps 
			Rve.nOut =  10000 #every nOut timestep data recorded 
			Rve.stopStep = 1900000 #when to stop shaking 
			Rve.A = 0.0 #Amplitude of shaking 
			Rve.T = 0.0 #Time period of shaking 
			Rve.topWallPos = 2520  #position of top wall
			Rve.rightWallPos = 600 #position of right wall 
			Rve.Force = 0.0 #force applied on top wall 
			Rve.dropDist = 50 #just ignore this 
			Rve.randomMult = 200 #random number which is maximum starting velocity in a direction 

			####################################################################################### end parameters
			#Rve.createParticles() #create particles. comment out if particles already created. 
			Rve.makeInitPositions() 		#Now set up pluviation simulation
			Rve.executeDEM(pluviate=0,run=1) #run equilibration

			#Rve.saveObj(rveDir + 'RVE_R' + "%.2f"%roundness + "_n" + str(attempt)) #save simulation object 	

#######################################################################################################3

#SECTION 2: apply shaking to simulation. plotting is handled in this section, so don't run section 3 

if (shake):				
	rveFileList =['RVE_R0.95_n0.pkl', 'RVE_R0.95_n1.pkl', 'RVE_R0.95_n2.pkl', 'RVE_R0.95_n3.pkl', 'RVE_R0.95_n4.pkl']#helper.getSortedFileList(rveDir)
	print (rveFileList)
	for rveFile in (rveFileList):
		Rve = helper.loadObj(rveDir + rveFile)
		Rve.getDataFromFiles()	
		mainDirName = Rve.mainDir

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


		Rve.A = 0.01 #amplitude of shaking 
		Rve.T = 1000 #time period of shaking! 
		Rve.nOut = 10000
		Rve.nTot = 1000000
		
		#Rve.showVals() #print the new parameters to check they're right
		#Rve.executeDEM(pluviate=0,run=0) #run it! 

		#simple post processing just for shaking 		
		Rve.getDataFromFiles() #collect data for visualization 
		visDir = "roundnessTestShakePlots/" #directory for visualizing this RVE 
		plotDir = visDir + "R%0.2f/"%Rve.Mu_roundness#folder to hold plots
		figDir = plotDir + "figs/" #where the figures will be put, before being made into a video
		vidName  = plotDir + "shake_R%0.2f_Num%d.mp4"%(Rve.Mu_roundness,Rve.trial)
		Rve.plot(Rve.rightWallPos,Rve.rightWallPos,figDir,vidName,1,1)
		(D,PF) = Rve.getPackingFraction(plotDir,plotDir + "packingFraction","Equilibrium Density",kostas= True) #get packing fraction, return radius (D) and packing fraction (PF)
		Index = int( (Rve.Mu_roundness - roundnessVals[0])/(roundnessVals[1] - roundnessVals[0]) )
		PFs[ Index ,Rve.trial] = PF #save it
		Z = Rve.getAvgCoordinationNumber("Coordination Number") #kostas method
		print ("Z = ", Z)
		CNs[ Index ,Rve.trial] = Z  #save it
		np.savetxt(fname = visDir + "packingFractions.txt",X = PFs, fmt = "%.4f")		#Save global data - use in resultsPlot.py 
		np.savetxt(fname = visDir + "aspectRatios.txt",X = aspectVals, fmt = "%.4f")
		np.savetxt(fname = visDir + "coordNums.txt",X = CNs, fmt = "%.4f")


##############################################################################################

#Post Processing. Plots, finds packing fraction and coordination number for all rve's stored in rveDir

if (postProcess):
	rveFileList = ['RVE_R0.85_n2.pkl']#, 'RVE_R0.85_n3.pkl', 'RVE_R0.85_n4.pkl', 'RVE_R0.90_n0.pkl', 'RVE_R0.90_n1.pkl', 'RVE_R0.90_n2.pkl', 'RVE_R0.90_n3.pkl', 'RVE_R0.90_n4.pkl', 'RVE_R0.95_n0.pkl', 'RVE_R0.95_n1.pkl', 'RVE_R0.95_n2.pkl', 'RVE_R0.95_n3.pkl', 'RVE_R0.95_n4.pkl']#helper.getSortedFileList(rveDir)#get list of saved rve's
	print (rveFileList)
	for rveFile in rveFileList:
		#Visualization parameters 
		Rve = helper.loadObj(rveDir + rveFile)
		visDir = "roundnessTestPlotsPreShake/" #directory for visualizing this RVE 
		plotDir        = visDir + "R%0.2f/"%Rve.Mu_roundness#folder to hold plots 
		vidName      = plotDir + "equil_R%0.2f_Num%d.mp4"%(Rve.Mu_roundness,Rve.trial) #name of video of simulation

		figDir = plotDir + "figs/" #where the figures will be put, before being made into a video 
		PFfileName = plotDir + "equilibriumDensity_num_kostas%d"%Rve.trial #file name for packing fraction vs. radius plot 
		CNfileName = plotDir + "equilibriumCN"  #file name for packing fraction vs. radius plot 

		lX = Rve.rightWallPos #max X coordinate in plot 
		lY = Rve.rightWallPos #max Y coordinate in plot 
		try: 
			Rve.getDataFromFiles() #get simulation output (including grain data)
			Rve.plot(lX,lY,figDir,vidName,makePics = 1,makeVideos = 1)
			Index = int( (Rve.Mu_roundness - roundnessVals[0])/(roundnessVals[1] - roundnessVals[0]) )
			(D,PF) = Rve.getPackingFraction(plotDir,PFfileName,"Equilibrium Density",kostas= True) #get packing fraction, return radius (D) and packing fraction (PF)
			PFs[ Index ,Rve.trial] = PF #save it
			Z = Rve.getAvgCoordinationNumber("Coordination Number") #kostas method
			print ("Z = ", Z)
			CNs[ Index ,Rve.trial] = Z  #save it
			np.savetxt(fname = visDir + "packingFractions.txt",X = PFs, fmt = "%.4f")		#Save global data - use in resultsPlot.py 
			np.savetxt(fname = visDir + "aspectRatios.txt",X = aspectVals, fmt = "%.4f")
			np.savetxt(fname = visDir + "coordNums.txt",X = CNs, fmt = "%.4f")
		except: 
			continue 



		

