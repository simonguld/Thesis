#generate/analyze simple circles in square domain 

import helper
import plotConfiguration
import analyzeSim
import computePackingFraction
from rve import rve
import numpy as np



def run():
    trials = 2
    for trial in np.arange(trials):
        morphDirName = "MLstuff/trial_%d/"%trial #directory where morph files are stored 
        mainDirName = "MLstuff/trial_%d/"%trial #directory where all relevant information to specific simulation is stored 

        Rve = rve(morphDirName,mainDirName) #create instance of rve class for this simulation 
        Rve.nShapes = 1 #number of different unique morphs 
        Rve.nGrains = 100  #number of grains in the experiment 
        Rve.simDir = "LSDEM2D/" #directory where simulation will take place 
        Rve.nTot = 2000000 #total timesteps 
        Rve.nOut = 50000  #how often data is output from LSDEM 
        Rve.shearHistFileIn = "None" #shear histories input file name. 'None' if there's none 
        Rve.shearHistFileOut = "shearHistories.dat" #shear histories output file name
        Rve.nShapes = 1 

        Rve.numParticlesRow = 10 #pluviation parameters for putting particles in a grid
        Rve.numParticlesCol = Rve.nGrains/Rve.numParticlesRow 
        Rve.grainRadiusVert = 25 #distances between particles in grid (vertical )
        Rve.grainRadiusHorz = 25 #'' (horizontal)
        Rve.startSpot = [Rve.grainRadiusHorz,Rve.grainRadiusVert] #bottom left of grid
        Rve.rightWallPos = 275

        Rve.showVals() #show current attributes 
        Rve.createParticles() #create particles given default roundness/circularity distribution moments
        Rve.makeInitPositions() #set up initial positions/velocities 
        Rve.executeDEM(pluviate=0,run=1) #run simulation. If run == 1, runs locally. If run == 0, creates a scriptrun.sh file that, if executes, submits job on supercomputer  
        Rve.saveObj("MLstuff/rvesML/rveML_%d"%trial) #save this rve in directory rveTest

def squish():
    rveDir = "MLstuff/rvesML/"
    rveFileList = helper.getSortedFileList(rveDir)
    print (rveFileList)
    for rveFile in (rveFileList):
        print (rveFile)
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

        maxVert = np.max(newPositions[:,1])
        Rve.nOut = 1000
        Rve.nTot = 10000
        Rve.Force = 0
        Rve.topWallPos = maxVert + 10
        
        #Rve.showVals() #print the new parameters to check they're right
        Rve.executeDEM(pluviate=0,run=1) #run it!
        Rve.getDataFromFiles() #extract data from simulation 
        outputDir = Rve.mainDir 
        Rve.plot(Rve.rightWallPos,Rve.rightWallPos,outputDir + "figs/",outputDir + "vid.mp4",makePics = 1, makeVideos = 1) #plot simulation 
 
 



def plot():
    rveDir = "MLstuff/rvesML/"
    rveFileList = helper.getSortedFileList(rveDir)#get list of saved rve's
    print (rveFileList)
    for rveFile in rveFileList:
        Rve = helper.loadObj(rveDir + rveFile) #load this rve 
        Rve.getDataFromFiles() #extract data from simulation 
        outputDir = Rve.mainDir 
        Rve.plot(Rve.rightWallPos,Rve.rightWallPos,outputDir + "figs/",outputDir + "vid.mp4",makePics = 1, makeVideos = 1) #plot simulation 

#run()
squish()
#plot()