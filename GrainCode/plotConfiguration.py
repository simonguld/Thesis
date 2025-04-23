# -*- coding: utf-8 -*-
"""

@author: konstantinos
@date: November, 2016
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from grain import Grain
import helper

def plotSim(Rve,figDir,vidName,lX,lY,makePics = 1,video = 1,forces = 0,justCentroids=0):
    #get objects
    morphDir = Rve.morphDir
    morphFile = Rve.morphFile
    nGrains            = Rve.nGrains
    posRot             = Rve.positions
    posRot0            = Rve.startPositions
    nSteps             = Rve.nSteps
    morphID = Rve.morph 
    print ("figDir",figDir)

    rho = 2.65 #g/pixel^3   

    if not os.path.exists(figDir):
        os.mkdir(figDir)

    # Read grain morphology data

    # Instantiate grains
    grains = Rve.grains

    if not (makePics):
        nSteps = 0

    print ("nSteps",nSteps)
    for step in range(0,nSteps,1):
        # Set up the figure
        fig, ax = plt.subplots(figsize = (5,5))
        ax.set_xlim(0,lX)
        ax.set_ylim(0,lY)
        ax.autoscale_view()
                
        # Update grain positions
        posRotStep = posRot[step]
        for n in range(nGrains):
            grains[n].updateXY(posRotStep[n])
        
        # Collect in patches
        if not justCentroids: 
            patches = []
            for n in range(nGrains):
                poly = Polygon(grains[n]._newPoints, True) 
                patches.append(poly)
            # Setup up patchCollection 
            pCol = PatchCollection(patches, facecolors='dimgray',edgecolors='black', lw=0.1)
            ax.add_collection(pCol)
       
        else: 
            plt.plot(posRotStep[:,0],posRotStep[:,1],'o')


        if forces and step == nSteps - 1:
            cInfoFiles = ["cinfo_0.dat"]
            cInfoFinal = cInfoFiles[-1] #get filename of final cInfo  
            cAll =  np.loadtxt(Rve.cInfoDir + cInfoFinal)
            cAll = Rve.getCumulativeContacts(cAll)       
            print (cAll.shape)
            for row in cAll:
                ID1,ID2= row[0],row[1]
                pos1,pos2 = posRotStep[int(ID1),:2],posRotStep[int(ID2),:2]
                force = row[2:5]
                normal = row[5:8]
                forceMag = np.abs(force.dot(normal)) #magnitude of normal force 
                lineThickness = forceMag/300000000
                plt.plot( [pos1[0],pos2[0]],[pos1[1],pos2[1]],'b-',linewidth=lineThickness)
 

        #plt.axis('off')
        plt.savefig(figDir + '/step_' + str(step) + '.png', format='png', dpi =  200 )
        #plt.show(block=True)
        plt.close(fig)

    # Create Video
    if (video): 
        string = "ffmpeg -start_number 0 -i %sstep_" % figDir + "%d.png -y -vcodec mpeg4 " + vidName
      #  print (string)
        os.system(string)
    
    return grains
