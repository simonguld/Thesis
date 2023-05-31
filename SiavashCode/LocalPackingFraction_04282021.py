#example showing usage of rve 
import helper
import plotConfiguration
import analyzeSim
import computePackingFraction
from rve import rve

morphDirName = "config-21/" #directory where morph files are storied 
mainDirName = "config-21/" #directory where all relevant information to specific simulation is stored 

Rve = rve(morphDirName,mainDirName) #create instance of rve class for this simulation 
Rve.getDataFromFiles()
Rve.getPackingFraction('.','config21.png','Last Step',kostas = True,allSteps = False)



