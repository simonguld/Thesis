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
from scipy.optimize import minimize

def calculate_perimeter(a,b):
	perimeter = np.pi * ( 3*(a+b) - np.sqrt( (3*a + b) * (a + 3*b) ) )
	return perimeter

def ellipseCost(AR):
	a,b = AR,1 
	perim = calculate_perimeter(a,b)
	area = np.pi * a * b
	circularityCurr = (2*np.sqrt(np.pi*area)/(perim))**2
	return (circularityCurr - circularity)**2 

aspectRatioList = np.arange(0.1,0.9,0.1)[::-1]
convexityList = [1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]

count = 0 
for j,aspectRatio in enumerate(aspectRatioList):
	for i,convexity in enumerate(convexityList):
		count += 1 
		print ("AR",aspectRatio,"C",convexity)
		attempt = 0 
		mainDirName = "convexityTest/C_%.2f_AR_%.2f/"%(convexity,aspectRatio) 
		morphDirName = mainDirName
		Rve = rve(morphDirName,mainDirName) #initialize RVE instance with default variables 
		Rve.Convexity = convexity
		#Parameters for cloning 
		Rve.aspectRatio = aspectRatio # max axis/min axis of starting ellipse
		Rve.trial = attempt
		Rve.nGrains = 1
		Rve.nShapes = 1
		check  = Rve.createParticles()
 
