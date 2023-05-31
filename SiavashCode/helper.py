#helper functions for runDEM that are not in a class
import os 
import pathlib
import numpy as np
import natsort
import pickle as pickle ##NB pickle5 was replaced by pickle 


def loadObj(name):
	with open(name, 'rb') as f:
		print (f)
		return pickle.load(f)

def getSortedFileList(Dir):
	files = [f for f in os.listdir(Dir) if os.path.isfile(os.path.join(Dir, f)) and f[-3:] == "pkl" ]
	return natsort.natsorted(files)

def addValue(PFs,newA,newT,PF,trial): #add PF to PFs, for ampltidude newA and period newT
	for rownum,row in enumerate(PFs): #is there already a row for this state?
		A,T = row[0],row[1]
		if (A == newA) and (T == newT):
			PFs[rownum] = [newA,newT,trial,PF]
			return PFs	

	#if this row doesn't exist 
	PFs += [[newA,newT,trial,PF]]
	return PFs

