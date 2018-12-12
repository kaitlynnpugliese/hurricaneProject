#!/usr/bin/env python
import json
import numpy as np
import pandas as pd
import subprocess
import dask.array as da
import matplotlib.pyplot as plt

#subprocess.call('cp /home/clee/CMIP5/semi-auto/*.py .',shell=True)

#rename module_GenBamPred_noSynWind file to GBP and import
#rename Namelist file to gv and import
import module_GenBamPred_noSynWind as GBP
import Namelist as gv

print(gv.Model, gv.ENS, gv.TCGIinput)
EXP = 'HIST'
#######################
### Run CHAZ       ####
#######################
if gv.runCHAZ:
    ### running genesis,track,predictors,& intensity
    ichaz = 0
    #loop through from initial year to final year (taken from gv file)
    for iy in range(gv.Year1,gv.Year2+1):
        #genesis
        climInitDate, climInitLon, climInitLat = GBP.readIBTrACs(gv.ibtracs,iy)#readIBTrACs file is a file with observed genesis location of hurricanes
        if gv.calBam: #calBam is set in gv file
           print(iy, 'Bam') #print the year
           fst = GBP.getBam(climInitDate,climInitLon,climInitLat,iy,ichaz,EXP) #output file
print(climInitDate, climInitLon, climInitLat)
print(fst)
plt.plot(fst['lon'], fst['lat'], 'k-')
plt.show()
