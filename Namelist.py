#global.py
from scipy.io import netcdf_file
### Experiment settings
Model = 'ERAInterim'
ENS = 'r1i1p1'
TCGIinput = 'TCGI_CRH_SST'
CHAZ_ENS = 1
CHAZ_Int_ENS = 40
PImodelname = 'ERA'
### CHAZ parameters
uBeta = -2.5
vBeta = 1.0
survivalrate = 0.78
seedN = 1000 #annual seeding rate for random seeding
ipath = '/Users/kaitlynnpugliese/Desktop/APAMHurricane/'
opath = '/Users/kaitlynnpugliese/Desktop/APAMHurricane/bt_global_predictors.pik'
obs_bt_path = '/Users/kaitlynnpugliese/Desktop/APAMHurricane/'
Year1 = 1985 
Year2 = 1985 
ibtracs = '/Users/kaitlynnpugliese/Desktop/APAMHurricane/Allstorms.ibtracs_all.v03r08.nc'

landmaskfile = '/Users/kaitlynnpugliese/Desktop/APAMHurricane/landmask.nc'
f = netcdf_file(landmaskfile)
llon = f.variables['lon'][:]
llat = f.variables['lat'][:]
lldmask = f.variables['landmask'][:,:]
ldmask = lldmask[-12::-24,::24]
lldmask = lldmask[::-1,:]

###################################################
# Preporcesses                                 ####
# ignore variavbles when run CHAZ is False     ####
###################################################
###################################################
# CHAZ                                         ####
# ignore variavbles when run CHAZ is False     ####
###################################################
runCHAZ=True
### track
calBam = True 
