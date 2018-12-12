#!/usr/bin/env python
import json
import numpy as np
import calendar
import random
import time
import sys
import dask.array as da
import os
import gc
import pickle
import copy
import pandas as pd
import Namelist as gv
import pandas as pd
import pdb
from netCDF4 import Dataset
from scipy import stats
from util import argminDatetime
from util import int2str,date_interpolation
from scipy.io import loadmat, netcdf_file
from datetime import datetime,timedelta

def func_first(x):
    if x.first_valid_index() is None:
        return None
    else:
        return x.first_valid_index()
def func_last(x):
    if x.last_valid_index() is None:
        return None
    else:
        return x.last_valid_index()

def calF(nday):
     #### furior function
     dt = 1.0*60*60 #1 hr
     #T = np.float(nday)
     T = np.float(15)
     N = 15
     nt = np.arange(0,nday*60*60*24,dt)
     F = np.zeros([nt.shape[0],4])
     F1 = np.zeros([nt.shape[0],4])

     #F = np.zeros([24,4])
     #F1 = np.zeros([24,4])
     for iff in range(0,4,1):
         X = np.zeros([15])
         X = [random.uniform(0,1) for iN in range(N)]
         for itt in range(nt.shape[0]):
         #for itt in range(24):
             F[itt,iff] = (np.sqrt(2.0/np.sum([iN**(-3.0) for iN in range(1,N+1,1)]))*\
                       np.sum([(iN**(-3.0/2.0))*np.sin(2.0*np.pi*(iN*itt/(24.*T)+X[iN-1]))
                       for iN in range(1,N+1,1)]))

     return F

def readIBTrACs(ibtracs,iy):
    nc = Dataset(ibtracs,'r', format='NETCDF3_CLASSIC')
    nsource = 0
    lon = nc.variables['source_lon'][:,:,nsource].T
    lat = nc.variables['source_lat'][:,:,nsource].T
    days = nc.variables['source_time'][:,:].T
    wspd = nc.variables['source_wind'][:,:,nsource].T
    pres = nc.variables['source_pres'][:,:,nsource].T

    iIO = np.argwhere(np.array(lon[0,:])!=-30000.)[:,0]
    lon[lon==-30000] = np.float('nan')
    lat[lat==-30000] = np.float('nan')
    lon = np.array(lon[:,iIO])+360
    lat = np.array(lat[:,iIO])
    wspd = np.array(wspd[:,iIO])
    date = []
    year = []
    count = 0
    for i in iIO:
        date.append(datetime(1858,11,17,0,0)+timedelta(days=np.array(days)[0,i]))
        year.append(date[-1].year)
    year = np.array(year)
    na = np.argwhere(year==iy).ravel()
    climInitDate = np.array(date)[na]
    climInitLon = lon[0,:][na]
    climInitLat = lat[0,:][na]
    return climInitDate,climInitLon,climInitLat

def getTrackPrediction(u250,v250,u850,v850,dt,fstLon,fstLat,fstDate):
       #### modify Beta
       earth_rate = 7.2921150e-5 #mean earth rotation rate in radius per second
       r0 = 6371000 # mean earth radius in m
       lat0 = np.arange(-90,100,10)
       phi0 = lat0/180.*np.pi #original latitude in radian (ex. at 15 degree N)
       beta0 = 2.0*earth_rate*np.cos(phi0)/r0 # per second per m
       beta0 = beta0/beta0[10]
       ratio = np.interp(fstLat,np.arange(-90,100,10),beta0)
       uBeta = gv.uBeta*ratio
       vBeta = gv.vBeta*ratio
       ################

       alpha = 0.8
       uTrack = alpha*u850+(1.-alpha)*u250+uBeta
       vTrack = alpha*v850+(1.-alpha)*v250+vBeta*np.sign(np.sin(fstLat*np.pi/180.))
       dx = uTrack*dt
       dy = vTrack*dt
       lon2,lat2 = getLonLatfromDistance(fstLon,fstLat,dx,dy)
       fstLon,fstLat = lon2,lat2
       fstDate += timedelta(seconds=dt)
       #print uBeta, vBeta,fstLon,fstLat
       return fstLon,fstLat,fstDate

def getLonLatfromDistance(lonInit,latInit,dx,dy):
    er = 6371000 #km
    londis = 2*np.pi*er*np.cos(latInit/180*np.pi)/360.
    lon2 = lonInit+dx/londis
    latdis = 2*np.pi*er/360.
    lat2 = latInit+dy/latdis
    return lon2,lat2


def bam(iS,block_id=None):
     #print iS
     dt = 1.0*60*60
     T = 15
     N = 15
     F = calF(15)
     nt = np.arange(0,T*60*60*24,dt)
     b = np.int(iS.mean(keepdims=True))
     #b = iS
     fstDate = climInitDate[b]
     fstLon = climInitLon[b]
     fstLat = climInitLat[b]
     fstlon[0,b] = fstLon
     fstlat[0,b] = fstLat
     fstldmask[0,b] = 0
     endhours = fstlon.shape[0]-1
     endDate = climInitDate[b] + timedelta(hours = endhours)
     count,year0,month0,day0 = 1,0,0,0
     fpath = gv.ipath+'/'
     while fstDate < endDate:
        if (fstDate.year != year0) :
            fileName = fpath+str(fstDate.year)+'_'+gv.ENS+'.nc'
            #pdb.set_trace()
            if os.path.isfile(fileName):
                nc = Dataset(fileName,'r',format='NETCDF3_CLASSIC')
                u250m = nc.variables['ua2002'][:]
                u850m = nc.variables['ua8502'][:]
                v250m = nc.variables['va2002'][:]
                v850m = nc.variables['va8502'][:]
                xlong = nc.variables['Longitude'][:]
                xlat = nc.variables['Latitude'][:]
                xxlong,xxlat = np.meshgrid(xlong,xlat)
                nc.close()

                if fstDate.day != day0:
                    u250m2d = date_interpolation(fstDate,u250m)
                    v250m2d = date_interpolation(fstDate,v250m)
                    u850m2d = date_interpolation(fstDate,u850m)
                    v850m2d = date_interpolation(fstDate,v850m)

                    FileAName =\
                    fpath+'A_'+str(fstDate.year)+int2str(fstDate.month,2)+'.nc'
                    ncA = Dataset(FileAName,'r',format='NETCDF3_CLASSIC')
                    A = ncA.variables['A'][:,fstDate.day-1,:,:]
                    ncA.close()
                    day0 = fstDate.day

                distance = np.sqrt((fstLon-xxlong)**2+(fstLat-xxlat)**2)
                iy,ix = np.unravel_index(np.argmin(distance),distance.shape)
                iy1,ix1 = np.max([iy-2,0]),np.max([ix-2,0])
                iy2,ix2 = np.min([iy+2,distance.shape[0]]),np.min([ix+2,distance.shape[1]])
                iit = np.mod(count,nt.shape[0])
                u250 = u250m2d[iy1:iy2+1,ix1:ix2+1]+A[0,iy1:iy2+1,ix1:ix2+1]*F[iit,0]
                v250 = v250m2d[iy1:iy2+1,ix1:ix2+1]+A[1,iy1:iy2+1,ix1:ix2+1]*F[iit,0]+A[2,iy1:iy2+1,ix1:ix2+1]*F[iit,1]
                u850 = u850m2d[iy1:iy2+1,ix1:ix2+1]+A[3,iy1:iy2+1,ix1:ix2+1]*F[iit,0]+\
                                 A[4,iy1:iy2+1,ix1:ix2+1]*F[iit,1]+A[5,iy1:iy2+1,ix1:ix2+1]*F[iit,2]
                v850 = v850m2d[iy1:iy2+1,ix1:ix2+1]+A[6,iy1:iy2+1,ix1:ix2+1]*F[iit,0]+\
                                 A[7,iy1:iy2+1,ix1:ix2+1]*F[iit,1]+A[8,iy1:iy2+1,ix1:ix2+1]*F[iit,2]+\
                                 A[9,iy1:iy2+1,ix1:ix2+1]*F[iit,3]
                u250 = np.nanmean(u250)
                u850 = np.nanmean(u850)
                v250 = np.nanmean(v250)
                v850 = np.nanmean(v850)
                fstLon, fstLat, fstDate = getTrackPrediction(u250,v250,u850,v850,dt,fstLon,fstLat,fstDate)
                if ((fstLon<0.0) or (fstLon>360) or (fstLat<-60) or (fstLat>60)):
                        print(b, 'break for going to the space')
                        break
                fstlon[count,b] = fstLon
                fstlat[count,b] = fstLat
                print(gv.ldmask[iy1:iy2+1,ix1:ix2+1])
                fstldmask[count,b] = np.rint(np.nanmean(gv.ldmask[iy1:iy2+1,ix1:ix2+1]))
                del u250,u850,v250,v850
                count += 1
            else:
                print('no'+fileName)
                break
            #print(fstDate, count)
            #print(endDate, count)

     #pdb.set_trace()
     return b

def get_landmask(filename):
    """
    read 0.25degree landmask.nc
    output:
    lon: 1D
    lat: 1D
    landmask:2D

    """
    f = netcdf_file(filename)
    lon = f.variables['lon'][:]
    lat = f.variables['lat'][:]
    landmask = f.variables['landmask'][:,:]
    f.close()

    return lon, lat, landmask

def removeland(iS):
    b = np.int(iS.mean(keepdims=True))
    iT3 = -1
    if b<fstldmask.shape[1]:
        a = np.argwhere(fstldmask[:,b]==3)
        if a.size:
           if a.size>3:
              iT3 = a[0]+2
              fstlon[iT3:,b]=np.NaN
              fstlat[iT3:,b]=np.NaN
    return iT3


def getBam(cDate,cLon,cLat,iy,ichaz,exp):
    global EXP,climInitDate,climInitLon,climInitLat
    EXP,climInitDate,climInitLon,climInitLat = exp,cDate,cLon,cLat
    nnt = np.int_(31) # longest track time
    nS = climInitLon.shape[0]
    global fstlon, fstlat, fstldmask
    fstlon = np.zeros([nnt*24+1,nS])*np.NaN
    fstlat = np.zeros([nnt*24+1,nS])*np.NaN
    fstldmask = np.zeros(fstlat.shape)
    diS = da.from_array(np.int32(np.arange(0,nS,1)),chunks=(1,))
    niS = np.int32(np.arange(0,nS,1))
    #new = da.map_blocks(bam,diS,chunks=(1,1))
    for iiS in niS:
        # print(niS)
        b = bam(iiS)
    #b = new.compute()
    print('removeland')
    fstlon = fstlon[::6,:]
    fstlat = fstlat[::6,:]
    fstldmask = fstldmask[::6,:]
    new = da.map_blocks(removeland, diS, chunks=(1,))
    c = new.compute()
    print('give times')
    ### give times
    fsttime = np.empty(fstlon.shape,dtype=object)
    fsttime[0,:] = climInitDate
    dummy = pd.DataFrame(fstlon)
    iT1 = np.int16(dummy.apply(func_first,axis=0))+1
    iT2 = np.int16(dummy.apply(func_last,axis=0))
    for iS in niS:
        fsttime[iT1[iS]:iT2[iS]+1,iS] = \
        [climInitDate[iS]+timedelta(hours=6*iit) for iit in range(iT1[iS],iT2[iS]+1,1)]
        fst = {'lon':fstlon,'lat':fstlat,'Time':fsttime,'ldmask':fstldmask}
    f = open("'track_'+str(iy)+'_ens'+int2str(ichaz,3)+'.pik'",'wb+')
    pickle.dump(fst,f)
    f.close()
    return fst
