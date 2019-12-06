#!/usr/bin/env python

from RPTWorkflow import *
import numpy as np
import lasio
from seppyio import *
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate

matplotlib.rcParams.update({'font.size':15})
matplotlib.rcParams.update({'lines.linewidth':3})

widths=[5,5,5]
gs_kw=dict(width_ratios=widths)
f,(ax1,ax4,ax5)=plt.subplots(1,3,sharey=True,figsize=(15,8),gridspec_kw=gs_kw)

mudSS160=np.genfromtxt('../wells/MudWeightSS160.txt')
mudSS187=np.genfromtxt('../wells/MudWeightSS187.txt')
mudSS191=np.genfromtxt('../wells/MudWeightSS191.txt')
mudST200=np.genfromtxt('../wells/MudWeightST200.txt')
mudST143=np.genfromtxt('../wells/MudWeightST143.txt')
mudST168=np.genfromtxt('../wells/MudWeightST168.txt')

mudUpper=np.genfromtxt('../wells/MudWeightUpper1.txt')
mudLower=np.genfromtxt('../wells/MudWeightLower.txt')

ax1.plot(mudLower[:,1],mudLower[:,0]*0.3048e-3,color='k',label='lower')
ax1.plot(mudUpper[:,1],mudUpper[:,0]*0.3048e-3,color='r',label='upper')

ax1.scatter(mudSS160[:,1],mudSS160[:,0]*0.3048e-3,s=50,label='SS160')
ax1.scatter(mudSS187[:,1],mudSS187[:,0]*0.3048e-3,s=50,c='b',label='SS187')
ax1.scatter(mudSS191[:,1],mudSS191[:,0]*0.3048e-3,s=50,c='k',label='SS191')
ax1.scatter(mudST200[:,1],mudST200[:,0]*0.3048e-3,s=50,label='ST200')
ax1.scatter(mudST143[:,1],mudST143[:,0]*0.3048e-3,s=50,c='orange',label='ST143')
ax1.scatter(mudST168[:,1],mudST168[:,0]*0.3048e-3,s=50,c='g',label='ST168')

ax1.set_ylim([0,5])
ax1.set_xlim([8,18])
ax1.set_xlabel('Mud weight (ppg)')
ax1.set_ylabel('Depth (km)')
ax1.set_xticks(np.arange(8,17,step=2))
ax1.legend()
ax1.invert_yaxis()

dtaum=64.5 # in mus/ft
X=1.97
sigma0=26000 # in psi
rhow=1.05;
beta0=6.5
beta1=14

# SS191
vel0=np.reshape(sepread('../models/velocity_SS191.H',(1,600)),(600,1))
vel0=vel0
depthvel0=np.linspace(0,32.8084*599,600)

lower=np.reshape(sepread('../wells/lowerSS191.H',(1,200)),(200,1))
upper=np.reshape(sepread('../wells/upperSS191.H',(1,200)),(200,1))
upper0=np.reshape(sepread('../wells/upperSS191.0.H',(1,200)),(200,1))

lowerc=np.reshape(sepread('../wells/lowerSS191.corrected.H',(1,200)),(200,1))
tlowerc=np.convolve(lowerc[:,0],np.ones((5,))/5,mode='same')
tlowerc[0:2]=lowerc[0:2,0]
tlowerc[197:200]=lowerc[197:200,0]
lowerc[:,0]=tlowerc

upperc=np.reshape(sepread('../wells/upperSS191.corrected.H',(1,200)),(200,1))
tupperc=np.convolve(upperc[:,0],np.ones((5,))/5,mode='same')
tupperc[0:2]=upperc[0:2,0]
tupperc[197:200]=upperc[197:200,0]
upperc[:,0]=tupperc+100

depthbound=np.linspace(0,0.025*199,200)

las=lasio.read('../wells/177114136300_Orig+Edit+RckPhys.las')
kb=94
wdepth=72
depthlog=las['DEPTH']
depthlog=depthlog[np.isfinite(las['DTLN_ED2'])]
soniclog=las['DTLN_ED2'][np.isfinite(las['DTLN_ED2'])]

Age_depth=np.array([[0],[7.4678e3],[1.0755e4]]) 
mudweight=np.array([[10.5],[10.5],[14.8],[15.8]])
muddepth=np.array([[10927],[13150],[13325],[15078]])

depthlog=depthlog-kb-wdepth

maxdepth=19685 #  6 km deep
dz=0.5
depth=np.arange(0,maxdepth,dz)

Age=np.array([[0],[2.58],[5.33]]) #in My
interp1=interpolate.interp1d(Age_depth[:,0],Age[:,0],axis=0,fill_value='extrapolate')
age=interp1(depth)
t=np.linspace(np.min(age),np.max(age),500)
dt=t[1]-t[0]
interp1=interpolate.interp1d(Age[:,0],Age_depth[:,0],axis=0,fill_value='extrapolate')
deptht=interp1(t)

BHT=np.array([[62],[265],[280],[300]]) # in F
BHT_depth=np.array([[0],[16860],[17352],[19161]])
interp1=interpolate.interp1d(BHT_depth[:,0],BHT[:,0],axis=0,fill_value='extrapolate')
T=interp1(deptht)

smecFrac=smectiteFraction(T,dt)
betaFunc=betaFunction(smecFrac,beta0,beta1)

S=0.000005432*deptht*deptht+0.8783*deptht+0.455*wdepth
pFrac=0.975*S;
pHydro=(deptht+wdepth)*rhow*0.433;

soniclog=np.convolve(soniclog,np.ones((20,))/20,mode='same')

#plt.figure(figsize=(6,8))

dtHydro=p2dtau(pHydro,S,betaFunc,dtaum,X,sigma0)
dtFrac=p2dtau(pFrac,S,betaFunc,dtaum,X,sigma0)
ax5.plot(lowerc*1e-3,depthbound,color='r',label='lower')
ax5.plot(upperc*1e-3,depthbound,color='k',label='upper')
ax5.plot(0.3048e3/soniclog[10:-10]-0.025,depthlog[10:-10]*0.3048e-3,color='m',label='sonic')
ax5.plot(vel0*1e-3-0.025,depthvel0*0.3048e-3,color='c',label='seismic')
#ax5.plot(0.3048e3/dtHydro,deptht*0.3048e-3,label='from hydro')
#ax5.plot(0.3048e3/dtFrac,deptht*0.3048e-3,label='from fracture')

pGrad=10
#for i in range(4):
#    p=pGrad*0.0519*deptht
#    dtau=p2dtau(p,S,betaFunc,dtaum,X,sigma0)
#    ax5.plot(0.3048e3/dtau,deptht*0.3048e-3,linestyle='--',label=str(pGrad)+' ppg')
#    pGrad=pGrad+2
ax5.set_ylim((0,5))
ax5.set_xlim((1.5,4.5))
ax5.set_xlabel('Velocity (km/s)')
#ax5.ylabel('Depth (m)')
#ax5.invert_yaxis()
ax5.legend()
#plt.savefig('../Fig/RPTSS191.pdf',bbox_inches='tight')
#plt.show()

interp1=interpolate.interp1d(depthlog,soniclog,axis=0,fill_value='extrapolate')
sonict=interp1(deptht)
pSonic=dtau2p(sonict,S,betaFunc,dtaum,X,sigma0)
interp1=interpolate.interp1d(depthvel0,vel0[:,0],axis=0,fill_value='extrapolate')
vel=interp1(deptht)
vel=0.3048e6/vel
pVel=dtau2p(vel,S,betaFunc,dtaum,X,sigma0)

pMud=0.0519*mudweight*muddepth;
pMudLower=0.0519*mudLower[:,1]*mudLower[:,0];
pMudUpper=0.0519*mudUpper[:,1]*mudUpper[:,0];

deptht=deptht*0.3048e-3
#plt.figure(figsize=(4,8))
ax4.plot(pMudLower,mudLower[:,0]*0.3048e-3,color='k',label='lower')
ax4.plot(pMudUpper,mudUpper[:,0]*0.3048e-3,color='r',label='upper')
#ax4.plot(pHydro,deptht,label='hydro')
ax4.plot(pSonic[pSonic>0],deptht[pSonic>0],color='m',label='from sonic')
ax4.plot(pVel,deptht,color='c',label='from seismic')
ax4.plot(pFrac,deptht,label='fracture')
ax4.plot(S,deptht,label='overburden')
ax4.scatter(pMud,muddepth*0.3048e-3,s=50,c='k',zorder=10,label='mud')
ax4.set_xlabel('Pressure (psi)')
ax4.set_ylim((0,5))
ax4.invert_yaxis()
ax4.legend()

plt.savefig('../Fig/mudboundSS191.pdf',bbox_inches='tight')
#plt.show()
