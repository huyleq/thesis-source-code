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

dtaum=64.5 # in mus/ft
X=1.97
sigma0=26000 # in psi
rhow=1.05;
beta0=6.5
beta1=14

# SS187
vel0=np.reshape(sepread('../models/velocity_SS187.H',(1,600)),(600,1))
depthvel0=np.linspace(0,32.8084*599,600)

las=lasio.read('../wells/177114129700_Orig+Edit+RckPhys.las')
kb=96.5
wdepth=60
depthlog=las['DEPTH']
depthlog=depthlog[np.isfinite(las['DT_ED5'])]
soniclog=las['DT_ED5'][np.isfinite(las['DT_ED5'])]

Age_depth=np.array([[0],[7.2277e3],[1.0304e4]]) 



depthlog=depthlog-kb-wdepth

maxdepth=19685 #  6 km deep
dz=0.5
depth=np.arange(0,maxdepth,dz)

widths=[3,3,3,4,6]
gs_kw=dict(width_ratios=widths)
f,(ax1,ax2,ax3,ax4,ax5)=plt.subplots(1,5,sharey=True,figsize=(19,10),gridspec_kw=gs_kw)

Age=np.array([[0],[2.58],[5.33]]) #in My
interp1=interpolate.interp1d(Age_depth[:,0],Age[:,0],axis=0,fill_value='extrapolate')
age=interp1(depth)
t=np.linspace(np.min(age),np.max(age),500)
dt=t[1]-t[0]
interp1=interpolate.interp1d(Age[:,0],Age_depth[:,0],axis=0,fill_value='extrapolate')
deptht=interp1(t)

#plt.figure(figsize=(4,8))
ax2.plot(age,depth*0.3048e-3,label='linear fit')
ax2.scatter(Age[1:],Age_depth[1:]*0.3048e-3,s=50,c='k',label='interpreted',zorder=10)
ax2.set_ylim((0,6))
ax2.legend()
ax2.set_xlabel('Age (My)')
#ax2.set_ylabel('Depth (m)')
ax2.invert_yaxis()
#plt.savefig('../Fig/ageSS187.pdf',bbox_inches='tight')
#plt.show()

BHT=np.array([[62],[265],[280],[300]]) # in F
BHT_depth=np.array([[0],[16860],[17352],[19161]])
interp1=interpolate.interp1d(BHT_depth[:,0],BHT[:,0],axis=0,fill_value='extrapolate')
T=interp1(deptht)

#plt.figure(figsize=(4,8))
ax1.plot(T,deptht*0.3048e-3,label='linear fit')
ax1.scatter(BHT[1:],BHT_depth[1:]*0.3048e-3,s=50,c='k',label='BHT',zorder=10)
#ax1.set_ylim((0,7000))
ax1.legend()
ax1.set_xlabel('Temperature (F)')
ax1.set_ylabel('Depth (km)')
ax1.invert_yaxis()
#plt.savefig('../Fig/tempSS187.pdf',bbox_inches='tight')
#plt.show()

smecFrac=smectiteFraction(T,dt)
betaFunc=betaFunction(smecFrac,beta0,beta1)

#plt.figure(figsize=(4,8))
ax3.plot(smecFrac,deptht*0.3048e-3)
ax3.set_ylim((0,6))
ax3.set_xlabel('Smectite fraction')
#plt.ylabel('Depth (m)')
ax3.invert_yaxis()
#plt.savefig('../Fig/betaSS187.pdf',bbox_inches='tight')
#plt.show()

S=0.000005432*deptht*deptht+0.8783*deptht+0.455*wdepth
pFrac=0.975*S;
pHydro=(deptht+wdepth)*rhow*0.433;

soniclog=np.convolve(soniclog,np.ones((20,))/20,mode='same')

#plt.figure(figsize=(6,8))

dtHydro=p2dtau(pHydro,S,betaFunc,dtaum,X,sigma0)
ax5.plot(0.3048e3/dtHydro,deptht*0.3048e-3,label='from hydro')
dtFrac=p2dtau(pFrac,S,betaFunc,dtaum,X,sigma0)
ax5.plot(0.3048e3/dtFrac,deptht*0.3048e-3,label='from fracture')
ax5.plot(0.3048e3/soniclog[10:-10],depthlog[10:-10]*0.3048e-3,label='sonic')
ax5.plot(vel0*1e-3,depthvel0*0.3048e-3,label='seismic')

pGrad=10
for i in range(4):
    p=pGrad*0.0519*deptht
    dtau=p2dtau(p,S,betaFunc,dtaum,X,sigma0)
    ax5.plot(0.3048e3/dtau,deptht*0.3048e-3,linestyle='--',label=str(pGrad)+' ppg')
    pGrad=pGrad+2
ax5.set_ylim((0,6))
ax5.set_xlim((1.5,4.5))
ax5.set_xlabel('Velocity (km/s)')
#ax5.ylabel('Depth (m)')
ax5.invert_yaxis()
ax5.legend()
#plt.savefig('../Fig/RPTSS187.pdf',bbox_inches='tight')
#plt.show()

interp1=interpolate.interp1d(depthlog,soniclog,axis=0,fill_value='extrapolate')
sonict=interp1(deptht)
pSonic=dtau2p(sonict,S,betaFunc,dtaum,X,sigma0)

interp1=interpolate.interp1d(depthvel0,vel0[:,0],axis=0,fill_value='extrapolate')
vel=interp1(deptht)
vel=0.3048e6/vel
pVel=dtau2p(vel,S,betaFunc,dtaum,X,sigma0)

mudweight=np.array([[9.3],[9.7],[11],[15.4],[16.6],[17],[17.7],[17.9]]);
muddepth=np.array([[3500],[9775],[11800],[12870],[14343],[14671],[15961],[17500]]);
pMud=0.0519*mudweight*muddepth;

deptht=deptht*0.3048e-3

#plt.figure(figsize=(4,8))
ax4.plot(pHydro,deptht,label='hydro')
ax4.plot(pFrac,deptht,label='fracture')
ax4.plot(pSonic,deptht,label='from sonic')
ax4.plot(pVel,deptht,label='from seismic')
ax4.plot(S,deptht,label='overburden')
ax4.scatter(pMud,muddepth*0.3048e-3,s=50,c='k',zorder=10,label='mud')
ax4.set_xlabel('Pressure (psi)')
#ax4.ylabel('Depth (m)')
ax4.set_ylim((0,6))
ax4.invert_yaxis()
ax4.legend()
plt.tight_layout()
#plt.savefig('../Fig/SS187.pdf',bbox_inches='tight')
plt.show()
