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

dtaum,X=get_param('dtaum','X')
sigma0=get_param('sigma0')
beta0,beta1=get_param('beta0','beta1')
rhow=1.05;

# SS160
vel0=read('vel',(600,1))
depthvel0=np.linspace(0,32.8084*599,600)

las=lasio.read('../wells/177114095100_Orig+Edit+RckPhys.las')
kb=94.5
wdepth=50
depthlog=las['DEPTH']
depthlog=depthlog[np.isfinite(las['DT_ED7'])]
soniclog=las['DT_ED7'][np.isfinite(las['DT_ED7'])]

Age_depth=np.array([[0],[5.9538e3],[8.2555e3]]) 
mudweight=np.array([[9.3],[9.7],[11],[15.4],[16.6],[17],[17.7],[17.9]])
muddepth=np.array([[3500],[9775],[11800],[12870],[14343],[14671],[15961],[17500]])

depthlog=depthlog-kb-wdepth

maxdepth=19685 #  6 km deep
dz=0.5
depth=np.arange(0,maxdepth,dz)

widths=[4,6]
gs_kw=dict(width_ratios=widths)
f,(ax4,ax5)=plt.subplots(1,2,sharey=True,figsize=(10,10))

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

mudUpper=np.genfromtxt('../wells/MudWeightUpper1.txt')
pMudUpper=0.0519*mudUpper[:,0]*mudUpper[:,1]
pUpper=0.0519*deptht*np.interp(deptht,mudUpper[:,0],mudUpper[:,1])

w,pct=get_param('w','pct') ## low frequency velocity is 6% slower than high frequency
w=int(w)
soniclog=(1/(1-pct))*np.sqrt(np.convolve(soniclog*soniclog,np.ones((w,))/w,mode='same'))

#plt.figure(figsize=(6,8))

dtHydro=p2dtau(pHydro,S,betaFunc,dtaum,X,sigma0)
ax5.plot(0.3048e3/dtHydro,deptht*0.3048e-3,label='from hydro')
dtFrac=p2dtau(pFrac,S,betaFunc,dtaum,X,sigma0)
ax5.plot(0.3048e3/dtFrac,deptht*0.3048e-3,label='from fracture')
ax5.plot(0.3048e3/soniclog[w/2:-w/2],depthlog[w/2:-w/2]*0.3048e-3,label='sonic')
ax5.plot(vel0*1e-3,depthvel0*0.3048e-3,label='seismic')
dtUpper=p2dtau(pUpper,S,betaFunc,dtaum,X,sigma0)
ax5.plot(0.3048e3/dtUpper,deptht*0.3048e-3,c='k',label='Upper')


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
#ax5.invert_yaxis()
ax5.legend()
#plt.savefig('../Fig/RPTSS160.pdf',bbox_inches='tight')
#plt.show()

interp1=interpolate.interp1d(depthlog,soniclog,axis=0,fill_value='extrapolate')
sonict=interp1(deptht)
pSonic=dtau2p(sonict,S,betaFunc,dtaum,X,sigma0)
interp1=interpolate.interp1d(depthvel0,vel0[:,0],axis=0,fill_value='extrapolate')
vel=interp1(deptht)
vel=0.3048e6/vel
pVel=dtau2p(vel,S,betaFunc,dtaum,X,sigma0)

pMud=0.0519*mudweight*muddepth;

deptht=deptht*0.3048e-3
#plt.figure(figsize=(4,8))
ax4.plot(pHydro,deptht,label='hydro')
ax4.plot(pFrac,deptht,label='fracture')
ax4.plot(pSonic[pSonic>0],deptht[pSonic>0],label='from sonic')
ax4.plot(pVel,deptht,label='from seismic')
ax4.plot(pUpper,deptht,c='k',label='Upper')
#ax4.plot(S,deptht,label='overburden')
ax4.scatter(pMud,muddepth*0.3048e-3,s=50,c='k',zorder=10,label='mud')
ax4.set_xlabel('Pressure (psi)')
ax4.set_ylabel('Depth (km)')
ax4.set_ylim((0,6))
ax4.invert_yaxis()
ax4.legend()
plt.tight_layout()
#plt.savefig('../Fig/ppRPTSS160.pdf',bbox_inches='tight')
f.suptitle('SS160')
plt.show()
