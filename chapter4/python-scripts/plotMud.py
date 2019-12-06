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

widths=[4,7]
gs_kw=dict(width_ratios=widths)
f,(ax1,ax2)=plt.subplots(1,2,sharey=True,figsize=(11,8),gridspec_kw=gs_kw)

muda=np.genfromtxt('../wells/MudWeightAverage.txt')
mudSS187=np.genfromtxt('../wells/MudWeightSS187.txt')
mudSS191=np.genfromtxt('../wells/MudWeightSS191.txt')
mudST143=np.genfromtxt('../wells/MudWeightST143.txt')
mudST168=np.genfromtxt('../wells/MudWeightST168.txt')

ax1.plot(muda[:,1],muda[:,0]*0.3048e-3,c='k',label='average')
ax1.scatter(mudSS187[:,1],mudSS187[:,0]*0.3048e-3,s=50,label='SS187')
ax1.scatter(mudSS191[:,1],mudSS191[:,0]*0.3048e-3,s=50,label='SS191')
ax1.scatter(mudST143[:,1],mudST143[:,0]*0.3048e-3,s=50,label='ST143')
ax1.scatter(mudST168[:,1],mudST168[:,0]*0.3048e-3,s=50,label='ST168')
ax1.set_ylim([0,7])
ax1.set_xlim([8,17])
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

# ST168
vel0=np.reshape(sepread('../wells/velocity_ST168.H',(1,1500)),(1500,1))
depthvel0=np.linspace(0,32.8084*1499,1500)

las=lasio.read('../wells/177154117301.las')
kb=130
wdepth=70
depthlog=las['DEPT']
depthlog=depthlog-kb-wdepth

maxdepth=22966 #  7 km deep
dz=0.5
depth=np.arange(0,maxdepth,dz)

Age=np.array([[0],[2.58],[5.33]]) #in My
Age_depth=np.array([[0],[6.9781e3],[9.7039e3]]) 
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

velST168=np.reshape(sepread('../line1/bgv1nosaltwater.ST168.H',(1,401)),(401,))
velMudST168=np.reshape(sepread('../line1/vel1MudWeightAverage.ST168.H',(1,401)),(401,))
depthST168=np.linspace(0,25*400,401)
ax2.plot(velST168*1e-3,depthST168*1e-3,label='legacy',zorder=10)
ax2.plot(velMudST168*1e-3,depthST168*1e-3,c='k',label='mud',zorder=10)

pGrad=8
for i in range(9):
    p=pGrad*0.0519*deptht
    dtau=p2dtau(p,S,betaFunc,dtaum,X,sigma0)
    ax2.plot(0.3048e3/dtau,deptht*0.3048e-3,linestyle='--',label=str(pGrad)+' ppg')
    pGrad=pGrad+1
ax2.set_ylim((0,7))
ax2.set_xlim((1.5,4.5))
ax2.set_xlabel('Velocity (km/s)')
ax2.invert_yaxis()
ax2.legend()

plt.savefig('../Fig/mudweights.pdf',bbox_inches='tight')
plt.show()
