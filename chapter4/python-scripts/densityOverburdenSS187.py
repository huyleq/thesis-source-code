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

maxdepth=22966 #  5 km deep
dz=0.5
depth=np.arange(0,maxdepth,dz)
n=depth.shape[0]
#vel0=np.reshape(sepread('../wells/velocity_SS187.H',(1,1500)),(1500,1))
vel0=np.reshape(sepread('../models/vz_SS187.H',(1,600)),(600,1))
depthvel0=np.arange(0,32.8084*vel0.shape[0],32.8084)
vel=np.interp(depth,depthvel0,vel0[:,0])

las=lasio.read('../wells/177114129700_Orig+Edit+RckPhys.las')
kb=96.5
wdepth=60
depthlog=las['DEPTH']
depthlog=depthlog-kb-wdepth
bdepthlog=np.min(depthlog)
edepthlog=np.max(depthlog)

rho=np.zeros((n,1))
rhogard=np.zeros((n,1))
for i in range(n):
    if depth[i]<bdepthlog or depth[i]>edepthlog:
        rho[i]=0.31*np.power(vel[i],0.25)
        rhogard[i]=0.31*np.power(vel[i],0.25)
    else:
        j=int((depth[i]-bdepthlog)/dz)
        if np.isfinite(las['RHOB'][j]):
            rho[i]=las['RHOB'][j]
        elif np.isfinite(las['RHOZ'][j]):
            rho[i]=las['RHOZ'][j]
        else:
            rho[i]=0.31*np.power(vel[i],0.25)
        if np.isfinite(las['RHOGARD'][j]):
            rhogard[i]=las['RHOGARD'][j]
        else:
            rhogard[i]=0.31*np.power(vel[i],0.25)

s=overburden(wdepth,rho,dz)
sgard=overburden(wdepth,rhogard,dz)
semp=0.000005432*depth*depth+0.8783*depth+0.455*wdepth

Age=np.array([[0],[2.58],[5.33]]) #in My
Age_depth=np.array([[0],[7.2277e3],[1.0304e4]]) 
interp1=interpolate.interp1d(Age_depth[:,0],Age[:,0],axis=0,fill_value='extrapolate')
age=interp1(depth)
t=np.linspace(np.min(age),np.max(age),500)
dt=t[1]-t[0]
interp1=interpolate.interp1d(Age[:,0],Age_depth[:,0],axis=0,fill_value='extrapolate')
deptht=interp1(t)

BHT=np.array([[62],[265],[280],[300]]) # in F
BHT_depth=np.array([[0],[16860],[17352],[19161]])
interp1=interpolate.interp1d(BHT_depth[:,0],BHT[:,0],axis=0,fill_value='extrapolate')
tem=interp1(depth)
T=interp1(deptht)

smecFrac=smectiteFraction(T,dt)
beta0=6.5
beta1=13
betaFunc=betaFunction(smecFrac,beta0,beta1)

W=5
depth0=3300
depth1=12830
gamLog1=np.convolve(las['GRR'],np.ones((W,))/W,mode='same')
rhoLog1=np.convolve(las['RHOZ'],np.ones((W,))/W,mode='same')
dtauLog=np.convolve(las['DT_ED5'],np.ones((W,))/W,mode='same')
shaleDepth1,shaleGam1,shaleRho1=selectSandShale(depthlog,gamLog1,5,90,rhoLog1,depth0,depth1)[3:]
shaleDtau1=selectSandShale(depthlog,gamLog1,5,90,dtauLog,depth0,depth1)[5]

depth0=12920
depth1=15960
gamLog2=np.convolve(las['GRS'],np.ones((W,))/W,mode='same')
rhoLog2=np.convolve(las['RHOB'],np.ones((W,))/W,mode='same')
shaleDepth2,shaleGam2,shaleRho2=selectSandShale(depthlog,gamLog2,5,90,rhoLog2,depth0,depth1)[3:]
shaleDtau2=selectSandShale(depthlog,gamLog2,5,90,dtauLog,depth0,depth1)[5]

shaleDepth=np.array(shaleDepth1+shaleDepth2)
shaleGam=np.array(shaleGam1+shaleGam2)
shaleRho=np.array(shaleRho1+shaleRho2)
shaleDtau=np.array(shaleDtau1+shaleDtau2)

interp1=interpolate.interp1d(deptht,smecFrac[:,0],axis=0,fill_value='extrapolate')
shaleSmecFrac=interp1(shaleDepth)
A=np.concatenate((np.reshape(shaleDtau,(-1,1)),np.reshape(shaleSmecFrac,(-1,1)),1-np.reshape(shaleSmecFrac,(-1,1))),axis=1)
At=np.transpose(A)
AtA=np.dot(At,A)
b=np.dot(At,shaleRho)
x=np.linalg.solve(AtA,b)
a=x[0]
bs=x[1]
bi=x[2]
print a,bs,bi

#shaleDepth=shaleDepth*0.3048
#depthlog=depthlog*0.3048
#plt.subplot(1,3,1)
#plt.plot(las['GRR'],depthlog,'blue',gamLog1,depthlog,'green',las['GRS'],depthlog,'blue',gamLog2,depthlog,'green',zorder=0)
#plt.scatter(shaleGam,shaleDepth,s=20,color='red',zorder=1)
#plt.gca().invert_yaxis()
#plt.subplot(1,3,2)
#plt.plot(las['RHOZ'],depthlog,'blue',rhoLog1,depthlog,'green',las['RHOB'],depthlog,'blue',rhoLog2,depthlog,'green',zorder=0)
#plt.scatter(shaleRho,shaleDepth,s=20,color='red',zorder=1)
#plt.gca().invert_yaxis()
#plt.subplot(1,3,3)
#plt.plot(las['DT_ED5']*3.28084e-3,depthlog,'blue',dtauLog*3.28084e-3,depthlog,'green',zorder=0)
#plt.scatter(shaleDtau*3.28084e-3,shaleDepth,s=20,color='red',zorder=2)
#plt.gca().invert_yaxis()
#plt.show()

dtau=np.linspace(60,180,200)
rhoSmec=a*dtau+bs
rhoIll=a*dtau+bi
interp1=interpolate.interp1d(0.3048e6/vel[0:30000],depth[0:30000],axis=0,fill_value='extrapolate')
depth1=interp1(dtau)
interp1=interpolate.interp1d(deptht,smecFrac[:,0],axis=0,fill_value='extrapolate')
smecFrac1=interp1(depth1)
rhoDia=smecFrac1*(a*dtau+bs)+(1-smecFrac1)*(a*dtau+bi)
dtauinms=0.3048e6/dtau
shaleDepth=shaleDepth*0.3048*1e-3
b=1
e=5
be=0.5*(b+e)
plt.figure()

#plt.plot(dtauinms,rhoDia,c='k',label='diagenetic')
#plt.plot(dtauinms,rhoSmec,c='k',linestyle='--',label='smectite')
#plt.plot(dtauinms,rhoIll,c='k',linestyle='-.',label='illite')
#sc=plt.scatter(0.3048e6/shaleDtau,shaleRho,c=shaleDepth,cmap='jet',vmin=b,vmax=e)
#plt.xlabel('Velocity (m/s)')

plt.plot(dtau,rhoDia,c='k',label='diagenetic')
plt.plot(dtau,rhoSmec,c='k',linestyle='--',label='smectite')
plt.plot(dtau,rhoIll,c='k',linestyle='-.',label='illite')
sc=plt.scatter(shaleDtau,shaleRho,c=shaleDepth,cmap='jet',vmin=b,vmax=e)
plt.xlabel('Transit time (us/ft)')

plt.legend()
cbar=plt.colorbar(sc,ticks=[b,0.5*(b+be),be,0.5*(be+e),e])
cbar.ax.set_ylabel('Depth (km)')
plt.ylabel('Density (g/cc)')
#plt.show()
#plt.savefig('../Fig/densitySS187.pdf',bbox_inches='tight')

depthinm=depth*0.3048
interp1=interpolate.interp1d(deptht,smecFrac[:,0],axis=0,fill_value='extrapolate')
smecFrac1=interp1(depth)
rhoDia1=smecFrac1*(a*0.3048e6/vel+bs)+(1-smecFrac1)*(a*0.3048e6/vel+bi)
plt.figure()
plt.plot(rho,depthinm,'r',label='Log')
plt.plot(rhogard,depthinm,'g',label='Gardner')
plt.plot(rhoDia1,depthinm,'b',label='Diagenetic')
plt.legend()
plt.xlabel('Density (g/cc)')
plt.ylabel('Depth (m)')
plt.ylim((0,5000))
plt.gca().invert_yaxis()
#plt.savefig('../Fig/densities.pdf',bbox_inches='tight')
#plt.show()

matplotlib.rcParams.update({'lines.linewidth':2})
sDia1=overburden(wdepth,rhoDia1,dz)
plt.figure()
plt.plot(s,depthinm,'r',label='Log')
plt.plot(sgard,depthinm,'g',label='Gardner')
plt.plot(sDia1,depthinm,'b',label='Diagenetic')
plt.plot(semp,depthinm,'k',label='Emperical')
plt.legend()
plt.xlabel('Pressure (psi)')
plt.ylabel('Depth (m)')
plt.ylim((0,5000))
plt.xlim((0,20000))
plt.gca().invert_yaxis()
#plt.savefig('../Fig/overburden.pdf',bbox_inches='tight')
plt.show()
