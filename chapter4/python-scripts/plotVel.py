#!/usr/bin/env python

from RPTWorkflow import *
import numpy as np
import lasio
from seppyio import *
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate

matplotlib.rcParams.update({'font.size':15})
matplotlib.rcParams.update({'lines.linewidth':2})

nx=1067
nz=402
dx=0.025
dz=0.025
minv=1.5
maxv=4.5
dv=maxv-minv

legacy=sepread('../line1/bgv1nosaltwater.H',(nz,nx))*1e-3
velmud=sepread('../line1/vel1MudWeightAverage.H',(nz,nx))*1e-3

fig,ax = plt.subplots()
cax=ax.imshow(legacy,interpolation='none',aspect='auto',extent=[0,(nx-1)*dx,(nz-1)*dz,0],cmap='jet',vmin=minv,vmax=maxv)
plt.xlabel('X (km)')
plt.ylabel('Z (km)')
cbar=fig.colorbar(cax,ticks=[minv,minv+dv/5,minv+2*dv/5,minv+3*dv/5,minv+4*dv/5,maxv])
cbar.ax.set_ylabel('Velocity (km/s)')
plt.savefig('../Fig/legacy.pdf')
#plt.show()

mio1=sepread('../line1/TopMio1.H',(1,1067))*1e-3
mio2=sepread('../line1/Mio1.H',(1,1067))*1e-3
plio1=sepread('../line1/TopPlio1.H',(1,1067))*1e-3
x=np.linspace(0,1066*25,1067)*1e-3

fig,ax = plt.subplots()
cax=ax.imshow(velmud,interpolation='none',aspect='auto',extent=[0,(nx-1)*dx,(nz-1)*dz,0],cmap='jet',vmin=minv,vmax=maxv)
ax.plot(x,mio1[0,:],x,mio2[0,:],x,plio1[0,:],c='k',linestyle='--')
plt.xlabel('X (km)')
plt.ylabel('Z (km)')
cbar=fig.colorbar(cax,ticks=[minv,minv+dv/5,minv+2*dv/5,minv+3*dv/5,minv+4*dv/5,maxv])
cbar.ax.set_ylabel('Velocity (km/s)')
plt.savefig('../Fig/velmud.pdf')
#plt.show()

