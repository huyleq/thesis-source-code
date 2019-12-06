#!/usr/bin/env python

import numpy as np
from seppyio import *
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 15})

shiftx=650
shifty=-20
nx,ny=from_header("vfile","n1","n2")
ox,dx=from_header("vfile","o1","d1")
oy,dy=from_header("vfile","o2","d2")
dx=dx*1e-3
dy=dy*1e-3

wellx=np.array([650796.7716,662940.6096,651775.7857,683131.4882,679538.8049,669467.0754])#-ox
wellx=wellx*1e-3-shiftx
welly=np.array([1048.853339,-4868.383954,-8624.902802,-22954.51988,-6968.619542,-10249.09604])#-oy
welly=welly*1e-3-shifty
wellname=['SS160','SS187','SS191','ST200','ST143','ST168']

ox=ox*1e-3-shiftx
oy=oy*1e-3-shifty

v=read("vfile",(ny,nx))
v=v*1e-3
minv=1.5
maxv=4.5

nsou=from_header('souloc','n2')
souloc=read('souloc',(nsou,5))
souloc[:,0]=souloc[:,0]*1e-3-shiftx
souloc[:,1]=souloc[:,1]*1e-3-shifty
ntrace=from_header('recloc','n2')
nreclocdim=from_header('recloc','n1')
recloc=read('recloc',(ntrace,nreclocdim))
recloc[:,0]=recloc[:,0]*1e-3-shiftx
recloc[:,1]=recloc[:,1]*1e-3-shifty

fig, ax = plt.subplots(figsize=(10,8))
#cax=ax.imshow(v,interpolation='none',aspect='auto',extent=[0,(nx-1)*dx,(ny-1)*dy,0],cmap='jet',vmin=minv,vmax=maxv)
#cax=ax.imshow(v,interpolation='none',aspect='auto',alpha=0.5,extent=[ox,ox+(nx-1)*dx,oy+(ny-1)*dy,oy],cmap='jet',vmin=minv,vmax=maxv)

ax.scatter(recloc[0:ntrace:10,0],recloc[0:ntrace:10,1],s=10,marker='.',c='y',label='sou')
#ax.scatter(recloc[:,0],recloc[:,1],s=10,marker='.',c='y',label='sou')
ax.scatter(souloc[:,0],souloc[:,1],s=10,marker='.',c='b',label='rec')
#ax.scatter(gxy[:,0],gxy[:,1],s=10,marker='.',c='b',label='rec')
#ax.scatter(cmpxy[:,0],cmpxy[:,1],s=10,marker='.',c='y',label='cmp')
#ax.scatter(wellx,welly,s=100,marker='*',c='k',label='well')
#
#for i,txt in enumerate(wellname):
#    ax.annotate(txt,xy=(wellx[i],welly[i]),xytext=(wellx[i]-1.5,welly[i]+0.5),color='black')

cax=ax.imshow(v,interpolation='none',aspect='auto',alpha=0.2,extent=[ox,ox+(nx-1)*dx,oy+(ny-1)*dy,oy],cmap='jet',vmin=minv,vmax=maxv,zorder=10)

plt.xlabel('X (km)')
plt.ylabel('Y (km)')
plt.gca().invert_yaxis()

#plt.legend(loc='upper right')
plt.xlim(0,15)
plt.ylim(0,15)

#dv=maxv-minv
#cbar=fig.colorbar(cax,ticks=[minv,minv+dv/5,minv+2*dv/5,minv+3*dv/5,minv+4*dv/5,maxv])
#cbar.ax.set_ylabel('Velocity (km/s)')

plt.savefig('./Fig/datasr.pdf',bbox_inches='tight')
plt.show()
#print 'saving figure'

#x=np.linspace(0,1066*25,1067)
#x=x*1e-3
#y=np.linspace(-10320-oy,-10320-oy,1067)
#y=y*1e-3
#plt.plot(x,y,linewidth=3,linestyle='--',c='k')
#
##plt.show()
#plt.savefig('../Fig/welllocation1.pdf',bbox_inches='tight')

#plt.show()
