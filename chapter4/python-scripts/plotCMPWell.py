#!/usr/bin/env python

import numpy as np
from seppyio import *
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams.update({'lines.linewidth':2})

nx,ny=from_header("vfile","n1","n2")
ox,dx=from_header("vfile","o1","d1")
oy,dy=from_header("vfile","o2","d2")
oox=0
ooy=0

wellx=np.array([650796.7716,662940.6096,651775.7857,683131.4882,679538.8049,669467.0754])-oox
wellx=wellx*1e-3
welly=np.array([1048.853339,-4868.383954,-8624.902802,-22954.51988,-6968.619542,-10249.09604])-ooy
welly=welly*1e-3
wellname=['SS160','SS187','SS191','ST200','ST143','ST168']

v=read("vfile",(ny,nx))
v=v*1e-3
minv=1.5
maxv=4.5

fig, ax = plt.subplots(figsize=(17,10))
cax=ax.imshow(v,interpolation='none',aspect='auto',extent=[(ox-oox)*1e-3,(ox+(nx-1)*dx-oox)*1e-3,(oy+(ny-1)*dy-ooy)*1e-3,(oy-ooy)*1e-3],cmap='jet',vmin=minv,vmax=maxv)
ax.scatter(wellx,welly,s=100,marker='*',c='k')

for i,txt in enumerate(wellname):
    ax.annotate(txt,xy=(wellx[i],welly[i]),xytext=(wellx[i]-1.5,welly[i]+0.5),color='black')

dv=maxv-minv
plt.xlabel('X (km)')
plt.ylabel('Y (km)')
cbar=fig.colorbar(cax,ticks=[minv,minv+dv/5,minv+2*dv/5,minv+3*dv/5,minv+4*dv/5,maxv])
cbar.ax.set_ylabel('Velocity (km/s)')
plt.gca().invert_yaxis()

colors=['b','k','c','m','gold','brown']
S34=['S3_','S4_']
ph=['ph1_','ph4_','ST03_']
k=0
for i in range(2):
    for j in range(3):
        filename='../'+S34[i]+ph[j]+'cmpxy.H'
        ntrace=int(searchFile(filename,'n2'))
        cmpxy=sepread(filename,(ntrace,2))
        minx=(min(cmpxy[:,0])-oox)*1e-3
        maxx=(max(cmpxy[:,0])-oox)*1e-3
        miny=(min(cmpxy[:,1])-ooy)*1e-3
        maxy=(max(cmpxy[:,1])-ooy)*1e-3
        print filename,ntrace,minx,maxx,miny,maxy
        x=np.linspace(minx,maxx,200)
        y=np.linspace(miny,maxy,200)
        plt.plot(x,miny*np.ones((200,)),color=colors[k],label=S34[i]+ph[j])
        plt.plot(x,maxy*np.ones((200,)),color=colors[k])
        plt.plot(minx*np.ones((200,)),y,color=colors[k])
        plt.plot(maxx*np.ones((200,)),y,color=colors[k])
        k=k+1

#plt.xlim((-10,45))
#plt.ylim((-5,30))
plt.legend()
#plt.show()
plt.savefig('../Fig/cmpwell.pdf',bbox_inches='tight')

