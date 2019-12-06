#!/usr/bin/env python

import numpy as np
from seppyio import *
import matplotlib
import matplotlib.pyplot as plt

nz,nx=from_header("sem1","n1","n2")
oz,dz=from_header("sem1","o1","d1")
ox,dx=from_header("sem1","o2","d2")

sem1=np.transpose(read("sem1",(nx,nz)))
sem2=np.transpose(read("sem2",(nx,nz)))

minx,maxx=get_param("minx","maxx")
bx=int((minx-ox)/dx)
ex=int((maxx-ox)/dx)
minz,maxz=get_param("minz","maxz")
bz=int((minz-oz)/dz)
ez=int((maxz-oz)/dz)

label1=searchArgv('label1')
label2=searchArgv('label2')

matplotlib.rcParams.update({'font.size': 15})
plt.figure()
plt.hist(sem1[bz:ez,bx:ex].flatten(),100,facecolor='magenta',label=label1)
plt.hist(sem2[bz:ez,bx:ex].flatten(),100,facecolor='blue',alpha=0.4,label=label2)
plt.legend()
plt.xlim((0.01,1))
#plt.ylim((0,1200))
plt.xlabel('Semblance')
plt.ylabel('Count')
#plt.show()
figname=searchArgv("figname")
plt.savefig(figname,bbox_inches='tight')
