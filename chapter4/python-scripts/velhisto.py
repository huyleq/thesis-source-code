#!/usr/bin/env python

import numpy as np
from seppyio import *
import matplotlib
import matplotlib.pyplot as plt

nx,nz=from_header("vel1","n1","n2")
ox,dx=from_header("vel1","o1","d1")
oz,dz=from_header("vel1","o2","d2")

vel1=read("vel1",(nz,nx))
vel2=read("vel2",(nz,nx))

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
plt.hist(vel1[bz:ez,bx:ex].flatten(),100,facecolor='magenta',label=label1)
plt.hist(vel2[bz:ez,bx:ex].flatten(),100,facecolor='blue',alpha=0.4,label=label2)
plt.legend()
plt.xlabel('Velocity')
plt.ylabel('Count')
plt.show()
figname=searchArgv("figname")
#plt.savefig(figname,bbox_inches='tight')
