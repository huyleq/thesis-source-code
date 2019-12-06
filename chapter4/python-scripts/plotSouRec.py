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

ox=648867
oy=-24896
ntrace=989838
sgxy0=sepread('../line1/S4_ph4_line1.sxygxy.H',(ntrace,4))
sgxy0=sgxy0[0::100,:]
cmpxy=sepread('../line1/S4_ph4_line1.cmpxy.H',(ntrace,2))
cmpxy=cmpxy[0::100,:]
sgxy1=sepread('../line1/S4_ph4_line1_rotated.sxygxy.H',(ntrace,4))
sgxy1=sgxy1[0::100,:]

plt.figure()
plt.scatter((cmpxy[:,0]-ox)*1e-3,(cmpxy[:,1]-oy)*1e-3,label='cmp',c='k')
plt.scatter((sgxy0[:,0]-ox)*1e-3,(sgxy0[:,1]-oy)*1e-3,facecolor='none',edgecolor='r',label='source')
plt.scatter((sgxy0[:,2]-ox)*1e-3,(sgxy0[:,3]-oy)*1e-3,facecolor='none',edgecolor='b',label='receiver')
plt.xlabel('X (km)')
plt.xlim([-10,25])
plt.ylim([13,16.5])
plt.ylabel('Y (km)')
plt.legend(loc=1)
plt.savefig('../Fig/geom0.pdf',bbox_inches='tight')
#plt.show()

plt.figure()
plt.scatter((sgxy1[:,0]-ox)*1e-3,(sgxy1[:,1]-oy)*1e-3,facecolor='none',edgecolor='r',label='source')
plt.scatter((sgxy1[:,2]-ox)*1e-3,(sgxy1[:,3]-oy)*1e-3,facecolor='none',edgecolor='b',label='receiver')
plt.xlabel('X (km)')
plt.ylabel('Y (km)')
plt.xlim([-10,25])
plt.ylim([13,16.5])
plt.legend(loc=1)
plt.savefig('../Fig/geom1.pdf',bbox_inches='tight')
#plt.show()
