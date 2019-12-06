#!/usr/bin/env python

import numpy as np
from seppyio import *
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 15})

#objfunc=[1.593009e+08,1.378206e+08,1.375341e+08,1.375149e+08,1.375149e+08,1.375149e+08,1.375149e+08,1.375149e+08,1.375149e+08,1.375149e+08]
objfunc=get_array('objfunc')
objfunc=objfunc/np.amax(objfunc)

plt.plot(range(objfunc.shape[0]),objfunc)
plt.xticks(np.arange(objfunc.shape[0]))
temp=searchArgv('xlabel')
if temp!=False: 
    xlabel=temp
    plt.xlabel(xlabel)

temp=searchArgv('ylabel')
if temp!=False: 
    ylabel=temp
    plt.ylabel(ylabel)

temp=searchArgv('title')
if temp==False: 
    title=''
else:
    title=temp
plt.title(title)

temp=searchArgv('out')
if temp==False: 
    plt.show()
else:
    plt.savefig(temp,bbox_inches='tight')

